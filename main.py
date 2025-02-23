#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["NCCL_P2P_DISABLE"] = '1'
os.environ["WANDB_SILENT"] = "true"
os.environ["WANDB__SERVICE_WAIT"] = "300"

import numpy as np
import argparse
import time
import torch
from ncp import NeuralClustering
from data_generator import get_generator
from utils import *
from geweke_test import geweke_test_histogram, geweke_test_multiple_N
from params import get_parameters
from evaluation import eval_stats, plot_samples_and_histogram
import shutil
from collections import OrderedDict
import random


try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False
    

def main(args):
    
    datasetname = args.dataset
    load_model = args.load_model
    eval_best_model = args.eval_best_model
    run_geweke = args.run_geweke
    params = get_parameters(datasetname)
    params['device'] = torch.device("cuda:0" if args.cuda else "cpu") 
    params['dataset_name'] = datasetname

    if args.data_path:
        params['data_path'] = args.data_path

    seed = args.seed
    set_seed(seed)

    if args.wandb:
        wnb = init_wandb(args, params)
    else:
        wnb = None
    
    batch_size = params['batch_size']
    loss_str = params['loss_str']
    max_it = params['max_it']
    epochs = 1
    num_sched_steps_per_epoch = max_it // params['sched_lr_update']
    lr = params['lr']
    min_lr = params['min_lr']
    weight_decay = params['weight_decay']
    weight_decay_end = params['weight_decay_end']
    device = params['device']
    show_histogram = args.show_histogram  # A flag for analyzing a trained model (histogram)
    plot_freq = params['plot_freq']
    save_model_freq = params['save_model_freq']
    unsup_flag = params['unsup_flag']

    # Define the model:
    #dpmm = NeuralClustering(params).to(params['device'])
    net = NeuralClustering(params)
    dpmm = torch.nn.DataParallel(net, device_ids=list(range(0, torch.cuda.device_count()))).to(torch.device('cuda'))

    # Define the data generator:
    data_generator = get_generator(params)
    dataset_test_size = data_generator.dataset_test_size
    
    # Define learning rate and optimizers:
    optimizer = torch.optim.Adam(dpmm.parameters() , lr=lr, weight_decay=weight_decay)

    # Define lr and weight decay schedulers:
    print("Use Cosine LR scheduler")
    lr_schedule_values = cosine_scheduler(
        lr, min_lr, epochs, num_sched_steps_per_epoch,
        warmup_epochs=0, warmup_steps=0,)
    
    print('lr_schedule_values size:', len(lr_schedule_values))
    print('Learning rate:', params['lr'])
    
    if weight_decay_end is None:
        weight_decay_end = weight_decay
    
    wd_schedule_values = cosine_scheduler(
        weight_decay, weight_decay_end, epochs, num_sched_steps_per_epoch)
    
    it = 0
    it_lr_sched = 0

    # Object that stores the model info for saving:
    state = dict(optimizer=optimizer, model=dpmm, step=0)
    
    # Create checkpoints directory
    checkpoint_dir = os.path.join('saved_models/', datasetname, 'checkpoints')
    # Intermediate checkpoints to resume training after pre-emption in cloud environments
    checkpoint_meta_dir = os.path.join('saved_models/', datasetname, 'checkpoints-meta', 'checkpoint.pth')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(checkpoint_meta_dir), exist_ok=True)

    # Load the trained model and set start_it, if required:
    if load_model or run_geweke or eval_best_model:
        state = restore_checkpoint(checkpoint_meta_dir, state, params['device'])
        start_it = state['step']
        print('\nRestore model from iteration:', state['step'])
    else:
        start_it = it
        
    # This line helps wnb to get updated with the iteration number when loading from checkpoint:       
    if args.wandb:
        wnb.log({'it': start_it}, step=start_it)
    
    # Initialize dictionary for eval stats:
    stats = {'NMI_max': 0, 'ARI_max': 0, 'ACC_max': 0, 'LL_max': -float('Inf'), 'MC_min': float('Inf'), 
             'NMI_max_it': 0, 'ARI_max_it': 0, 'ACC_max_it': 0, 'LL_max_it': 0, 'MC_min_it': 0}
    

    # ----------------------------------------------
    #              Main training loop:
    # ----------------------------------------------
    
    print('start_it:', start_it)
    print('max_it:', max_it)
    print('Start training.') 
    
    for it in range(start_it, max_it):
        
        dpmm.train()  # new model

        # Update learning rate & weight decay:
        if (lr_schedule_values is not None or wd_schedule_values is not None) and it % params['sched_lr_update'] == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it_lr_sched]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it_lr_sched]
            
            it_lr_sched = it_lr_sched + 1
            
        # Evaluate the model periodically:
        if plot_freq != -1 and it % plot_freq == 0 and args.wandb:
            # print('\nPloting samples, compute NMI, ARI, LL, iteration ' + str(it) + '.. \n')   
            
            # NMI, ARI, LL.
            # data, cs_gt, clusters, K = data_generator.generate(N=None, batch_size=batch_size, train=False)  # data: [1, N, 2] or [1, N_sampling, 28, 28] or [1, N, 3, 28, 28]
            stats = eval_stats(wnb, data_generator, batch_size, params, dpmm, it, stats, M=dataset_test_size//batch_size)
            
            # Plots. Here we must use N=20 because we need to plot the results:
            data, cs_gt, clusters, K, _ = data_generator.generate(N=20, batch_size=1, train=False)  # data: [1, N, 2] or [1, N, 28, 28] or [1, N, 3, 28, 28]            
            plot_samples_and_histogram(wnb, data, cs_gt[0, :], params, dpmm, it, N=20, show_histogram=show_histogram)
            
        # Save the model periodically:
        if it % save_model_freq == 0 and it > 1:
            print('\Saving model.. \n') 
            save_model(state, it, dpmm, optimizer, checkpoint_dir, checkpoint_meta_dir)
  
        # Generate one batch for training
        data, cs, clusters, K, uniform_c = data_generator.generate(N=None, batch_size=batch_size, train=True, unsup=unsup_flag)
        
        N = data.shape[1]
        
        # Training of one point: FW and Backprop of one batch.
        # (Each training step includes a few permutations of the data order)   
        dpmm.train()
        
        # Forward step (includes: 1 fw step of encode backbone + N fw steps of the main backbone)
        loss, logprob_sum, entrpy, cs_pred_train, K = dpmm(data, cs, it=it, uniform_c=uniform_c)
                
        # Average on outputs from all devices
        loss = loss.mean()
        entrpy = entrpy.mean()

        loss.backward()    # this accumulates the gradients for each permutation
        optimizer.step()      # the gradients used in this step are the sum of the gradients for each permutation 
        optimizer.zero_grad()    
        
        NMI_train = compute_NMI(cs[0, :], cs_pred_train, None)
        ARI_train = compute_ARI(cs[0, :], cs_pred_train, None)           
                
        # Store statistics in wandb:
        if args.wandb:
            sts = update_stats_train(it, N, K, loss, entrpy, NMI_train, ARI_train)  # stats.update({'train_acc1': acc_train})
            wandb.log(sts, step=it)

        if it % 10 == 0:
            print('\n(train) iteration: {0}, N: {1}, K: {2}, NMI_train: {3:.3f}, ARI_train: {4:.3f}, Loss: {5:.3f}'.format(it, N, int(K[0].detach().cpu().numpy()), NMI_train, ARI_train, loss))

        it += 1
        
    # Print avg metrics:
    print('\n\n * Best stats: \n',
            'Max NMI (test): {0:.3f} (on iter:) {1:.3f}'.format(stats['NMI_max'], stats['NMI_max_it']), '\n',
            'Max ARI (test): {0:.3f} (on iter:) {1:.3f}'.format(stats['ARI_max'], stats['ARI_max_it']), '\n',
            'Max LL (test): {0:.3f} (on iter:) {1:.3f}'.format(stats['LL_max'], stats['LL_max_it']), '\n',
            'Min MC (test): {0:.3f} (on iter:) {1:.3f}'.format(stats['MC_min'], stats['MC_min_it']), '\n')
    
    if args.wandb:
        wnb.finish()
    

def set_seed(seed):
    os.environ['PYTHONHASHSEED']=str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    
def init_wandb(args, params):
    if has_wandb:
        wnb = wandb.init(entity='bgu_cs_vil', project="NCP_EB", name='wnb_experiment', config=args, settings=wandb.Settings(_service_wait=300))
        wnb.log_code(".")  # log source code of this run
        wnb.config.update(params)
    else:
        wnb = None
        print("Problem with initiating wandb.")
    
    return wnb
   
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Neural Clustering Process')

    parser.add_argument('--dataset', type=str, default='Gauss2D', metavar='S',
                    choices = ['Gauss2D','MNIST', 'FASHIONMNIST', 'CIFAR', 'IN50_ftrs', 'IN100_ftrs', 'IN200_ftrs', 'CIFAR_ftrs', 'tinyimagenet'],
                    help='Generative Model: Gauss2D or MNIST (default: Gauss2D)')
    parser.add_argument('--data-path', type=str, default='', metavar='S',
                    help='path for dataset')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=123, metavar='S',
                    help='random seed (default: 123)')
    parser.add_argument('--show-histogram', action='store_true', default=False,
                    help='flag for analyzing a trained model')
    parser.add_argument('--load-model', action='store_true', default=False,
                    help='flag for loading model or start from scratch')    
    parser.add_argument('--eval-best-model', action='store_true', default=False,
                    help='flag for loading model or start from scratch')    
    parser.add_argument('--run-geweke', action='store_true', default=False,
                    help='flag for running Gewekes Test from a learned model')  
    parser.add_argument('--wandb', action='store_true', default=False,
                    help='flag for using wandb for logging data')
        
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    if not args.load_model:
        # Remove saved models
        model_dir = 'saved_models'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)

        shutil.rmtree(model_dir)
        
    if os.path.exists('wandb'): 
        shutil.rmtree('wandb')
    
    main(args)

