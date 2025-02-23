#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["NCCL_P2P_DISABLE"] = '1'
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["WANDB_SILENT"] = "true"
os.environ["WANDB__SERVICE_WAIT"] = "300"

import numpy as np
import argparse
import time
# import tensorflow as tf
import torch
from ncp import NeuralClustering
from data_generator import get_generator
from utils import *
from geweke_test import geweke_test_histogram, geweke_test_multiple_N
from params import get_parameters
from evaluation import *
from plot_histogram import data_invariance_metric
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
    
    seed = args.seed
    set_seed(seed)

    wnb = init_wandb(args, params)
    
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
    wnb.log({'it': start_it}, step=start_it)
    
    # Initialize dictionary for eval stats:
    stats = {'NMI_max': 0, 'ARI_max': 0, 'ACC_max': 0, 'LL_max': -float('Inf'), 'MC_min': float('Inf'), 
             'NMI_max_it': 0, 'ARI_max_it': 0, 'ACC_max_it': 0, 'LL_max_it': 0, 'MC_min_it': 0}


    if eval_best_model:
        print('Start evaluation of chosen model')
        
        dpmm.eval()
        
        with torch.no_grad():
        
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            
            # # Compute NMI, ARI, MC_test:
            # if params['eval_it'] != -1:
            #     for i in params['eval_it']:
            #         checkpoint_path = os.path.join('saved_models/', datasetname, 'checkpoints', 'checkpoint_' + str(i) + '.pth')
            #         state = restore_checkpoint(checkpoint_path, state, params['device'])
            #         print('\nRestore model from iteration:', state['step'])
            #         M = (dataset_test_size//batch_size)*2
            #         stats = eval_stats(wnb, data_generator, batch_size, params, dpmm, it, stats, M=M)           
            # else:
            #     M = (dataset_test_size//batch_size)*2
            #     stats = eval_stats(wnb, data_generator, batch_size, params, dpmm, it, stats, M=M)             

            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            
            # # Compute NMI using Beam Search:
            # M = 1
            # stats = eval_stats_Beam_Search(wnb, data_generator, batch_size, params, dpmm, it, stats, M=M)
            
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            
            # # Get histogram of invariance metric:
            # perms = 500
            # Z = 3
            # fig, plt = data_invariance_metric(data_generator, dpmm, perms=perms, Z=Z)
            # image = wandb.Image(fig)
            # wnb.log({f"Plots_invariance_hist/hist_invariance_{it}": image}, step=it)
            # plt.clf()
            
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            
            # # Run Geweke's Test:
            # if not os.path.exists('output/geweke/'):
            #     os.makedirs('output/geweke/', exist_ok=True)
                
            # fig1, plt1 = geweke_test_histogram(dpmm, data_generator, params)
            # image = wandb.Image(fig1)
            # wnb.log({f"Plots_Geweke/Geweke_histogram": image}, step=it)
            # plt1.clf()
                
            # fig2, plt2 = geweke_test_multiple_N(dpmm, data_generator, params)
            # image = wandb.Image(fig2)
            # wnb.log({f"Plots_Geweke/Geweke_multiple_N": image}, step=it)
            # plt2.clf()
            
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            
            # Compare qualitative results of order permutations:
            N = 20
            
            # ----- Extract dataset of one data point in different permutations, save it locally:
            data, cs_gt, _, _, _ = data_generator.generate(N=N, batch_size=1, train=False)  
            
            data_all = []
            cs_gt_all = []
            for i in range(64):
                # Prepare a permutation of the original data:
                arr = np.arange(N)
                np.random.shuffle(arr)   # permute the order in which the points are queried
                data_perm = data[:, arr, :]
                cs_perm = cs_gt[:, arr]
                data_all.append(torch.clone(data_perm))
                cs_gt_all.append(torch.clone(cs_perm))
                print(cs_perm)
            
            print('---')
            data_all = torch.cat(data_all, dim=0)
            cs_gt_all = torch.cat(cs_gt_all, dim=0)
            print(data_all.shape, cs_gt_all.shape)
            print(cs_gt_all)
            
            torch.save(data_all, 'data_perms.pt') 
            torch.save(cs_gt_all, 'cs_gt_perms.pt')
            1/0
        
            #  ----- Perform the test:
            data = torch.load('data_perms.pt')  # data: [64, N, 28, 28] or [64, N, 2]
            cs_gt = torch.load('cs_gt_perms.pt')  # cs_gt: [64, N]
            plot_data_perm_for_paper(data, cs_gt, params, dpmm, it, N=N)
        
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            
        dpmm.train()
        
        return

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
        if plot_freq != -1 and it % plot_freq == 0:
            # print('\nPloting samples, compute NMI, ARI, LL, iteration ' + str(it) + '.. \n')   
            
            # NMI, ARI, LL.
            data, cs_gt, clusters, K = data_generator.generate(N=None, batch_size=batch_size, train=False)  # data: [1, N, 2] or [1, N_sampling, 28, 28] or [1, N, 3, 28, 28]
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
        
        # DEBUG: HOW AUGMENTED IMAGES LOOK LIKE
        # data, cs, clusters, K = data_generator.generate(N=20, batch_size=1, train=True, unsup=True)  # data: [1, N, 2] or [1, N, 28, 28] or [1, N, 3, 28, 28]            
        # plot_samples_and_histogram(wnb, data, cs[0, :], params, dpmm, it, N=20, show_histogram=show_histogram)
        
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
        wnb = wandb.init(entity='bgu_cs_vil', project="NCP_EB", name=args.experiment, config=args, settings=wandb.Settings(_service_wait=300))
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
    parser.add_argument('--experiment', default='', type=str, metavar='NAME',
                   help='name of wandb experiment')   
        
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    if not args.load_model and not args.run_geweke and not args.eval_best_model:
        # EXPECT EMPTY FOLDER CALLED saved_model:
        model_dir = 'saved_models'
        if not os.path.exists(model_dir):
            raise NameError('No saved_models folder, please create it manually.')
        elif len(os.listdir(model_dir)) != 0:
            raise NameError('The saved_models folder contains checkpoints, but load_model flag is off.')

    if os.path.exists('wandb'): 
        shutil.rmtree('wandb')
    
    main(args)

