#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["NCCL_P2P_DISABLE"] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["WANDB_SILENT"] = "true"

import numpy as np
import argparse
import time
# import tensorflow as tf
import torch
from ncp import NeuralClustering
from ncp_old import NeuralClustering_old
from data_generators import get_generator
from utils import *
from params import get_parameters
from losses import kl_loss_func, j_loss_func, mc_loss_func
from evaluation import sample_periodically, sample_periodically_old
import shutil
from collections import OrderedDict
import random

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False
    

def main(args):

    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)
        
    datasetname = args.dataset
    params = get_parameters(datasetname)
    params['device'] = torch.device("cuda:0" if args.cuda else "cpu") 
    params['dataset_name'] = datasetname
    
    seed = args.seed
    set_seed(seed)

    wnb = init_wandb(args, params)
    
    batch_size = params['batch_size']
    loss_str = params['loss_str']
    is_old_kl = params['is_old_kl']
    max_it = params['max_it']
    epochs = 1
    lr = params['lr']
    min_lr = params['min_lr']
    num_sched_steps_per_epoch = max_it // params['sched_lr_update']
    weight_decay = params['weight_decay']
    weight_decay_end = params['weight_decay_end']
    device = params['device']
    N_sampling = args.N_sampling
    show_histogram = args.show_histogram  # A flag for analyzing a trained model (histogram)
    lambda_mc = params['lambda_mc']
    lambda_j = params['lambda_j']
    lambda_entrpy = params['lambda_entrpy']
    mc_weights = params['mc_weights']
    j_weights = params['j_weights']
    plot_freq = params['plot_freq']
    
    # Define the model:
    dpmm = NeuralClustering(params).to(params['device'])
    dpmm_old = NeuralClustering_old(params).to(params['device'])
    
    # Define the data generator:
    data_generator = get_generator(params)
    
    # Define learning rate and optimizers:
    optimizer = torch.optim.Adam(dpmm.parameters() , lr=lr, weight_decay=weight_decay)
    optimizer_old = torch.optim.Adam(dpmm_old.parameters() , lr=lr, weight_decay=weight_decay)
    
    
    # Define lr and weight decay schedulers:
    print("Use Cosine LR scheduler")
    lr_schedule_values = cosine_scheduler(
        lr, min_lr, epochs, num_sched_steps_per_epoch,
        warmup_epochs=0, warmup_steps=0,)
    
    print(lr_schedule_values)
    
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
    # tf.io.gfile.makedirs(checkpoint_dir)
    # tf.io.gfile.makedirs(os.path.dirname(checkpoint_meta_dir))

    # Load the trained model if required:
    state = restore_checkpoint(checkpoint_meta_dir, state, params['device'])
    print('\nRestore model from iteration:', state['step'])
    print('Learning rate:', params['lr'])
    
    
    # -------------------------------------
    #         Main training loop:
    # -------------------------------------
    
    # data_0, cs_0, clusters_0, K_0 = data_generator.generate(None, batch_size)    
    # N = data_0.shape[1]
    
    while it < max_it:
                
        dpmm.train()  # new model
        if is_old_kl:
            dpmm_old.train()  # old model (NCP from Ari's paper)
        
        # Update learning rate & weight decay:
        if (lr_schedule_values is not None or wd_schedule_values is not None) and it % params['sched_lr_update'] == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it_lr_sched]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it_lr_sched]
            
            if is_old_kl:
                for i, param_group in enumerate(optimizer_old.param_groups):
                    if lr_schedule_values is not None:
                        param_group["lr"] = lr_schedule_values[it_lr_sched]
                    if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                        param_group["weight_decay"] = wd_schedule_values[it_lr_sched]
            
            it_lr_sched = it_lr_sched + 1
            
        # Plot samples periodically: 
        if it % plot_freq == 0 or it == 1:
            print('\nPloting samples, compute NMI and test loss, iteration ' + str(it) + '.. \n')   
            # sample_periodically(None, None, params, dpmm, it, data_generator, N_sampling, show_histogram)
            data, cs_gt, clusters, K = data_generator.generate(N = N_sampling, batch_size=1)  # data: [1, N, 2] or [1, N_sampling, 28, 28] or [1, N, 3, 28, 28]
            sample_periodically(wnb, data, cs_gt, params, dpmm, it, N_sampling, show_histogram)
            if is_old_kl:
                sample_periodically_old(wnb, data, cs_gt, params, dpmm_old, it, N_sampling, show_histogram)
   
        # Save the model periodically:
        if it % 100 == 0:
            print('\Saving model.. \n') 
            save_model(state, it, dpmm, optimizer, checkpoint_dir, checkpoint_meta_dir)
  
        # Generate one batch for training
        data, cs, clusters, K = data_generator.generate(None, batch_size)    
        cs = relabel(cs)    # this makes cluster labels appear in cs[] in increasing order
        N = data.shape[1]
        
        # Display adaptive learning rate
        curr_lr = optimizer.param_groups[0]['lr']
        # print(optimizer.param_groups[0]['lr'])
            
        # print('Iteration:' + str(it) + ' N:', str(N) + ' Current lr:' + format(curr_lr, '.7f'))
        print('Iteration: {0} N: {1} Current lr: {2:.10f}'.format(it, N, curr_lr))
        
        # Saves the cs values computed during training, for NMI computation
        cs_train = np.zeros((batch_size, N))
        if is_old_kl:  
            cs_train_old = np.zeros((batch_size, N)) 
        
        # Training of one point: FW and Backprop of one batch.
        # (Each training step includes a few permutations of the data order)   
        dpmm.train()
        if is_old_kl:
            dpmm_old.train()    
        
        dpmm.encode(data)
        # dpmm_old.previous_n = 0   
        
        kl_loss = 0  
        kl_loss_old = 0
        mc_loss = 0
        log_pn = 0
        
        # FW step (includes N-1 FW steps in the net):
        for n in range(1, N):  # n is the next point to be assigned. Points up to n-1 (in each point in the batch, so there are B * (n-1)) were already assigned.                              
            E, E_mask = dpmm(cs, n)   # E is [B, K + 1]. This is the unnormalized logprob of p(c_{0:n} | x).
            
            # -------- KL Loss: --------------------
            # Get the logprobs which is log p(c_n|c_1..c_n-1, x)
            logprobs_kl = kl_loss_func(E, E_mask)
            c = cs[n] # The ground-truth cluster of the n-th point (which is similar in all B datasets)
            cs_train[:, n] = np.argmax(logprobs_kl.clone().detach().to('cpu').numpy(), axis=1)  # [B, 1] compute this for NMI computation later.
            kl_loss -= logprobs_kl[:, c].mean() # The loss is minus the value in the relevant cluster assignment in logprobs.
            # -------- END KL Loss: ----------------
            
            
            # # -------- J Loss: --------------------
            if n == N - 1:
                j_loss, data_E, entrpy = j_loss_func(dpmm, E, N, n, cs, it)  # scalar
            # # -------- END J Loss: ----------------
            
            
            # -------- MC Loss: --------------------
            mc_n_term, log_pn = mc_loss_func(E, E_mask, log_pn, n, N, cs, batch_size, it, device, epsilon=1e-5)
            mc_loss += mc_n_term
            # -------- END MC Loss: ----------------
            
            
            # -------- KL LOSS - OLD MODEL --------
            if is_old_kl:
                logprobs_kl_old  = dpmm_old(data, cs, n) # Shape: [B, K'+1]. This is the log probabilities of assigning the n-th point to the K' clusters we revealed so far, or to a new cluster.               
                c_old = cs[n] # The ground-truth cluster of the n-th point (which is similar in all B datasets)
                cs_train_old[:, n] = np.argmax(logprobs_kl_old.clone().detach().to('cpu').numpy(), axis=1)  # [B, 1]
                kl_loss_old -= logprobs_kl_old[:, c_old].mean() # The loss is minus the value in the relevant cluster assignment in logprobs.
            # ------- END KL LOSS - OLD MODEL --------


        mc_loss = mc_loss / N

        # Choose the loss:
        if loss_str == 'MC + J':
            loss = lambda_j * j_loss + lambda_mc * mc_loss - lambda_entrpy * entrpy
        elif loss_str == 'MC + J + KL':
            loss = lambda_j * j_loss + lambda_mc * mc_loss + kl_loss - lambda_entrpy * entrpy
        elif loss_str == 'J':
            loss = lambda_j * j_loss - lambda_entrpy * entrpy
        elif loss_str == 'MC':
            loss = lambda_mc * mc_loss
        elif loss_str == 'KL':
            loss = kl_loss
        
        if is_old_kl:
            loss_old = kl_loss_old

        loss.backward()    # this accumulates the gradients for each permutation
        if is_old_kl:
            loss_old.backward()    # this accumulates the gradients for each permutation
        
        NMI_val = compute_NMI(cs, cs_train)
        if is_old_kl:
            NMI_val_old = compute_NMI(cs, cs_train_old)
                
                
        # Store statistics in wandb:
        stats = OrderedDict(it=it)
        stats = update_stats_train(stats, N, K, loss, kl_loss, mc_loss, j_loss, entrpy, NMI_val)  # stats.update({'train_acc1': acc_train})
        wandb.log(stats, step=it)
        
        optimizer.step()      # the gradients used in this step are the sum of the gradients for each permutation 
        optimizer.zero_grad()    
        
        if is_old_kl:
            optimizer_old.step()      # the gradients used in this step are the sum of the gradients for each permutation 
            optimizer_old.zero_grad() 
        
        it += 1
        

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
        wnb = wandb.init(entity='bgu_cs_vil', project="NCP_EB", name=args.experiment, config=args)
        wnb.log_code(".")  # log source code of this run
        wnb.config.update(params)
    else:
        wnb = None
        print("Problem with initiating wandb.")
    
    return wnb
   
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Neural Clustering Process')

    parser.add_argument('--dataset', type=str, default='Gauss2D', metavar='S',
                    choices = ['Gauss2D','MNIST', 'FASHIONMNIST', 'CIFAR'],
                    help='Generative Model: Gauss2D or MNIST (default: Gauss2D)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=123, metavar='S',
                    help='random seed (default: 123)')
    parser.add_argument('--N-sampling', type=int, default=20, metavar='N',
                    help='N data points when sampling (default: 20)')
    parser.add_argument('--show-histogram', action='store_true', default=False,
                    help='flag for analyzing a trained model')
    parser.add_argument('--load-model', action='store_true', default=False,
                    help='flag for loading model or start from scratch')       
    parser.add_argument('--experiment', default='', type=str, metavar='NAME',
                   help='name of wandb experiment')   
        
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    if not args.load_model:
        # Remove saved models
        model_dir = 'saved_models'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)

        shutil.rmtree(model_dir)
    
    main(args)

