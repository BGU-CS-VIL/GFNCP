#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
# import tensorflow as tf
import os
import logging
from sklearn.metrics.cluster import normalized_mutual_info_score as NMI
from sklearn.metrics.cluster import adjusted_rand_score as ARI
import numpy as np 
import math
from collections import OrderedDict
import random


def relabel(cs):
    cs = cs.copy()
    # cs = torch.clone(cs)
    d={}
    k=0
    for i in range(len(cs)):
        j = cs[i]
        if j not in d:
            d[j] = k
            k+=1
        cs[i] = d[j]        

    return cs


def restore_checkpoint(ckpt_dir, state, device):
  if not os.path.exists(ckpt_dir):
    # tf.io.gfile.makedirs(os.path.dirname(ckpt_dir))
    os.makedirs(os.path.dirname(ckpt_dir), exist_ok=True)
    logging.warning(f"\nNo checkpoint found at {ckpt_dir}. "
                    f"Returned the same state as input")
    return state
  else:
    loaded_state = torch.load(ckpt_dir, map_location=device)
    state['optimizer'].load_state_dict(loaded_state['optimizer'])
    state['model'].load_state_dict(loaded_state['model'], strict=False)
    state['step'] = loaded_state['step']
    return state


def save_checkpoint(ckpt_dir, state):
  saved_state = {
    'optimizer': state['optimizer'].state_dict(),
    'model': state['model'].state_dict(),
    'step': state['step']
  }
  torch.save(saved_state, ckpt_dir)


def save_model(state, it, dpmm, optimizer, checkpoint_dir, checkpoint_meta_dir):
    state['step'] = it
    state['model'] = dpmm
    state['optimizer'] = optimizer
    save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{it}.pth'), state)
    save_checkpoint(checkpoint_meta_dir, state)


def compute_NMI(cs_gt, cs_pred, probs):
    ''' cs_gt is [1, N]
        cs_pred is [M, N]
        probs is [M,]
    '''
    
    if torch.is_tensor(cs_gt):
        cs_gt = cs_gt.detach().cpu().numpy()
    
    if torch.is_tensor(cs_pred):
        cs_pred = cs_pred.detach().cpu().numpy()
    
    # Return NMI of the most likely sample:
    if probs is None:  # Here we compute NMI on train data and the average NMI is on a batch with different data groups.
        NMI_all = 0
        for i in range(cs_pred.shape[0]):
            NMI_all += NMI(cs_gt, cs_pred[i, :])
            
        return NMI_all / cs_pred.shape[0]
    
    else:  # Here we compute NMI on test data, and show the NMI we get from the most likely sample.
        return NMI(cs_gt, cs_pred[np.argmax(probs), :])
    

def compute_ARI(cs_gt, cs_pred, probs):
    ''' cs_gt is [1, N]
        cs_pred is [M, N]
        probs is [M,]
    '''
    
    if torch.is_tensor(cs_gt):
        cs_gt = cs_gt.detach().cpu().numpy()
    
    if torch.is_tensor(cs_pred):
        cs_pred = cs_pred.detach().cpu().numpy()

    # Return ARI of the most likely sample:
    if probs is None:  # Here we compute ARI on train data and the average ARI is on a batch with different data groups.
        ARI_all = 0
        for i in range(cs_pred.shape[0]):
            ARI_all += ARI(cs_gt, cs_pred[i, :])
            
        return ARI_all / cs_pred.shape[0]
    
    else:  # Here we compute ARI on test data, and show the ARI we get from the most likely sample.
        return ARI(cs_gt, cs_pred[np.argmax(probs), :])





def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule 


def update_stats_train(it, N, K, loss, kl_loss, mc_loss, j_loss, entrpy, NMI_train, ARI_train):
    stats = OrderedDict(it=it)
    stats.update({'N': N})
    stats.update({'K': K})
    stats.update({'loss': loss})
    stats.update({'kl_loss': kl_loss})
    stats.update({'mc_loss': mc_loss})
    stats.update({'j_loss': j_loss})
    stats.update({'entrpy': entrpy})
    stats.update({'NMI_train': NMI_train})
    stats.update({'ARI_train': ARI_train})
    return stats



def get_argmax_from_probs(probs):
    # probs is a tensor of shape [B, K+1]
    # Get the argmax of probs. In case there is more than one occurrence of the max value, sample from the options.
    
    probs_ = probs.clone().detach()
    argmax = torch.zeros(probs_.shape[0])
    
    for i in range(probs_.shape[0]):
        max_occ = (probs_[i, :] == torch.max(probs_[i, :])).nonzero()   # numpy version: max_occ = np.argwhere(listy == np.amax(listy))
        argmax[i] = random.sample(list(max_occ), 1)[0]

    ss = argmax.to(torch.int32)

    # # Or, get the last occurrence of the argmax result:
    # probs_f = torch.flip(probs_, dims=(1,))
    # ss = probs_f.shape[1] - torch.argmax(probs_f, dim=1) - 1

    return ss


# # This function will be called on test data (using model.eval()), preferably when the learning converges.
# def compute_Geweke_test(N, alpha, S, dpmm):
#     # For a given N and alpha:
#     #    - Compute the ground-truth multinomial distribution over K (when sampled from CRP in the data loader)
#     #    - Compute the 
    
#     Kmax = N
    
#     # In these tensors: value in entry i is the number of times we got i clusters:
#     gt_mltn_samples = torch.zeros(S, Kmax) 
#     pred_mltn_samples = torch.zeros(S, Kmax)
    
#     for i in range(S):
#         # Multinomial sample from CRP   
#         cs = ? # [N,]
#         cs_cnts = # [Kmax,]
#         gt_mltn_samples[i, :] = cs_cnts
        
#         # Multinomial sample from the model (after getting the data related to the above cs), use batch_sz=1
#         pred_cs = ?    # Get the first row
#         pred_cs_cnts = # [Kmax,]
#         pred_mltn_samples[i, :] = pred_cs_cnts
        
#     # Estimate the multinomial distributions (probabilities to get each K):
#     #   (End up with one vector for gt and one for predicted)
#     gt_mltn_dist = ?  # [Kmax,]
#     pred_mltn_dist = ?  # [Kmax,]
    
#     # Compute KL divergance between the two polynomial distributions
    
#     # Display the two distributions as histograms
    
