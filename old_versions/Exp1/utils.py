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


def relabel(cs):
    cs = cs.copy()
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


def compute_NMI(cs_gt, cs_test, probs):
    ''' cs_gt is [1, N]
        cs_test is [M, N]
        probs is [M,]
    '''
    
    # Return NMI of the most likely sample:
    if probs is None:  # Here we compute NMI on train data and the average NMI is on a batch with different data groups.
        NMI_all = 0
        for i in range(cs_test.shape[0]):
            NMI_all += NMI(cs_gt, cs_test[i, :])
            
        return NMI_all / cs_test.shape[0]
    
    else:  # Here we compute NMI on test data, and show the NMI we get from the most likely sample.
        return NMI(cs_gt, cs_test[np.argmax(probs), :])
    

def compute_ARI(cs_gt, cs_test, probs):
    ''' cs_gt is [1, N]
        cs_test is [M, N]
        probs is [M,]
    '''
    
    # Return ARI of the most likely sample:
    if probs is None:  # Here we compute ARI on train data and the average ARI is on a batch with different data groups.
        ARI_all = 0
        for i in range(cs_test.shape[0]):
            ARI_all += ARI(cs_gt, cs_test[i, :])
            
        return ARI_all / cs_test.shape[0]
    
    else:  # Here we compute ARI on test data, and show the ARI we get from the most likely sample.
        return ARI(cs_gt, cs_test[np.argmax(probs), :])





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


def update_stats_train_ncp(it, N, K, loss, kl_loss, mc_loss, j_loss, entrpy, NMI_train, NCP_NMI_train, loss_old):
    stats = OrderedDict(it=it)
    stats.update({'N': N})
    stats.update({'K': K})
    stats.update({'loss': loss})
    stats.update({'kl_loss': kl_loss})
    stats.update({'mc_loss': mc_loss})
    stats.update({'j_loss': j_loss})
    stats.update({'entrpy': entrpy})
    stats.update({'NMI_train': NMI_train})
    stats.update({'NCP_NMI_train': NCP_NMI_train})
    stats.update({'NCP_kl_loss': loss_old})
    return stats