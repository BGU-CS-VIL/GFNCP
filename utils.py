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
from PIL import ImageFilter, ImageOps
import torch.nn as nn



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


# def compute_ACC(cs_gt, cs_pred):
#     ''' cs_gt is [1, N]
#         cs_pred is [M, N]
#     '''
#     cs_gt = cs_gt.repeat(cs_pred.shape[0], 1)  # [M, N] where all rows are the same

#     correct = (torch.argmax(y_pred, dim=1) == y).type(torch.FloatTensor)
    
#     correct = (torch.argmax(cs_gt, dim=1) == cs_pred).type(torch.FloatTensor)
#     acc = correct.mean() * 100.
#     return acc.detach().cpu().numpy()

    

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


def update_stats_train(it, N, K, loss, entrpy, NMI_train, ARI_train):
    stats = OrderedDict(it=it)
    stats.update({'N': N})
    stats.update({'K': K})
    stats.update({'loss': loss})
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



# ---- Augmentation methods: ----

class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )
        
        
class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class DropOutFeatures(object):
    """
    Apply dropout on a features vector
    """
    def __init__(self, p):
        self.dropout = nn.Dropout(p=p)

    def __call__(self, features_vec):
        return self.dropout(features_vec)


class GaussianNoise(object):
    """
    Apply Gaussian noise to a features vector
    """
    def __init__(self, std=0.05):
        self.normal_dist = torch.distributions.Normal(loc=torch.tensor([0.]), scale=torch.tensor([std]))

    def __call__(self, features_vec):
        n = self.normal_dist.sample((features_vec.view(-1).size())).reshape(features_vec.size())
        out = features_vec + n
        return out
       
        

    
