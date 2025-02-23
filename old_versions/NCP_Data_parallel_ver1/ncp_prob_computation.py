#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
from utils import relabel



def compute_prob(dpmm, data, cs):
    '''
    cs: the most likely clustering, shape: [N,]. The labels are in increasing order.
                (Here we compute the probability to get this clustering for the given data order)
    '''
    
    logprob_sum = 0
    N = data.shape[1]
    
    cs = relabel(cs)    # this makes cluster labels appear in cs[] in increasing order        
    cs = torch.tensor(cs)
    cs = cs.repeat(data.shape[0], 1)  # [B, N] where all rows are the same
    
    for n in range(1, N):
        
        # Compute E:
        E_, G_mask = dpmm(data, cs, n)

        # In each row put -inf in columns that their index is higher than the K (found so far) of that row.
        E_ = E_.to(torch.float64)
        E = torch.where(G_mask == 0.0, float('Inf'), E_).to(torch.float32)
        # E = E[0, :]  # take the first result, as all rows are equal.
        # E = E[None, :]  # [1, K + 1]
        
        # Normalize to compute p(c_n | c_{0:n-1}, x) for each entry in S:
        m, _ = torch.min(E, 1, keepdim=True)       
        logprobs = - E + m - torch.log( torch.exp(-E + m).sum(dim=1, keepdim=True))  # [S, K + 1]. logprob of p(c_n | c_{0:n-1}, x)
        
        # Sum of all logprobs of the most likely assignment:
        logprob_sum -= logprobs[0, cs[0, n]].to('cpu')
        
    prob_sum = np.exp(- logprob_sum.numpy())
    
    return prob_sum


