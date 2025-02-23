#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
from torch.distributions import Categorical
from utils import relabel


class NCP_prob_computation_old():
    
    
    def __init__(self, model, data):
        
        self.model = model
        self.h_dim = model.params['h_dim']
        self.g_dim = model.params['g_dim']
        self.device = model.params['device']
        self.img_sz = model.params['img_sz']
        self.channels = model.params['channels']
        
        self.model.eval()
        
        with torch.no_grad():
            assert data.shape[0] == 1              
            self.N = data.shape[1]
            
            if model.params['dataset_name'] == 'Gauss2D':
                data = torch.tensor(data).float().to(self.device)
                assert data.shape[2] == model.params['x_dim']
                data = data.view([self.N, model.params['x_dim']])
                
            elif model.params['dataset_name'] == 'MNIST' or model.params['dataset_name'] == 'FASHIONMNIST':
                data = data.clone().detach().to(self.device)
                data = data.view([self.N, self.img_sz, self.img_sz])
            
            elif model.params['dataset_name'] == 'CIFAR':
                data = data.clone().detach().to(self.device)
                data = data.view([self.N, self.channels, self.img_sz, self.img_sz])
            
            self.hs = model.h(data)   # [N, h]          
            self.qs = model.q(data)   # [N, h]            
            self.f = model.f
            self.g = model.g
        
    def compute_prob(self, cs):
        '''
        cs: the most likely clustering, shape: [N,]. The labels are in increasing order.
                    (Here we compute the probability to get this clustering for the given data order)
        '''
        
        self.model.eval()
        previous_K = 1
        logprob_sum = 0
        cs = relabel(cs)    # this makes cluster labels appear in cs[] in increasing order
        
        with torch.no_grad():
        
            for n in range(1, self.N):
                K = len(set(cs[:n]))
                    
                if n == 1:                
                    # Prepare H_k: sum of all points within each cluster. (until point n-1, including)         
                    self.Hs = torch.zeros([2, self.h_dim]).to(self.device)   # We prepare 1 entry for the first cluster (of n=0) and another for the K+1 cluster
                    self.Hs[0, :] = self.hs[0, :]   # [2, h]. This is the H_k of the single cluster we revealed so far (point n==0 was assigned to it).
                    
                    # Prepare U: sum of all unassigned datapoints (n+1,...,N):
                    self.Q = self.qs[2:, :].sum(dim=0).unsqueeze(0)     # [q_dim] 
                else:          
                    # Prepare H_k: sum of all points within each cluster. (until point n-1, including)  
                    if K > previous_K:            
                        new_h = torch.zeros([1, self.h_dim]).to(self.device)
                        self.Hs = torch.cat((self.Hs, new_h), dim=0) # [K, h]
   
                    self.Hs[cs[n - 1], :] += self.hs[n - 1, :]
                    
                    # Prepare U: sum of all unassigned datapoints (n+1,...,N):
                    if n == self.N - 1:
                        self.Q = torch.zeros([1, self.h_dim]).to(self.device)    #[1, h_dim]
                    else:
                        self.Q -= self.qs[n, :]
                        
                previous_K = K
                
                assert self.Hs.shape[0] == K + 1
                
                logprobs = torch.zeros([K + 1]).to(self.device)  # [K + 1,]. Stores the probabilities of the n-th point to be assiged to each cluster.

                for k in range(K + 1):
                    Hs2 = self.Hs.clone()
                    Hs2[k, :] += self.hs[n, :]   # [K + 1, h_dim]             
                    gs = self.g(Hs2).view([K + 1, self.g_dim])   # [K + 1, g_dim]  
                    
                    if k < K:
                        gs[K:, :] = 0   
                                     
                    Gk = gs.sum(dim=0)   # [g_dim,] 
                    uu = torch.cat((Gk, torch.squeeze(self.Q)), dim=0)  # [g_dim + h_dim,]
                    logprobs[k] = torch.squeeze(self.f(uu))  # scalar.  
                
                # print('(compute) logprobs of n=:', n, ': ', logprobs)
                
                # Normalize
                m = torch.max(logprobs)   # scalar.      
                logprobs = logprobs - m - torch.log(torch.exp(logprobs - m).sum())   # [K + 1,]. 
                                        
                # Sum of all logprobs of the most likely assignment:
                logprob_sum -= logprobs[cs[n]].to('cpu')
        
            logprob_sum = logprob_sum.numpy()
            prob_sum = np.exp(-logprob_sum)
        
        return prob_sum


