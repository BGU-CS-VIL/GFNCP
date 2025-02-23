#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
from torch.distributions import Categorical

class NCP_Sampler():
    
    
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
                assert data.shape[2] == model.params['x_dim']
                data = data.view([self.N, model.params['x_dim']])
                
            elif model.params['dataset_name'] == 'MNIST' or model.params['dataset_name'] == 'FASHIONMNIST':
                data = data.clone().detach().to(self.device)
                data = data.view([self.N, self.img_sz, self.img_sz])
            
            elif model.params['dataset_name'] == 'CIFAR':
                data = data.clone().detach().to(self.device)
                data = data.view([self.N, self.channels, self.img_sz, self.img_sz])
                
            self.hs = model.h(data)            
            self.qs = model.q(data)            
            self.f = model.f
            self.g = model.g
            
    def sample(self, S):

        #input S: number of samples
                 
        assert type(S) == int
        self.model.eval()
        cs = torch.zeros([S, self.N], dtype=torch.int64)  # Stores the S sampled assignments for data.
        previous_maxK = 1 
        nll = torch.zeros(S)
        
        with torch.no_grad():
            
            for n in range(1, self.N):
                
                Ks, _ = cs.max(dim=1)  # [S, 1]. Stores the previous K so far in each sampled assignment in cs.
                Ks += 1
                maxK  = Ks.max().item()
                minK  = Ks.min().item()
                
                inds = {}  # A list in the size of [(maxK-minK), S]. 
                           # Each inds[K] is in the size of S, and it contains "True" only in the entries of the assignments where this is the K in them. 
                           # Here we deal with the problem that among the S samples, the K found so far might be different.
                for K in range(minK, maxK + 1):
                    inds[K] = Ks==K
                    
                if n == 1:                
                    self.Q = self.qs[2:, :].sum(dim=0).unsqueeze(0)     #[1, q_dim]            
                    self.Hs = torch.zeros([S, 2, self.h_dim]).to(self.device)
                    self.Hs[:, 0, :] = self.hs[0, :]  
                else:            
                    if maxK > previous_maxK:            
                        new_h = torch.zeros([S, 1, self.h_dim]).to(self.device)
                        self.Hs = torch.cat((self.Hs, new_h), dim=1) 
   
                    self.Hs[np.arange(S), cs[:, n - 1], :] += self.hs[n - 1, :]
                    if n == self.N - 1:
                        self.Q = torch.zeros([1, self.h_dim]).to(self.device)    #[1, h_dim]
                    else:
                        self.Q[0, :] -= self.qs[n, :]
                        
                previous_maxK = maxK
                
                assert self.Hs.shape[1] == maxK + 1
                
                logprobs = torch.zeros([S, maxK + 1]).to(self.device)  # [S, K]. Stores the probabilities of the n-th point to be assiged to each cluster.
                rQ = self.Q.repeat(S, 1)

                for k in range(maxK + 1):
                    Hs2 = self.Hs.clone()
                    Hs2[:, k, :] += self.hs[n, :]                
                    Hs2 = Hs2.view([S * (maxK + 1), self.h_dim])                
                    gs = self.g(Hs2).view([S, (maxK + 1), self.g_dim])                
                    
                    # For each s in S the previous K might be different (because we sample from Cat distribution), so H and G rows might be different.
                    #   so here we fix it per row.
                    for K in range(minK, maxK + 1):
                        if k < K:
                            gs[inds[K], K:, :] = 0   
                        elif k == K and K < maxK:
                            gs[inds[K], (K + 1):, :] = 0   
              
                    Gk = gs.sum(dim=1)
                    uu = torch.cat((Gk, rQ), dim=1)
                    logprobs[:, k] = torch.squeeze(self.f(uu))    
                
                # For each s in S the previous K might be different (because we sample from Cat distribution), so logprobs rows might be different.
                #   so here we fix it per row.
                for K in range(minK, maxK):
                    logprobs[inds[K], K + 1:] = float('-Inf')
                
                # Normalize
                m,_ = torch.max(logprobs,1, keepdim=True)       
                logprobs = logprobs - m - torch.log( torch.exp(logprobs-m).sum(dim=1, keepdim=True))  # [S, K+1]
                
                probs = torch.exp(logprobs)  # [S, K+1]. Each row is the probs vector of the n-th point to be assigned to each cluster.                          
                m = Categorical(probs)
                ss = m.sample()  # [S,]. These are S samples of the cluster of the n-th point.
                cs[:, n] = ss 
                nll -= logprobs[np.arange(S), ss].to('cpu')  # Minus sum of all logprobs of the chosen clusters for all N points.
                   # Here, we can sample ss (cluster) that is not very likely, so we expect to see a small value for it in logprobs[np.arange(S), ss].
                
        cs = cs.numpy()  # [S, N]
        nll = nll.numpy() # [S,]
        
        sorted_nll =np.sort(list(set(nll)))    # sort the samples in order of increasing NLL
        Z = len(sorted_nll)                    # number of distinct samples among the S samples
        probs = np.exp(-sorted_nll)  # [Z,] This is the prop of the entire assignment of each sample.
        css = np.zeros([Z, self.N], dtype=np.int32) 
        
        # Organize the assignmnets according to the logprobs, after we took the distinct samples.
        for i in range(Z):
            snll= sorted_nll[i]
            r = np.nonzero(nll==snll)[0][0]
            css[i,:]= cs[r,:]
                
        return css, probs, nll.mean()

