#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from models.backbone_encoders import get_encoder
from models.backbones import get_backbone
import gc
# from scipy.stats import entropy

import os
import psutil




class NeuralClustering(nn.Module):
    
    def __init__(self, params):
        super(NeuralClustering, self).__init__()
        
        self.params = params
        self.previous_n = 0
        self.previous_K = 1
        self.g_dim = params['g_dim']
        self.h_dim = params['h_dim']
        H = params['H_dim']        
        self.device = params['device']
        self.img_sz = params['img_sz']

        # Prepare the relevant encoder for the input data:
        self.h = get_encoder(params)   # E.g: Input: [B * N, 28, 28], Output: [B * N, h]       
        self.q = get_encoder(params)   # E.g: Input: [B * N, 28, 28], Output: [B * N, h] 
        # self.hs = None
        # self.qs = None
        
        self.agg_clustered = AggregateClusteredSum(params)
        self.agg_unclustered = AggregateUnclusteredSum(params)
        
        # Input: [B, h_dim]
        # Output: [B, g_dim]
        self.g = get_backbone(params, input_dim=self.h_dim, output_dim=self.g_dim)
        
        # Input: [B, h_dim + g_dim]
        # Output: [B, 1]
        self.E = get_backbone(params, input_dim=self.g_dim + self.h_dim, output_dim=1, last_layer_bias=False)


    def sample(self, data, it):
        B, N = data.shape[0], data.shape[1]
        
        c_samples = torch.zeros([B, N], dtype = torch.int32)        
        E_samples = torch.zeros(B, N - 1)
        fake_E = None
        entrpy = 0
        
        data = data.to(torch.device('cuda'))
        c_samples = c_samples.to(torch.device('cuda'))
        
        for n in range(1, N):
            E_, G_mask = self.forward(data, c_samples, n, cs_is_sample_flag=True)   # E, G_mask: [B, K + 1]  

            # Get the min value per row, but before: in each row put -inf in columns that their index is higher than the K (found so far) of that row.
            #  (otherwise m takes values that are irrelevant (min value might be in cells where G_mask is 0, which is bad.)).
            #  Revert E to be with zeros after computing m.
            E_ = E_.to(torch.float64)
            E = torch.where(G_mask == 0.0, float('Inf'), E_).to(torch.float32)  
            
            m, _ = torch.min(E, 1, keepdim=True)    # [B, 1]      
            # E = torch.where(G_mask == 0.0, 0.0, E_).to(torch.float32)  # return E to be without Infs
            
            probs_un = torch.exp(- E + m) * G_mask  # [B, K + 1]
            
            c_sample_n = torch.multinomial(probs_un, num_samples=1).squeeze()  # [B, 1]. These are assignment of the n-th point to one of the clusters, for each row in the batch.
            c_samples[:, n] = c_sample_n 
            E_samples[:, n - 1] = E[range(B), c_sample_n]  # it's N-1 because we don't need the value of c_0.

            if n == N - 1:
               fake_E = - (- E[:, c_sample_n] + m - torch.log( torch.exp(- E + m).sum(dim=1, keepdim=True)))  # [B, 1]. logprob of p(c_N | c_{0:N-1}, x) using the sampled label for the N-th point.
           
        return E_samples[:, N - 2]  # , entrpy  # [B,]  # fake_E  # 

    # Here we use E the same way as we used it when learning KL loss:
    def sample_for_kl_eval(self, data):
        S, N = data.shape[0], data.shape[1]
        c_samples = torch.zeros([S, N], dtype = torch.int32)     
        nll = torch.zeros(S)
        
        data = data.to(torch.device('cuda'))
        c_samples = c_samples.to(torch.device('cuda'))
        
        for n in range(1, N):
            E_, G_mask = self.forward(data, c_samples, n, cs_is_sample_flag=True)   # E, G_mask: [S, K + 1]  (this is [S, max_K])
                        
            # In each row put -inf in columns that their index is higher than the K (found so far) of that row.
            E_ = E_.to(torch.float64)
            # E = torch.where(G_mask == 0.0, float('-Inf'), E_).to(torch.float32)
            E = torch.where(G_mask == 0.0, float('Inf'), E_).to(torch.float32)

            # Normalize to compute p(c_n | c_{0:n-1}, x) for each entry in S:
            m, _ = torch.min(E, 1, keepdim=True)       
            logprobs = - E + m - torch.log( torch.exp(-E + m).sum(dim=1, keepdim=True))  # [S, K + 1]. logprob of p(c_n | c_{0:n-1}, x)
 
            probs = torch.exp(logprobs)   # [S, K + 1]
            # print(probs.sum(dim=1))
    
            # Sample and compute negative LL: 
            ss = Categorical(probs).sample()   # [S, 1]. These are assignment of the n-th point to one of the clusters, for each row in the batch.                          
            # c_sample_n = torch.multinomial(probs, num_samples=1).squeeze()  # [S, 1]. These are assignment of the n-th point to one of the clusters, for each row in the batch.                          
            c_samples[:, n] = ss
            nll -= logprobs[np.arange(S), ss].to('cpu')   # [S,]. This is the log likelihood to compute the probability of the entire assignmet.

        c_samples = c_samples.cpu().numpy()  # [S, N]
        nll = nll.clone().detach().numpy() # [S,]
        
        sorted_nll = np.sort(list(set(nll)))    # sort the samples in order of increasing LL
        Z = len(sorted_nll)                    # number of distinct samples among the S samples
        probs = np.exp(-sorted_nll)  # [M,] This is the prop of the entire assignment of each sample.
        css = np.zeros([Z, N], dtype=np.int32) 
        
        # Organize the assignmnets according to the logprobs, after we took the distinct samples.
        for i in range(Z):
            snll= sorted_nll[i]
            r = np.nonzero(nll==snll)[0][0]
            css[i,:]= c_samples[r,:]
            
        probs = np.exp(-sorted_nll)  # [M,] This is the prop of the entire assignments of each sample.
        
        return css, probs, -nll.mean()  # c_samples: [M, N], prob_samples: [M,] where M is the number of succeeded (unique) samples.
    
    
    def forward(self, data, cs, n, cs_is_sample_flag=False):
        ''' 
            Here we compute the energy (flow) of point n being assigned to each existing/new cluster. Main for runs on n=1...N-1.
            Points n+1,...,N-1 are not assigned yet.
            
            Input:
                data: [B, N, data_shape]. Data to be clustered.
                cs: [B, N]. Data assignments. During training, all B row are the same ground-truth label assignment. During sampling, each row is a sampled assignment for the relevant group in the batch (Labels appear in increasing order).
                n: scalar. The index of the point to be assigned.
                cs_is_sample_flag: True if we call forward during sampling.
                
            Output:
                E: [B, K + 1]. Holds the energy (flow) of the n-th point assigned to each k option. This is the unnormalized logprob of p(c_{0:n} | x).
                G_mask: [B, K + 1]. Holds the 
                
            More:  
                self.hs: [B, N, h_dim]. Holds the data after encoding it from x_dim to h_dim.          
                Q: [B, h]. Holds the sum of all unassigned datapoints (after converting each point from x_dim to h_dim).
                G: [B, K + 1, g_dim]. For each k, holds the sum of all K values of g(H_k), where the n-th point was assigned to cluster k. 
        '''     
        print('-----n ', n, ' data ', data.device, '\n')
        B = data.shape[0]
        N = data.shape[1]
        # data = data.to(torch.device('cuda'))
        
        data = data.view(tuple((-1,)) + tuple(data.shape[2:])) # will be: [B*N, channels, img_sz, img_sz] or [B*N, img_sz, img_sz] or [B*N, h_dim]

        # Encode data:
        if n == 1:
            self.hs = self.h(data).view([B, N, self.h_dim])  # [B, N, h_dim]   
            self.qs = self.q(data).view([B, N, self.h_dim])  # [B, N, h_dim]

        print('-----hs ', self.hs.device, '\n')
        
        assert n > 0
        K = cs[:, :n].max().item() + 1   
             
        G, G_mask = self.agg_clustered(self.hs, cs, n, self.g, cs_is_sample=cs_is_sample_flag)  # G: [B, K + 1, g_dim]; G_mask: [B, K + 1]
        Q = self.agg_unclustered(self.qs, n) # [B, h_dim]
        Q = Q[:, None, :].expand([B, K + 1, self.h_dim])  # [B, K + 1, h_dim]. Here we repeat h_dim for K+1 times (for eavh element in the batch)
        uu = torch.cat((G, Q), dim=2)  # [B, K + 1, g_dim + h_dim] . Prepare argument for the call to E()
        uu = uu.view([B * (K + 1), self.g_dim + self.h_dim])        
        E = self.E(uu).view([B, K + 1]) # [B, K + 1]. This is the unnormalized logprob of p(c_{0:n} | x).
        
        return E, G_mask
 


class AggregateClusteredSum(nn.Module):
    
    def __init__(self, params):    
        super(AggregateClusteredSum, self).__init__()
        self.h_dim = params['h_dim']
        self.g_dim = params['g_dim']
        self.device = params['device']

    def forward(self, hs, cs_o, n, g, cs_is_sample=False):
        
        '''
            Here we get the data (hs) and the n-th point, and compute H_k and G_k for each k option (and also the relevent mask).
            
            Input:
                hs: [B, N, h_dim]. Holds the data after encoding it from x_dim to h_dim.          
                cs_o: [B, N]. Data assignments. During training, all B row are the same ground-truth label assignment. During sampling, each row is a sampled assignment for the relevant group in the batch (Labels appear in increasing order).
                n: scalar. The index of the point to be assigned.
                cs_is_sample_flag: True if we call forward during sampling. 
                
            Output: 
                G: [B, K + 1, g_dim]. For each k, holds the sum of all K values of g(H_k), where the n-th point was assigned to cluster k. 
                G_mask: [B, K + 1]. Holds for each entry in the batch, a vector with 1 only in indices 0,...,k_b where k_b is the k found for this entry. Otherwise the value is 0. 
        '''
        
        cs = cs_o.clone()
        cs[:, n:] = -1
            
        K = int(cs.max()) + 1   # This is the current K value until now. It's the number of clusters and not the max index (if cs has 0,1,2 then K is 3)
        batch_size = hs.shape[0]

        # Check if cs is ground-truth or a sample.
        if cs_is_sample:                   
            many_cs_patterns = True
            Ks, _ = cs.max(dim=1) # [B,]. Holds the max K in each element in the batch.
        else:
            many_cs_patterns = False
            cs = cs[0, :][None, :]

        H = torch.zeros([batch_size, 2 * K + 1, self.h_dim]).to(hs.device)    
        G = torch.zeros([batch_size, K + 1, self.g_dim]).to(hs.device)  # Holds the G_k values, each column k holds the G_k.       
        
        if many_cs_patterns:
            # gs_mask allows us to transfer the spasity pattern of H  to gs G
            gs_mask = torch.zeros([batch_size, 2 * K + 1, 1]).to(hs.device)
            # In each row in gs_mask, in the column k, the value is 1 if this k appear in cs (in the relevant row in the batch), and 0 otherwise.
        
        # Here we compute H until the n-th point, including:
        for k in range(K):   
            mask = 1.0 * (cs == k).to(hs.device)   # [B, N], in each element in B, there will be 1 in the entries where the label == k and 0 otherwise.
            mask = mask[:, :, None]  # [B, N, 1]
            
            H[:, k, :] = (hs * mask).sum(1)  # Sum the values 0...n-1 in hs into the k group according to cs assignments.           
            H[:, k + K, :] = H[:, k, :] + hs[:, n, :]  # Add the n-th point in hs to the H sum of each k option.

            if many_cs_patterns:
                gs_mask[:, k] = torch.any((cs == k), dim=1, keepdim=True) * 1.0
                gs_mask[:, k + K] = gs_mask[:, k]

        H[:, 2 * K, :] =  hs[:, n, :]  # Add the hs of the n-th point to the H sum of the K+1 option.
        
        if many_cs_patterns:
            gs_mask[:, 2 * K] = 1.0
        
        # Run self.g on H values:
        H = H.view([batch_size * (2 * K + 1), self.h_dim])
        gs = g(H).view([batch_size, 2 * K + 1, self.g_dim])  # [B, 2 * K + 1, g_dim]
        
        if many_cs_patterns:
            gs = gs_mask * gs  # Put zero in gs in places where the column index (=k) does not appear in the row.
            # For example: if row number 1 has only clusters 0,1 and doesn't have cluster 2, then in this row in gs, cloumn 2 will be zero.
        
        # Here we compute G until the n-th point, including:
        # Sum all K+1 groups in H for each k option:
        for k in range(K):     
            inds = torch.tensor([True] * K + [False] * (K + 1))  # Create array of: [True, ..., False, ...] where True appear K times, followed by K+1 Falses.       
            inds[k] = False
            inds[k + K] = True
            G[:, k, :] = gs[:, inds, :].sum(1)
        
        inds = torch.tensor([True] * K + [False] * K + [True])
        G[:, K, :] =  gs[:, inds, :].sum(1)                        
        
        if many_cs_patterns:
            G[:, :K] = G[:, :K] * gs_mask[:, :K]
        
        G_mask = torch.ones([batch_size, K + 1]).to(hs.device)
        
        # Here we arrange G: move the "K+1" value to be closer to the rest of the values, and put 0 starting from K+2 until the end.
        if many_cs_patterns:
            for k in range(K - 1):            
                which_k = (Ks == k)  # row indices in which k is the max k.                      
                G[which_k, k + 1, :] = G[which_k, -1, :]
                G[which_k, -1, :] = 0                
                G_mask[which_k, k + 2:] = 0
                
        return G, G_mask
        
    
    
class AggregateUnclusteredSum(nn.Module):           
    def __init__(self, params):    
        super(AggregateUnclusteredSum, self).__init__()

    def forward(self, qs, n):        
        Q = qs[:, n + 1:, ].sum(dim=1)  # [B, h_dim]   /// Check with Ari: it used to be: Q = hs[:, n:, ].sum(dim=1)  # [B, h_dim]        
        return Q
    