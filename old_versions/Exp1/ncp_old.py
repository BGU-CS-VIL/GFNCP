#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F





from utils import relabel

class MNIST_encoder(nn.Module):
    
    def __init__(self, params):
        
        super(MNIST_encoder, self).__init__()
        
        
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 256)
        self.fc2 = nn.Linear(256, params['h_dim'])

    def forward(self, x):
        '''
            Input: [B * N, 28, 28]
            Output: [B * N, h]
        '''

        x = x.unsqueeze(1)   # add channel index
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class CIFAR_encoder(nn.Module):
    
    def __init__(self, params):
        
        super(CIFAR_encoder, self).__init__()
        
        
        self.conv1 = nn.Conv2d(params['channels'], 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 256)
        self.fc2 = nn.Linear(256, params['h_dim'])

    def forward(self, x):
        '''
            Input: [B * N, 3, 28, 28]
            Output: [B * N, h]
        '''

        # x = x.unsqueeze(1)   # add channel index
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
    


class Mixture_Gaussian_encoder(nn.Module):
    
    def __init__(self, params):
        
        super(Mixture_Gaussian_encoder, self).__init__()
        
        H = params['H_dim']
        self.h_dim = params['h_dim']        
        self.x_dim = params['x_dim']
        
        self.h = torch.nn.Sequential(
                torch.nn.Linear(self.x_dim, H),
                torch.nn.PReLU(),
                torch.nn.Linear(H, H),
                torch.nn.PReLU(),    
                torch.nn.Linear(H, H),
                torch.nn.PReLU(),
                torch.nn.Linear(H, H),
                torch.nn.PReLU(),
                torch.nn.Linear(H, self.h_dim),
                )

    def forward(self, x):
        
        return self.h(x)


class NeuralClustering_old(nn.Module):
    
    
    def __init__(self, params):
        
        super(NeuralClustering_old, self).__init__()
        
        self.params = params
        self.previous_n = 0
        self.previous_K=1
        
        self.g_dim = params['g_dim']
        self.h_dim = params['h_dim']  # Used as the dim of u in the paper.
        H = params['H_dim']        
        
        self.device = params['device']

        if self.params['dataset_name'] == 'Gauss2D':
            self.h = Mixture_Gaussian_encoder(params)         
            self.q = Mixture_Gaussian_encoder(params)         
        elif self.params['dataset_name'] == 'MNIST':
            self.h = MNIST_encoder(params)  # Input: [B * N, 28, 28], Output: [B * N, h]     
            self.q = MNIST_encoder(params)  # Input: [B * N, 28, 28], Output: [B * N, h] 
        elif self.params['dataset_name'] == 'FASHIONMNIST':
            self.h = MNIST_encoder(params)  # Input: [B * N, 28, 28], Output: [B * N, h]     
            self.q = MNIST_encoder(params)  # Input: [B * N, 28, 28], Output: [B * N, h]  
        elif self.params['dataset_name'] == 'CIFAR':
            self.h = CIFAR_encoder(params)  # Input: [B * N, 3, 28, 28], Output: [B * N, h]     
            self.q = CIFAR_encoder(params)  # Input: [B * N, 3, 28, 28], Output: [B * N, h]         
        else:
            raise NameError('Unknown dataset_name '+ self.params['dataset_name'])
    
        # Input: [B, h]
        # Output: [B, g]
        self.g = torch.nn.Sequential(
                torch.nn.Linear(self.h_dim, H),
                torch.nn.PReLU(),    
                torch.nn.Linear(H, H),
                torch.nn.PReLU(),
                torch.nn.Linear(H, H),
                torch.nn.PReLU(),    
                torch.nn.Linear(H, H),
                torch.nn.PReLU(),
                torch.nn.Linear(H, H),
                torch.nn.PReLU(),
                torch.nn.Linear(H, self.g_dim),
                )
        
        # Input: [B, h+g]
        # Output: [B, 1]
        self.f = torch.nn.Sequential(
                torch.nn.Linear(self.g_dim + self.h_dim, H),
                torch.nn.PReLU(),    
                torch.nn.Linear(H, H),                
                torch.nn.PReLU(),    
                torch.nn.Linear(H, H),
                torch.nn.PReLU(),
                torch.nn.Linear(H, H),
                torch.nn.PReLU(),
                torch.nn.Linear(H, H),
                torch.nn.PReLU(),
                torch.nn.Linear(H, 1, bias=False),
                )
        

    
    def forward(self, data, cs, n):
        ''' 
            Input:
            data: [B, N, 28, 28] or [B, N, 3, 28, 28]. A batch of B points, each point is a dataset of N points with the same mixture.
            cs: [N]. Ground-truth labels of data.
            n: scalar. The index of the point to be assigned.
            
            Output:
            logprobs: [B, K + 1]. Holds the log probability of the n-th point to be assigned to each cluster.

            More:            
            self.Hs: [B, K, h]. Holds the sum of all points (after converting each point from x_dim to h_dim) within each cluster.
            self.Q: [B, h]. Holds the sum of all unassigned datapoints (after converting each point from x_dim to h_dim).
            Gk: [B, g]. Holds the sum of all K values of g(H_k), where the n-th point was summed to cluster k. 
        '''
    
        # n = 1,2,3..N
        # elements with index below or equal to n-1 are already assigned
        # element with index n is to be assigned. 
        # the elements from the n+1-th are not assigned
        
        assert(n == self.previous_n + 1)
        self.previous_n = self.previous_n + 1 
        K = len(set(cs[:n]))  # num of already created clusters

        if n == 1:
            self.batch_size = data.shape[0]
            self.N = data.shape[1]
            assert (cs == relabel(cs)).all() 
            
            if self.params['dataset_name'] == 'Gauss2D':
                # The data comes as a numpy vector
                data = data.reshape([self.batch_size * self.N, self.params['x_dim']])

            elif self.params['dataset_name'] == 'MNIST' or self.params['dataset_name'] == 'FASHIONMNIST':
                # The data comes as a torch tensor, we just move it to the device 
                data = data.to(self.device)    
                data = data.view([self.batch_size*self.N, 28, 28])
            
            elif self.params['dataset_name'] == 'CIFAR':
                # The data comes as a torch tensor, we just move it to the device 
                data = data.to(self.device)    
                data = data.view([self.batch_size*self.N, 3, 28, 28])
                                    
            # Prepare H_k: sum of all points within each cluster. (until point n-1, including)
            self.hs = self.h(data).view([self.batch_size, self.N, self.h_dim])   # [B, N, h]       
            self.Hs = torch.zeros([self.batch_size, 1, self.h_dim]).to(self.device) # [B, 1, h]
            self.Hs[:, 0, :] = self.hs[:, 0, :]  # [B, 1, h]. This is the H_k of the single cluster we revealed so far (point n==0 was assigned to it).
            
            # Prepare U: sum of all unassigned datapoints (n+1,...,N):
            self.qs = self.q(data).view([self.batch_size, self.N, self.h_dim])   # [B, N, h]             
            self.Q = self.qs[:, 2:, ].sum(dim=1)    # [B, h]
                
        else: 
            # Prepare H_k: sum of all points within each cluster. (until point n-1, including)           
            if K == self.previous_K:            
                self.Hs[:, cs[n - 1], :] += self.hs[:, n - 1, :] # [B, K, h]. K is the number of clusters we revealed so far.
            else:
                self.Hs = torch.cat((self.Hs, self.hs[:, n - 1, :].unsqueeze(1)), dim=1)

            # Prepare U: sum of all unassigned datapoints (n+1,...,N):
            if n == self.N - 1:
                self.Q = torch.zeros([self.batch_size,self.h_dim]).to(self.device)  # [B, h]
                self.previous_n = 0
            else:
                self.Q -= self.qs[:, n, ]
            
        self.previous_K = K
        
        assert self.Hs.shape[1] == K
        
        logprobs = torch.zeros([self.batch_size, K + 1]).to(self.device) # [B, K + 1]
            
        # Compute G_k for each existing cluster for the n-th point:
        for k in range(K):
            Hs2 = self.Hs.clone()  # [B, K, h]. K is the number of clusters we revealed so far.
            Hs2[:, k, :] += self.hs[:, n, :]  # Add h_i to the relevangt H_k
            Hs2 = Hs2.view([self.batch_size * K, self.h_dim])  # [B * K, h]              
            gs  = self.g(Hs2).view([self.batch_size, K, self.g_dim]) # [B, K, g]
            Gk = gs.sum(dim=1)   # [B, g]
            uu = torch.cat((Gk, self.Q), dim=1)  # [B, g+h]. This is the argument for the call to f(). 
            logprobs[:, k] = torch.squeeze(self.f(uu))  # self.f return the scalar which is the logprob of n being assigned to cluster k.
            
        # Compute G_k for a new cluster:
        Hs2 = torch.cat((self.Hs, self.hs[:, n, :].unsqueeze(1)), dim=1)  # [B, K+1, h]   
        Hs2 = Hs2.view([self.batch_size * (K + 1), self.h_dim]) # [B * (K+1), h]                
        gs  = self.g(Hs2).view([self.batch_size, K+1, self.g_dim]) # [B, K+1, g]
        Gk = gs.sum(dim=1)  # [B, g]
        uu = torch.cat((Gk, self.Q), dim=1)   # [B, g+h]. The argument for the call to f()
        logprobs[:, K] = torch.squeeze(self.f(uu))  # self.f return the scalar which is the logprob of n being assigned to cluster K.  

        # Normalize
        m, _ = torch.max(logprobs, 1, keepdim=True)  # [B, 1]
        logprobs = logprobs - m - torch.log( torch.exp(logprobs - m).sum(dim=1, keepdim=True))

        return logprobs



