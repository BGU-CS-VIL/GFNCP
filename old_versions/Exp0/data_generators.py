#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import torch
from torchvision import datasets, transforms


from utils import relabel



def get_generator(params):
    
    if params['dataset_name'] == 'MNIST':
        return MNIST_generator(params)        
    elif params['dataset_name'] == 'Gauss2D':         
        return Gauss2D_generator(params)   
    elif params['dataset_name'] == 'FASHIONMNIST':         
        return FASHIONMNIST_generator(params) 
    elif params['dataset_name'] == 'CIFAR':         
        return CIFAR_generator(params) 
    else:
        raise NameError('Unknown dataset_name '+ params['dataset_name'] )



class MNIST_generator():
    
    def __init__(self, params, train=True):
        
        self.Nmin = params['Nmin']
        self.Nmax = params['Nmax']
        self.img_sz = params['img_sz']
        
        self.params=params        
        self.dataset = datasets.MNIST(params['data_path'], train=train, download=True, \
                       transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]))
        # self.dataset[i][0] is the i-th image of shape [1, 28, 28]
        # self.dataset[i][1] is the i-th label (scalar))
        
        all_labels = np.zeros(len(self.dataset), dtype= np.int32)
        for i in range(len(self.dataset)):
            all_labels[i] = self.dataset[i][1]  #self.dataset[i][1].item()
            
        # Create a list of groups of images per class.
        # E.g: entry 0 in "self.label_data" will be of shape (S, 28, 28) where M is the number of images from class 0.
        self.label_data = {}
        for i in range(10):
            print('Processing label: ', i)
            label_inds = np.nonzero(all_labels == i)[0]   # Returns all indices in "all_labels" that their label value is i         
            S = label_inds.shape[0]  # S is the number of data points assigned to label i
            self.label_data[i] =torch.zeros([S, self.img_sz, self.img_sz])
            for s in range(S):
                self.label_data[i][s,:,:] = self.dataset[label_inds[s]][0][0,:,:]

        
    def generate(self, N=None, batch_size=1):
        
        K = 11
        while K>10:  # Generate clustering according to CRP with K<10.
            clusters, N, K = generate_CRP(self.params, N=N)  
            # "clusters": is in shape [N+2]. Entry i (from 1..N+1) holds the number of points assigned to label i.
        
        data = torch.zeros([batch_size, N, self.img_sz, self.img_sz])
        
        cumsum = np.cumsum(clusters)  # Cumulative sum over clusters. Shape: [N+2]
        
        # Fill in "data" and "cs": 
        #   "data": shape: [B, N, 28, 28]. Each point in B is a dataset of N images with the same mixture of K clusters (but with different classes).
        #   "cs": shape: [N]. This is a list of ground-truth labels of images from "data", which are relevant to all B point.
        for i in range(batch_size):
            labels = np.random.choice(10, size=K, replace = False )  #this is a sample from the 'base measure' for each cluster
            for k in range(K):
                l = labels[k]
                nk = clusters[k+1]
                inds = np.random.choice(self.label_data[l].shape[0],size=nk, replace = False )                
                data[i, cumsum[k]:cumsum[k+1], :,: ] = self.label_data[l][inds,:,:]

        cs = np.empty(N, dtype=np.int32)        
        for k in range(K):
            cs[cumsum[k]:cumsum[k + 1]]= k
        
        # Shuffle "data" and "cs" in the same way:
        arr = np.arange(N)
        np.random.shuffle(arr)
        cs = cs[arr]        
        data = data[:, arr, :, :]
        
        # Relabel cluster numbers so that they appear in order
        cs = relabel(cs)
        
        return data, cs, clusters, K    # data: [B, N, 28, 28], cs: [N], clusters: [N+2]
        


class FASHIONMNIST_generator():
    
    def __init__(self, params, train=True):
        
        self.Nmin = params['Nmin']
        self.Nmax = params['Nmax']
        self.img_sz = params['img_sz']
        
        self.params=params        
        
        self.dataset = datasets.FashionMNIST(params['data_path'],
                            transform=transforms.Compose([
                            transforms.Resize(self.img_sz),
                            transforms.CenterCrop(self.img_sz),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, ), (0.5, ))]),
                            train=train,
                            download=True)
        # self.dataset[i][0] is the i-th image of shape [1, 28, 28]
        # self.dataset[i][1] is the i-th label (scalar))
              
        all_labels = np.zeros(len(self.dataset), dtype= np.int32)
        for i in range(len(self.dataset)):
            all_labels[i] = self.dataset[i][1]  #self.dataset[i][1].item()
            
        # Create a list of groups of images per class.
        # E.g: entry 0 in "self.label_data" will be of shape (M, 28, 28) where M is the number of images from class 0.
        self.label_data = {}
        for i in range(10):
            print('Processing label: ', i)
            label_inds = np.nonzero(all_labels == i)[0]   # Returns all indices in "all_labels" that their label value is i         
            S = label_inds.shape[0]  # S is the number of data points assigned to label i
            self.label_data[i] =torch.zeros([S, self.img_sz, self.img_sz])
            for s in range(S):
                self.label_data[i][s,:,:] = self.dataset[label_inds[s]][0][0,:,:]

        
    def generate(self, N=None, batch_size=1):
        
        K = 11
        while K>10:  # Generate clustering according to CRP with K<10.
            clusters, N, K = generate_CRP(self.params, N=N)  
            # "clusters": is in shape [N+2]. Entry i (from 1..N+1) holds the number of points assigned to label i.
        
        data = torch.zeros([batch_size, N, self.img_sz, self.img_sz])
        
        cumsum = np.cumsum(clusters)  # Cumulative sum over clusters. Shape: [N+2]
        
        # Fill in "data" and "cs": 
        #   "data": shape: [B, N, 28, 28]. Each point in B is a dataset of N images with the same mixture of K clusters (but with different classes).
        #   "cs": shape: [N]. This is a list of ground-truth labels of images from "data", which are relevant to all B point.
        for i in range(batch_size):
            labels = np.random.choice(10, size=K, replace = False )  #this is a sample from the 'base measure' for each cluster
            for k in range(K):
                l = labels[k]
                nk = clusters[k+1]
                inds = np.random.choice(self.label_data[l].shape[0],size=nk, replace = False )                
                data[i, cumsum[k]:cumsum[k+1], :,: ] = self.label_data[l][inds,:,:]

        cs = np.empty(N, dtype=np.int32)        
        for k in range(K):
            cs[cumsum[k]:cumsum[k+1]]= k
        
        # Shuffle "data" and "cs" in the same way:
        arr = np.arange(N)
        np.random.shuffle(arr)
        cs = cs[arr]        
        data = data[:,arr,:,:]
        
        # Relabel cluster numbers so that they appear in order
        cs = relabel(cs)
        
        return data, cs, clusters, K    # data: [B, N, 28, 28], cs: [N], clusters: [N+2]
    
    
class CIFAR_generator():
    
    def __init__(self, params, train=True, transform=None):
        
        self.Nmin = params['Nmin']
        self.Nmax = params['Nmax']
        self.img_sz = params['img_sz']
        self.channels = params['channels']
        deterministic = False
        self.params = params        
        
        transform = transforms.Compose([
                    t for t in [
                        transforms.Resize(self.img_sz),
                        transforms.CenterCrop(self.img_sz),
                        (not deterministic) and transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        (not deterministic) and
                        transforms.Lambda(lambda x: x + 1. / 128 * torch.rand(x.size())),
                    ] if t is not False
                ]) if transform == None else transform
          
        self.dataset = datasets.CIFAR10(root=params['data_path'],
                                  train=train,
                                  download=True,
                                  transform=transform)
        # self.dataset[i][0] is the i-th image of shape [3, 28, 28]
        # self.dataset[i][1] is the i-th label (scalar))
              
        all_labels = np.zeros(len(self.dataset), dtype= np.int32)
        for i in range(len(self.dataset)):
            all_labels[i] = self.dataset[i][1]  #self.dataset[i][1].item()
            
        # Create a list of groups of images per class.
        # E.g: entry 0 in "self.label_data" will be of shape (S, 3, 28, 28) where S is the number of images from class 0.
        self.label_data = {}
        for i in range(10):
            print('Processing label: ', i)
            label_inds = np.nonzero(all_labels == i)[0]   # Returns all indices in "all_labels" that their label value is i         
            S = label_inds.shape[0]  # S is the number of data points assigned to label i
            self.label_data[i] =torch.zeros([S, self.channels, self.img_sz, self.img_sz])
            for s in range(S):
                self.label_data[i][s, :, :, :] = self.dataset[label_inds[s]][0][:,:,:]

        
    def generate(self, N=None, batch_size=1):
        
        K = 11
        while K>10:  # Generate clustering according to CRP with K<10.
            clusters, N, K = generate_CRP(self.params, N=N)  
            # "clusters": is in shape [N+2]. Entry i (from 1..N+1) holds the number of points assigned to label i.
        
        data = torch.zeros([batch_size, N, self.channels, self.img_sz, self.img_sz])
        
        cumsum = np.cumsum(clusters)  # Cumulative sum over clusters. Shape: [N+2]
        
        # Fill in "data" and "cs": 
        #   "data": shape: [B, N, 28, 28]. Each point in B is a dataset of N images with the same mixture of K clusters (but with different classes).
        #   "cs": shape: [N]. This is a list of ground-truth labels of images from "data", which are relevant to all B point.
        for i in range(batch_size):
            labels = np.random.choice(10, size=K, replace = False )  #this is a sample from the 'base measure' for each cluster
            for k in range(K):
                l = labels[k]
                nk = clusters[k+1]
                inds = np.random.choice(self.label_data[l].shape[0],size=nk, replace = False )                
                data[i, cumsum[k]:cumsum[k+1], :, :, :] = self.label_data[l][inds, :, :, :]

        cs = np.empty(N, dtype=np.int32)        
        for k in range(K):
            cs[cumsum[k]:cumsum[k+1]]= k
        
        # Shuffle "data" and "cs" in the same way:
        arr = np.arange(N)
        np.random.shuffle(arr)
        cs = cs[arr]        
        data = data[:, arr, :, :,:]
        
        # Relabel cluster numbers so that they appear in order
        cs = relabel(cs)
        
        return data, cs, clusters, K    # data: [B, N, 3, 28, 28], cs: [N], clusters: [N+2]
    

class Gauss2D_generator():
    
    def __init__(self,params):
        self.params = params
        

    def generate(self,N=None, batch_size=1):        
        
        lamb = self.params['lambda']
        sigma = self.params['sigma']
        x_dim = self.params['x_dim']    
        
        clusters, N, num_clusters = generate_CRP(self.params, N=N)
            
        
        cumsum = np.cumsum(clusters)  # Cumulative sum. Shape: [N+2]
        data = np.empty([batch_size, N, x_dim])
        cs =  np.empty(N, dtype=np.int32)
        
        for i in range(num_clusters):
            mu= np.random.normal(0,lamb, size = [x_dim*batch_size,1])
            samples= np.random.normal(mu,sigma, size=[x_dim*batch_size,clusters[i+1]] )
            
            samples = np.swapaxes(samples.reshape([batch_size, x_dim,clusters[i+1]]),1,2)        
            data[:,cumsum[i]:cumsum[i+1],:]  = samples
            cs[cumsum[i]:cumsum[i+1]]= i+1
            
        #%shuffle the assignment order
        arr = np.arange(N)
        np.random.shuffle(arr)
        cs = cs[arr]
        
        data = data[:,arr,:]
        
        # relabel cluster numbers so that they appear in order 
        cs = relabel(cs)
        
        #normalize data 
        #means = np.expand_dims(data.mean(axis=1),1 )    
        medians = np.expand_dims(np.median(data,axis=1),1 )    
        
        data = data-medians
        #data = 2*data/(maxs-mins)-1        #data point are now in [-1,1]
        
        data = torch.tensor(data).float().to(self.params['device'])
    
        return data, cs, clusters, num_clusters
        
            




# Group the data according to CRP (assign each point to a cluster, N is random and K is sampled from the CRP)
def generate_CRP(params, N, no_ones=False):
    
    alpha = params['alpha']   #dispersion parameter of the Chinese Restaurant Process
    crp_not_done = True
    
    while crp_not_done:
        if N is None or N==0:
            N = np.random.randint(params['Nmin'],params['Nmax'])
            
                
        clusters = np.zeros(N+2)
        clusters[0] = 0
        clusters[1] = 1      # we start filling the array here in order to use cumsum below 
        clusters[2] = alpha
        index_new = 2
        for n in range(N-1):     #we loop over N-1 particles because the first particle was assigned already to cluster[1]
            p = clusters/clusters.sum()
            z = np.argmax(np.random.multinomial(1, p))
            if z < index_new:  # Assign point n to an existing cluster
                clusters[z] +=1
            else:              # Assign point n to a new cluster
                clusters[index_new] =1
                index_new +=1
                clusters[index_new] = alpha
        
        clusters[index_new] = 0 
        clusters = clusters.astype(np.int32)
        
        if no_ones:
            clusters= clusters[clusters!=1]
        N = int(np.sum(clusters))
        crp_not_done = N==0                       
        
        
    K = np.sum(clusters>0)

    return clusters, N, K
