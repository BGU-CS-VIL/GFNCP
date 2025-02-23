#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from ncp_sampler import NCP_Sampler


def plot_avgs(losses, accs, rot_vars, w, save_name=None):
    up = -1 #3500
    
    avg_loss = []
    for i in range(w, len(losses)):
        avg_loss.append(np.mean(losses[i-w:i]))
    
    avg_acc = []
    for i in range(w, len(accs)):
        avg_acc.append(np.mean(accs[i-w:i]))
    
    avg_var = []
    for i in range(w, len(rot_vars)):
        avg_var.append(np.mean(rot_vars[i-w:i]))
    
    plt.figure(22, figsize=(13,10))
    plt.clf()
    
    plt.subplot(312)
    plt.semilogy(avg_loss[:up])
    plt.ylabel('Mean NLL')
    plt.grid()
    
    plt.subplot(311)
    plt.plot(avg_acc[:up])
    plt.ylabel('Mean Accuracy')
    plt.grid()
    
    plt.subplot(313)
    plt.semilogy(avg_var[:up])
    plt.ylabel('Permutation Variance' )
    plt.xlabel('Iteration')
    plt.grid()

    if save_name:
        plt.savefig(save_name)


def plot_samples_old(dataname, dpmm, data, nrows=5, N=20, seed = None, save_name=None):
    # rows: number of samples to show (e.g. we show the first 5 samples from css)

    if seed:
        np.random.seed(seed=seed)
    
    fig = None
    plt = None
    
    S = 100   # 5000  number of samples

    # if dataname == 'Gauss2D':
    #     data = data_orig.repeat(S, 1, 1)  # [S, N, 2]. This is only one data point repeated S times.
    # if dataname == 'MNIST' or dataname == 'FASHIONMNIST':
    #     data = data_orig.repeat(S, 1, 1, 1)  # [S, N, 28, 28]. This is only one data point repeated S times.
    # elif dataname == 'CIFAR':
    #     data = data_orig.repeat(S, 1, 1, 1, 1)  # [S, N, 3, 28, 28]. This is only one data point repeated S times.
    
    # DEBUG: use the same dataset in every iteration and also for evaluation:
    # data = np.repeat(np.expand_dims(data_0[0, :, :], 0), S, axis=0)  # [S, N, 2]. This is only one data point repeated S times.
    
    # Get the cs sample for data:
    # print(data[0].shape)
    # print(data[0,:])
    ncp_sampler = NCP_Sampler(dpmm, data)
    css, probs, nll = ncp_sampler.sample(S) # css (cs test): [S, N]; probs: [S,]
    
    if dataname == 'Gauss2D':
        data = data.clone().detach().to('cpu').numpy()
        fig, plt = plot_samples_Gauss2D(data, css, probs, rows=nrows, N=20, save_name=None)
    if dataname == 'MNIST' or dataname == 'FASHIONMNIST':
        fig, plt = plot_samples_MNIST(data, css, probs, rows=nrows, N=20, save_name=None)
    elif dataname == 'CIFAR':
        fig, plt = plot_samples_CIFAR(data, css, probs, rows=nrows, N=20, save_name=None)
        
    return fig, plt, css, nll, probs


def plot_samples(dataname, data, css, probs, nrows=5, N=20, seed=None, save_name=None):
    # rows: number of samples to show (e.g. we show the first 5 samples from css)

    if seed:
        np.random.seed(seed=seed)
    
    fig = None
    plt = None
    
    if dataname == 'Gauss2D':
        data = data.clone().detach().to('cpu').numpy()
        fig, plt = plot_samples_Gauss2D(data, css, probs, rows=nrows, N=N, save_name=None)
    if dataname == 'MNIST' or dataname == 'FASHIONMNIST':
        fig, plt = plot_samples_MNIST(data, css, probs, rows=nrows, N=N, save_name=None)
    elif dataname == 'CIFAR':
        fig, plt = plot_samples_CIFAR(data, css, probs, rows=nrows, N=N, save_name=None)
        
    return fig, plt


def plot_samples_Gauss2D(data, css, probs, rows=5, N=20, save_name=None): 
    # data: [S, N, 2]
    # css: [S, N]
    # probs: [S,]
    
    if rows == 1:
        fig = plt.figure(1, figsize=(8, 4))
        plt.clf()
        fig.suptitle('Most-likely Clustering', fontsize=20)
    else:
        fig = plt.figure(1, figsize=(35, 6))
        plt.clf()
        fig.suptitle('5 Top Clustering Samples', fontsize=20)
    
    _, ax = plt.subplots(ncols=rows+1, nrows=1, num=1)
        
    ax = ax.reshape(rows+1)

    N = data.shape[1]
    s = 26  #size for scatter
    fontsize = 15
    
    ax[0].scatter(data[0,:,0], data[0,:,1], color='gray',s=s)            
    ax[0].set_title(str(N) + ' Points',fontsize=fontsize )

    for axis in ['top','bottom','left','right']:
      ax[0].spines[axis].set_linewidth(2)

    if css.shape[0] < 5:
        rows = css.shape[0]
        # css = np.repeat(css, 5, axis=0)
        # probs = np.repeat(probs, 5, axis=0)
    
    for i in range(rows):
        ax[i+1].cla()
        cs= css[i,:]

        for j in range(N):        
            xs = data[0,j,0]
            ys = data[0,j,1]                
            ax[i+1].scatter(xs, ys, color='C'+str(cs[j]+1),s=s)
                
        K=len(set(cs))
        
        ax[i+1].set_title(str(K) + ' Clusters    Prob: '+ '{0:.2f}'.format(probs[i]), fontsize=fontsize)
        for axis in ['top','bottom','left','right']:
          ax[i+1].spines[axis].set_linewidth(0.8)
    
    if save_name:
        plt.savefig(save_name, bbox_inches='tight')
        
    return fig, plt


def plot_samples_MNIST(data, css, probs, rows=5, N=20, save_name=None): 
    # data: one small dataset with N images. Shape: [1, N, 28, 28] 
    # css: [S, N]
    # probs: [S,]
    
    W = 10

    fig = plt.figure(3, figsize=(15, 10))

    if rows == 1:
        plt.clf()
        fig.suptitle('Most-likely Clustering', fontsize=25)
    else:
        plt.clf()
        fig.suptitle('5 Top Clustering Samples', fontsize=25)
    
    step = 0.074
    
    for i in range(N):
        plt.subplot(W+1,26,i+1+1)
        plt.imshow(data[0,i,:,:], cmap='gray')
        plt.xticks([])
        plt.yticks([])
    
    fontsize=20   
    plt.gcf().text(0.35, .9, str(N)+ ' Observations', fontsize=fontsize)
    plt.gcf().text(0.35, 0.76, str(rows) + ' Cluster Samples', fontsize=fontsize)
    
    if css.shape[0] < 5:
        rows = css.shape[0]
        # css = np.repeat(css, 5, axis=0)
        # probs = np.repeat(probs, 5, axis=0)
        
    for w in range(0, rows):
        it = css[w,:]
        K = len(set(it))
        
        dat = {}
        for k in range(K):
            dat[k]=data[0,np.where(it==k)[0],:,:]

        fontsize=15
        strtext = 'K = ' + str(K) + '  Pr: ' + '{0:.2f}'.format(probs[w]) 
        plt.gcf().text(0.03, 0.63-(w-1)*step, strtext, fontsize=fontsize)
        
        i= (w+2)*26
        for k in range(K):
            for j in range(len(dat[k])):
                plt.subplot(W+1,26,i+1+1)                    
                plt.imshow(dat[k][j,:,:], cmap='gray')
                plt.xticks([])
                plt.yticks([])
                i+=1
            i+=1
    
    if save_name:
        plt.savefig(save_name, bbox_inches="tight")
        
    return fig, plt


def plot_samples_CIFAR(data, css, probs, rows=5, N=20, save_name=None): 
    # data: one small dataset with N images. Shape: [1, N, 3, 28, 28] 
    # css: [S, N]
    # probs: [S,]
    
    W = 10

    fig = plt.figure(3,figsize=(15, 10))
    
    if rows == 1:
        plt.clf()
        plt.title('Most-likely Clustering', fontsize='25')
    else:
        plt.clf()
        plt.title('5 Top Clustering Samples', fontsize='25')
            
    step = 0.074
    
    for i in range(N):
        plt.subplot(W+1,26,i+1+1)
        img = Image.fromarray((np.squeeze(np.moveaxis((data[0, i, :, :, :].clone().detach().cpu() * 0.5 + 0.5).numpy(), 0, -1)) * 255).astype(np.uint8))
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
    
    fontsize=20    
    plt.gcf().text(0.35, .9, str(N)+ ' Observations', fontsize=fontsize)
    plt.gcf().text(0.35, 0.76, str(rows) + ' Cluster Samples', fontsize=fontsize)

    if css.shape[0] < 5:
        rows = css.shape[0]
        # css = np.repeat(css, 5, axis=0)
        # probs = np.repeat(probs, 5, axis=0)
        
    for w in range(0, rows):
        it = css[w,:]
        K = len(set(it))
        
        dat = {}
        for k in range(K):
            dat[k]=data[0, np.where(it==k)[0], :, :, :]

        fontsize=15
        strtext = 'K = ' + str(K) + '  Pr: ' + '{0:.2f}'.format(probs[w]) 
        plt.gcf().text(0.03, 0.63-(w-1)*step, strtext, fontsize=fontsize)
        
        i= (w+2)*26
        for k in range(K):
            for j in range(len(dat[k])):
                plt.subplot(W+1,26,i+1+1)  
                img = Image.fromarray((np.squeeze(np.moveaxis((dat[k][j,:, :, :].clone().detach().cpu() * 0.5 + 0.5).numpy(), 0, -1)) * 255).astype(np.uint8)) 
                plt.imshow(img)
                plt.xticks([])
                plt.yticks([])
                i+=1
            i+=1
    
    if save_name:
        plt.savefig(save_name, bbox_inches="tight")
        
    return fig, plt