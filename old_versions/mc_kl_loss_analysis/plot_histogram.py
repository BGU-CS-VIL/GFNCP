import numpy as np
from ncp_prob_computation import NCP_prob_computation
import torch
import matplotlib.pyplot as plt
from utils import relabel
from plot_functions import plot_samples_CIFAR, plot_samples_Gauss2D, plot_samples_MNIST

        
def histogram_for_data_perms(data_orig, dpmm, params, N = 20, perms = 100):
    '''
    data_orig: one small dataset with N images. Shape: [1, N, channels, img_sz, img_sz]. Tensor.
    dpmm: a trained model
    N: number of images to cluster
    perms: number of data-order permutations to check. (We compute the probability of the most likely assignment given different data orders).
    '''
    
    # !!!! PERFORM THIS ANALYSIS SEVERAL TIMES TO GET LESS NOISY STD !!!
    
    probs_for_histogram = np.zeros(perms + 1) # Stores the probabilities of getting the most likely assignmnet, given each data-order permutation
    
    S = 100  # 5000  number of samples
    if params['channels'] == 0:
        data = data_orig.repeat(S, 1, 1)  # e.g.: [S, N, 2]. This is only one data point repeated S times.
    elif params['channels'] == 1:
        data = data_orig.repeat(S, 1, 1, 1)  # e.g.: [S, N, 28, 28]. This is only one data point repeated S times.
    elif params['channels'] == 3:
        data = data_orig.repeat(S, 1, 1, 1, 1)  # e.g.: [S, N, 28, 28, 3]. This is only one data point repeated S times.
    
    # Find the most likely clustering of "data":
    dpmm.encode(data)
    css, probs, _ = dpmm.sample_for_kl_eval()  # css: [M, N]; probs: [M,], where M is the number of succeeded samples.
    most_likely_clstr = css[np.argmax(probs), :]  # [N,]
    prob_most_likely_clstr = np.max(probs)  # scalar
        
    # Sanity check: compute the probability of getting "most_likely_clstr" using the original data order (no permutation):
    prob_computer = NCP_prob_computation(dpmm, data)
    prob_orig_order = prob_computer.compute_prob(most_likely_clstr)
    print('\nProbability of most-likely assignment, computation vs. sampler:', prob_orig_order, prob_most_likely_clstr, '\n')
    
    # Sample "perms" permutations of data and store their probability result from the model:    
    data = data[0, :]  # take the first row, as all rows are equal.
    data = data[None, :]
    for p in range(perms + 1):
        # Draw a permutation:
        arr = np.arange(N)
        np.random.shuffle(arr)   
        data_perm = data[:, arr, :] # permute the order of the data and the most likely assignment
        most_likely_clstr_perm = most_likely_clstr[arr]
        
        # Compute the probability of getting "most_likely_clstr_perm" with this data-order permutation:
        prob_computer = NCP_prob_computation(dpmm, data_perm)
        probs_for_histogram[p] = prob_computer.compute_prob(most_likely_clstr_perm)
    
    probs_for_histogram[perms] = prob_orig_order

    # Compute histogram variance:
    counts, bins = np.histogram(probs_for_histogram, bins=20, range=(0, 1))
    # print(counts, bins)
    fig = plt.figure(3, figsize=(15, 8))
    plt.clf()
    plt.hist(bins[:-1], bins, weights=counts)
    plt.ylabel("Probabilities")
    plt.xlabel("Bin Number")
    plt.title('Histogram of the Most-likely Clustering Probabilities', fontsize='25')
    plt.xlim([0, 1])
    
    # plt.hist(probs_for_histogram, 20, histtype = 'bar', facecolor = 'blue', rwidth=2)

    prob_variance = probs_for_histogram.var()
    prob_std = np.std(probs_for_histogram)
    prob_std_avg = prob_std / np.mean(probs_for_histogram)
    
    return fig, plt, most_likely_clstr, prob_most_likely_clstr, prob_variance, prob_std, prob_std_avg


def plot_best_clustering(dataname, data, css, prob_css):
    '''
    data: one small dataset with N images in the original order. e.g if mnist the shape is: [1, N, 28, 28] 
    css: the most likely clustering for data. [N,]
    prob_css: probability of the most likely clustering. Scalar.
    '''
    # Plot the most-likely clustering and the original data order
    
    if dataname == 'Gauss2D':
        data = data.clone().detach().to('cpu').numpy()
        fig, plt = plot_samples_Gauss2D(data, css, prob_css, rows=1, N=20, save_name=None)
    if dataname == 'MNIST' or dataname == 'FASHIONMNIST':
        fig, plt = plot_samples_MNIST(data, css, prob_css, rows=1, N=20, save_name=None)
    elif dataname == 'CIFAR':
        fig, plt = plot_samples_CIFAR(data, css, prob_css, rows=1, N=20, save_name=None)

    return fig, plt
