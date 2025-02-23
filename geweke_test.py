import numpy as np
import matplotlib.pyplot as plt
import torch
from data_generator import generate_CRP
from evaluation import sample_from_model
import seaborn as sns



def geweke_test_histogram(dpmm, data_generator, params):
    
    N = params['N_Geweke']
    alpha = params['alpha_Geweke']
    M = params['M_Geweke']
    fig = plt.figure(3, figsize=(10, 10))
    plt.clf()
    
    dpmm.eval()
    with torch.no_grad():

        # Get multinomial samples:
        gt_K, pred_K, gt_mltn, pred_mltn = compute_K_multinomial_dist(dpmm, data_generator, params, N=N, alpha=alpha, M=M)
            # gt_K: [M,]; pred_K: [M * S',]; gt_mltn and pred_mltn: [Kmax==N,]

        # Plotting overlapping histograms with KDE
        sns.histplot(data=gt_K, color='blue', bins=10, discrete=True, shrink=0.8, alpha=0.5, kde=True, label='CRP', stat='probability')
        sns.histplot(data=pred_K, color='orange', bins=10, discrete=True, shrink=0.8, alpha=0.5, kde=True, label='GFNCP', stat='probability')  # kde_kws={'bw_adjust': 2}
        plt.xlabel('K')
        plt.ylabel('Density')
        plt.legend()
        plt.xticks(range(10))
        plt.title('K distribution (prior vs. model)', fontsize='15')
        plt.savefig('output/geweke/geweke_hist.png')
        
    dpmm.train()
    
    return fig, plt
    


def geweke_test_multiple_N(dpmm, data_generator, params):
    
    N_range = params['N_range_Geweke']
    alpha = params['alpha_Geweke']
    M = params['M_Geweke']
    cnt = N_range[1] - N_range[0]
    gt_mean_std = np.zeros((cnt, 2))
    pred_mean_std = np.zeros((cnt, 2))
    
    fig = plt.figure(4, figsize=(10, 10))
    plt.clf()
    
    dpmm.eval()
    with torch.no_grad():

        for n in range(N_range[0], N_range[1]):
            print('n', n)
            
            # Get multinomial samples for N=n:
            gt_K, pred_K, gt_mltn, pred_mltn = compute_K_multinomial_dist(dpmm, data_generator, params, N=n, alpha=alpha, M=M)
                # gt_K: [M,]; pred_K: [M * S',]; gt_mltn and pred_mltn: [Kmax==N,]
                    
            # compute mean and std of the samples:
            i = n - N_range[0]
            gt_mean_std[i, 0], gt_mean_std[i, 1] = np.mean(gt_K), np.std(gt_K)
            pred_mean_std[i, 0], pred_mean_std[i, 1] = np.mean(pred_K), np.std(pred_K)
            
        # Plot the results:
        x = np.arange(N_range[0], N_range[1], 1)
        y_gt = gt_mean_std[:, 0]
        y_gt_top = y_gt + gt_mean_std[:, 1]
        y_gt_bottom = y_gt - gt_mean_std[:, 1]
        
        y_pred = pred_mean_std[:, 0]
        y_pred_top = y_pred + pred_mean_std[:, 1]
        y_pred_bottom = y_pred - pred_mean_std[:, 1]
        
        plt.plot(x, y_gt, color='blue', label='CRP')
        plt.plot(x, y_pred, color='orange', label='GFNCP')
        plt.fill_between(x, y_gt_bottom, y_gt_top, color='blue', alpha=0.5)
        plt.fill_between(x, y_pred_bottom, y_pred_top, color='orange', alpha=0.5)
        plt.xlabel('Number of points')
        plt.ylabel('Average number of clusters')
        plt.legend()
        # plt.xticks(range(N_range[0], N_range[1]))
        plt.title('K distribution (prior vs. model)', fontsize='15')
        plt.savefig('output/geweke/geweke_multiple_N.png')

    
    dpmm.train()
    
    return fig, plt


def compute_K_multinomial_dist(dpmm, data_generator, params, N=30, alpha=0.7, M=1000):
    '''
        N, alpha: the fixed N and alpha values that will be used for the histogram results
        M: number of samples from the CRP or Model that the multinomial distribution will be based on.
    '''
    
    # In these tensors: value in entry i is the number of times we got i clusters (we assume Kmax == N):
    S = 1 #100  # number of samples from the model for one data point 
    gt_mltn = np.zeros(N) 
    pred_mltn = np.zeros(N)
    gt_K = []
    pred_K = []

    for i in range(M):
        print('i', i)
        
        # Sample from CRP:
        clusters, N, K = generate_CRP(params, N=N, alpha=alpha, train=False)  
        gt_mltn[K - 1] += 1
        gt_K.append(K)
        
        # Sample from model:
        data, cs_gt, _, K, _ = data_generator.generate(N=N, batch_size=1, train=False)  # data: [1, N, 2] or [1, N_sampling, 28, 28] or [1, N, 3, 28, 28]
        cs_gt = cs_gt[0, :] 
        cs_test, probs, _, _, _ = sample_from_model(params['channels'], data, dpmm, S=S, take_max=True)  # cs_test: [S', N] (where S' is the unique array of the samples [S, N])
        
        for s in range(cs_test.shape[0]):
            K_s = np.max(cs_test[s, :]) + 1
            pred_mltn[K_s - 1] += 1 
            pred_K.append(K_s)

    gt_K = np.array(gt_K)
    pred_K = np.array(pred_K)
    
    return gt_K, pred_K, gt_mltn, pred_mltn  # gt_K: [M,]; pred_K: [M * S,]; gt_mltn and pred_mltn: [Kmax==N,]
        
        
        
    
        
