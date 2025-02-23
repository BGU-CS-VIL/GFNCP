import torch
import numpy as np
from plot_functions import plot_samples, plot_samples_old
from plot_histogram import histogram_for_data_perms, plot_best_clustering
from plot_histogram_old import histogram_for_data_perms_old, plot_best_clustering_old
from utils import compute_NMI, compute_ARI
import wandb
from collections import OrderedDict


def sample_periodically(wnb, data_orig, cs_gt, params, dpmm, it, N_sampling, show_histogram, stats):
    torch.cuda.empty_cache()  
    dataname = params['dataset_name']
    dpmm.eval()

    with torch.no_grad():
        
        # Get sampled clustering assignments given the same data group:
        cs_test, probs, nll, data = sample_from_model(dataname, data_orig, dpmm, S=100, seed=it)
                
        if params['dataset_name'] != 'Features':
            fig1, plt1 = plot_samples(dataname, data, cs_test, probs, seed=it)
            image = wandb.Image(fig1)
            wnb.log({f"Plots/sampling_{it}": image}, step=it)
            plt1.clf()
        
        NMI_test = compute_NMI(cs_gt, cs_test, probs)
        ARI_test = compute_ARI(cs_gt, cs_test, probs)
        curr_stats = OrderedDict(it=it)
        curr_stats.update({'NMI_test': NMI_test})
        curr_stats.update({'ARI_test': ARI_test})
        curr_stats.update({'nll_test': nll})
        wandb.log(curr_stats, step=it)
        
        # Update the general stats:
        stats['NMI_sum'] += NMI_test
        stats['ARI_sum'] += ARI_test
        stats['nll_sum'] += nll
        stats['count'] += 1
        if NMI_test > stats['NMI_max']:
            stats.update({'NMI_max': NMI_test})
            stats.update({'NMI_max_it': it})
        if ARI_test > stats['ARI_max']:
            stats.update({'ARI_max': ARI_test})
            stats.update({'ARI_max_it': it})
               
        # Plot sampling of another permutation of the same data:
        if params['dataset_name'] != 'Features':
            arr = np.arange(N_sampling)
            np.random.shuffle(arr)   # permute the order in which the points are queried
            data_orig_perm2 = data_orig[:, arr, :]
            cs_gt_perm2 = cs_gt[arr]
            
            cs_test2, probs2, nll2, data2 = sample_from_model(dataname, data_orig_perm2, dpmm, S=100, seed=it)
            fig2, plt2 = plot_samples(dataname, data2, cs_test2, probs2, seed=it)
            image = wandb.Image(fig2)
            wnb.log({f"Plots/sampling_with_permuted_data_{it}": image}, step=it)
            plt2.clf()
            # NMI_val = compute_NMI(cs_gt_perm2, cs_test2)
            # globals.run['Sampling/NMI'].append(NMI_val)
            # globals.run['Sampling/loss_test'].append(nll2)
        
        # Analyze the sample results of different data orders (histogram of probabilities): 
        if show_histogram:
            fig3, plt3, most_likely_clstr, prob_most_likely_clstr, prob_variance, prob_std, prob_std_avg = histogram_for_data_perms(data, dpmm, params, N=N_sampling, perms=500)
            image = wandb.Image(fig3)
            wnb.log({f"Plots_hist/hist_data_permutations_{it}": image}, step=it)
            plt3.clf()
            
            if params['dataset_name'] != 'Features':
                fig4, plt4 = plot_best_clustering(dataname, data, most_likely_clstr[None, :], np.array(prob_most_likely_clstr)[None])
                image = wandb.Image(fig4)
                wnb.log({f"Plots_hist/best_clustering_{it}": image}, step=it)
                plt4.clf()
            
            curr_stats = OrderedDict(it=it)
            curr_stats.update({'hist_prob_variance': prob_variance})
            curr_stats.update({'hist_prob_std': prob_std})
            curr_stats.update({'hist_prob_std_avg': prob_std_avg})
            wandb.log(curr_stats, step=it)
            
    dpmm.train()
    return stats
        

def sample_periodically_full_test(wnb, data_orig, cs_gt, params, dpmm, it, stats):
    torch.cuda.empty_cache()  
    dataname = params['dataset_name']
    dpmm.eval()

    with torch.no_grad():
        
        # Get sampled clustering assignments given the same data group:
        cs_test, probs, nll, data = sample_from_model(dataname, data_orig, dpmm, S=1, seed=it)
        print('cs_test.shape:', cs_test.shape)
        print('probs.shape:', probs.shape)
        print('data.shape:', data.shape)

        NMI_full_test = compute_NMI(cs_gt, cs_test, probs)
        ARI_full_test = compute_ARI(cs_gt, cs_test, probs)
        curr_stats = OrderedDict(it=it)
        curr_stats.update({'NMI_full_test': NMI_full_test})
        curr_stats.update({'ARI_full_test': ARI_full_test})
        curr_stats.update({'nll_full_test': nll})
        wandb.log(curr_stats, step=it)
        
        # Update the general stats:
        if NMI_full_test > stats['NMI_full_test_max']:
            stats.update({'NMI_full_test_max': NMI_full_test})
            stats.update({'NMI_full_test_max_it': it})
        if ARI_full_test > stats['ARI_full_test_max']:
            stats.update({'ARI_full_test_max': ARI_full_test})
            stats.update({'ARI_full_test_max_it': it})
            
    dpmm.train()
    return stats


def sample_from_model(dataname, data_orig, dpmm, S=100, seed=None):
    # S: number of samples given the same data
    # data_orig: [1, N, ..data_dim..]
    
    if seed:
        np.random.seed(seed=seed)
    
    if S == 1:
        data = data_orig
    else:
        if dataname in ('Gauss2D', 'Features'):
            data = data_orig.repeat(S, 1, 1)  # [S, N, 2]. This is only one data point repeated S times.
        elif dataname == 'MNIST' or dataname == 'FASHIONMNIST':
            data = data_orig.repeat(S, 1, 1, 1)  # [S, N, 28, 28]. This is only one data point repeated S times.
        elif dataname == 'CIFAR':
            data = data_orig.repeat(S, 1, 1, 1, 1)  # [S, N, 3, 28, 28]. This is only one data point repeated S times.
        else:
            raise NameError('Unknown dataset_name ' + dataname)
        
    # Get the cs sample for data:
    dpmm.encode(data)
    css, probs, nll = dpmm.sample_for_kl_eval()  # css (cs test): [S, N]; probs: [S,]
    return css, probs, nll, data
    

 
def sample_periodically_old(wnb, data, cs_gt, params, dpmm_ncp, it, N_sampling, show_histogram, stats):
    torch.cuda.empty_cache()  
    dataname = params['dataset_name']
    dpmm_ncp.eval()
    
    with torch.no_grad():
        
        fig1, plt1, cs_test, nll, probs  = plot_samples_old(dataname, dpmm_ncp, data, N=N_sampling, seed=it)
        image = wandb.Image(fig1)
        wnb.log({f"NCP_Plots/sampling_{it}": image}, step=it)
        plt1.clf()
        NMI_test = compute_NMI(cs_gt, cs_test, probs)
        curr_stats = OrderedDict(it=it)
        curr_stats.update({'NCP_NMI_test': NMI_test})
        curr_stats.update({'NCP_nll_test': nll})
        wandb.log(curr_stats, step=it)
        
        # Update the general stats:
        stats['NMI_ncp_sum'] += NMI_test
        stats['nll_ncp_sum'] += nll
        stats['count_ncp'] += 1
        if NMI_test > stats['NMI_ncp_max']:
            stats.update({'NMI_ncp_max': NMI_test})
            stats.update({'NMI_ncp_max_it': it})

        # Plot sampling of another permutation of the same data:
        arr = np.arange(N_sampling)
        np.random.shuffle(arr)   # permute the order in which the points are queried
        data_perm2 = data[:, arr, :]
        cs_gt_perm2 = cs_gt[arr]
        fig2, plt2, cs_test2, nll2, probs  = plot_samples_old(dataname, dpmm_ncp, data_perm2, N = N_sampling, seed=it)
        image = wandb.Image(fig2)
        wnb.log({f"NCP_Plots/sampling_with_permuted_data_{it}": image}, step=it)
        plt2.clf()
        # NMI_val = compute_NMI(cs_gt_perm2, cs_test2)
        # globals.run['Sampling/old_NCP/NMI'].append(NMI_val)
        # globals.run['Sampling/old_NCP/loss_test'].append(nll2)

        # Analyze the sample results of different data orders (histogram of probabilities): 
        if show_histogram:
            fig3, plt3, most_likely_clstr, prob_most_likely_clstr, prob_variance, prob_std, prob_std_avg = histogram_for_data_perms_old(data, dpmm_ncp, params, N=N_sampling, perms=500)
            image = wandb.Image(fig3)
            wnb.log({f"NCP_Plots_hist/hist_data_permutations_{it}": image}, step=it)
            plt3.clf()
            fig4, plt4 = plot_best_clustering_old(dataname, data, most_likely_clstr[None, :], np.array(prob_most_likely_clstr)[None])
            image = wandb.Image(fig4)
            wnb.log({f"NCP_Plots_hist/best_clustering_{it}": image}, step=it)
            plt4.clf()
            
            curr_stats = OrderedDict(it=it)
            curr_stats.update({'NCP_hist_prob_variance': prob_variance})
            curr_stats.update({'NCP_hist_prob_std': prob_std})
            curr_stats.update({'NCP_hist_prob_std_avg': prob_std_avg})
            wandb.log(curr_stats, step=it)
 
    dpmm_ncp.train()
    return stats