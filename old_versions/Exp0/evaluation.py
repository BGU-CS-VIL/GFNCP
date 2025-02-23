import torch
import numpy as np
from plot_functions import plot_samples, plot_samples_old
from plot_histogram import histogram_for_data_perms, plot_best_clustering
from plot_histogram_old import histogram_for_data_perms_old, plot_best_clustering_old
from utils import compute_NMI
import wandb
from collections import OrderedDict


def sample_periodically(wnb, data, cs_gt, params, dpmm, it, N_sampling, show_histogram):
    torch.cuda.empty_cache()  
    datasetname = params['dataset_name']
    dpmm.eval()

    with torch.no_grad():
        
        fig1, plt1, cs_test, nll = plot_samples(datasetname, dpmm, data, N = N_sampling, seed=it)
        image = wandb.Image(fig1)
        wnb.log({f"Plots/sampling_{it}": image}, step=it)
        plt1.clf()
        NMI_val = compute_NMI(cs_gt, cs_test)
        stats = OrderedDict(it=it)
        stats.update({'NMI_val': NMI_val})
        stats.update({'nll_test': nll})
        wandb.log(stats, step=it)
        
        # Plot sampling of another permutation of the same data:
        arr = np.arange(N_sampling)
        np.random.shuffle(arr)   # permute the order in which the points are queried
        data_perm2 = data[:, arr, :]
        cs_gt_perm2 = cs_gt[arr]
        fig2, plt2, cs_test2, nll2 = plot_samples(datasetname, dpmm, data_perm2, N = N_sampling, seed=it)
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
            fig4, plt4 = plot_best_clustering(datasetname, data, most_likely_clstr[None, :], np.array(prob_most_likely_clstr)[None])
            image = wandb.Image(fig4)
            wnb.log({f"Plots_hist/best_clustering_{it}": image}, step=it)
            plt4.clf()
            
            stats = OrderedDict(it=it)
            stats.update({'hist_prob_variance': prob_variance})
            stats.update({'hist_prob_std': prob_std})
            stats.update({'hist_prob_std_avg': prob_std_avg})
            wandb.log(stats, step=it)
            
    dpmm.train()
        
    
def sample_periodically_old(wnb, data, cs_gt, params, dpmm_old, it, N_sampling, show_histogram):
    torch.cuda.empty_cache()  
    datasetname = params['dataset_name']
    dpmm_old.eval()
    
    with torch.no_grad():
        
        fig1, plt1, cs_test, nll = plot_samples_old(datasetname, dpmm_old, data, N=N_sampling, seed=it)
        image = wandb.Image(fig1)
        wnb.log({f"NCP_Plots/sampling_{it}": image}, step=it)
        plt1.clf()
        NMI_val = compute_NMI(cs_gt, cs_test)
        stats = OrderedDict(it=it)
        stats.update({'NCP_NMI_val': NMI_val})
        stats.update({'NCP_nll_test': nll})
        wandb.log(stats, step=it)

        # Plot sampling of another permutation of the same data:
        arr = np.arange(N_sampling)
        np.random.shuffle(arr)   # permute the order in which the points are queried
        data_perm2 = data[:, arr, :]
        cs_gt_perm2 = cs_gt[arr]
        fig2, plt2, cs_test2, nll2 = plot_samples_old(datasetname, dpmm_old, data_perm2, N = N_sampling, seed=it)
        image = wandb.Image(fig2)
        wnb.log({f"NCP_Plots/sampling_with_permuted_data_{it}": image}, step=it)
        plt2.clf()
        # NMI_val = compute_NMI(cs_gt_perm2, cs_test2)
        # globals.run['Sampling/old_NCP/NMI'].append(NMI_val)
        # globals.run['Sampling/old_NCP/loss_test'].append(nll2)

        # Analyze the sample results of different data orders (histogram of probabilities): 
        if show_histogram:
            fig3, plt3, most_likely_clstr, prob_most_likely_clstr, prob_variance, prob_std, prob_std_avg = histogram_for_data_perms_old(data, dpmm_old, params, N=N_sampling, perms=500)
            image = wandb.Image(fig3)
            wnb.log({f"NCP_Plots_hist/hist_data_permutations_{it}": image}, step=it)
            plt3.clf()
            fig4, plt4 = plot_best_clustering_old(datasetname, data, most_likely_clstr[None, :], np.array(prob_most_likely_clstr)[None])
            image = wandb.Image(fig4)
            wnb.log({f"NCP_Plots_hist/best_clustering_{it}": image}, step=it)
            plt4.clf()
            
            stats = OrderedDict(it=it)
            stats.update({'NCP_hist_prob_variance': prob_variance})
            stats.update({'NCP_hist_prob_std': prob_std})
            stats.update({'NCP_hist_prob_std_avg': prob_std_avg})
            wandb.log(stats, step=it)
 
    dpmm_old.train()