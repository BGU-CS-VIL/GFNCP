import torch
import numpy as np
from plot_functions import plot_samples
from plot_histogram import histogram_for_data_perms, plot_best_clustering
from utils import compute_NMI, compute_ARI
import wandb
from collections import OrderedDict


def eval_stats(wnb, data_generator, batch_size, params, dpmm, it, stats, M=50):
    # M: number of test samples to compute stats on
    
    torch.cuda.empty_cache()  
    dataname = params['dataset_name']
    channels = params['channels']
    dpmm.eval()
        
    with torch.no_grad():
        
        NMI_test = 0
        ARI_test = 0
        # ACC_test = 0
        LL_test = 0
        MC_test = 0
        for i in range(M):
            data, cs_gt, _, _ = data_generator.generate(N=None, batch_size=batch_size, train=False)  # data: [1, N, 2] or [1, N_sampling, 28, 28] or [1, N, 3, 28, 28]
            cs_gt = cs_gt[0, :] 

            # Get sampled clustering assignments given the data groups:
            cs_test, _, ll, mc_loss, _ = sample_from_model(channels, data, dpmm, S=1, seed=it)
            
            NMI_test += compute_NMI(cs_gt, cs_test, None)
            ARI_test += compute_ARI(cs_gt, cs_test, None)
            # ACC_test += compute_ACC(cs_gt, cs_test)
            LL_test += ll
            MC_test += mc_loss
        
        NMI_test = NMI_test / M
        ARI_test = ARI_test / M
        LL_test = LL_test / M
        MC_test = MC_test / M
        # ACC_test = ACC_test / M
        
        curr_stats = OrderedDict(it=it)
        curr_stats.update({'NMI_test': NMI_test})
        curr_stats.update({'ARI_test': ARI_test})
        # curr_stats.update({'ACC_test': ACC_test})
        curr_stats.update({'LL_test': LL_test})
        curr_stats.update({'MC_test': MC_test})
        wandb.log(curr_stats, step=it)
        
        # Update the general stats: 
        #   # (THIS WAS RELEVANT WHEN WE COMPUTED STATS ONLY ON ONE BATCH, WE NEEDED THIS TO CIMPUTE AVERAGE AT THE END)
        # if it >= params['iter_stats_avg']:
        #     stats['NMI_sum'] += NMI_test
        #     stats['ARI_sum'] += ARI_test
        #     stats['LL_sum'] += ll
        #     stats['count'] += 1
        
        if NMI_test > stats['NMI_max']:
            stats.update({'NMI_max': NMI_test})
            stats.update({'NMI_max_it': it})
        if ARI_test > stats['ARI_max']:
            stats.update({'ARI_max': ARI_test})
            stats.update({'ARI_max_it': it})
        # if ACC_test > stats['ACC_max']:
        #     stats.update({'ACC_max': ACC_test})
        #     stats.update({'ACC_max_it': it})
        if LL_test > stats['LL_max']:
            stats.update({'LL_max': LL_test})
            stats.update({'LL_max_it': it})
        if MC_test < stats['MC_min']:
            stats.update({'MC_min': MC_test})
            stats.update({'MC_min_it': it})
                                    
    dpmm.train()
    return stats
        


def plot_samples_and_histogram(wnb, data_orig, cs_gt, params, dpmm, it, N=20, show_histogram=False):
    torch.cuda.empty_cache()  
    dataname = params['dataset_name']
    channels = params['channels']
    dpmm.eval()

    with torch.no_grad():
        
        # Get sampled clustering assignments given the same data group:
        cs_test, probs, _, _, data = sample_from_model(channels, data_orig, dpmm, S=100, take_max=False, seed=it)
        NMI_test_sampling = compute_NMI(cs_gt, cs_test, probs)
        
        if params['dataset_name'] != 'Features':
            fig1, plt1 = plot_samples(params, dataname, data, cs_test, probs, nmi=NMI_test_sampling, seed=it)
            image = wandb.Image(fig1)
            wnb.log({f"Plots/sampling_{it}": image}, step=it)
            plt1.clf()
               
            # Plot sampling of another permutation of the same data:
            arr = np.arange(N)
            np.random.shuffle(arr)   # permute the order in which the points are queried
            data_orig_perm2 = data_orig[:, arr, :]
            cs_gt_perm2 = cs_gt[arr]
            
            cs_test2, probs2, _, _, data2 = sample_from_model(channels, data_orig_perm2, dpmm, S=100, take_max=False, seed=it)
            NMI_test_sampling2 = compute_NMI(cs_gt_perm2, cs_test2, probs2)
            fig2, plt2 = plot_samples(params, dataname, data2, cs_test2, probs2, nmi=NMI_test_sampling2, seed=it)
            image = wandb.Image(fig2)
            wnb.log({f"Plots/sampling_with_permuted_data_{it}": image}, step=it)
            plt2.clf()
        
        # Analyze the sample results of different data orders (histogram of probabilities): 
        if show_histogram:
            fig3, plt3, most_likely_clstr, prob_most_likely_clstr, prob_variance, prob_std, prob_std_avg = histogram_for_data_perms(data_orig, dpmm, params, N=N, perms=500)
            image = wandb.Image(fig3)
            wnb.log({f"Plots_hist/hist_data_permutations_{it}": image}, step=it)
            plt3.clf()
            
            if params['dataset_name'] != 'Features':
                fig4, plt4 = plot_best_clustering(params, dataname, data, most_likely_clstr[None, :], np.array(prob_most_likely_clstr)[None])
                image = wandb.Image(fig4)
                wnb.log({f"Plots_hist/best_clustering_{it}": image}, step=it)
                plt4.clf()
            
            curr_stats = OrderedDict(it=it)
            curr_stats.update({'hist_prob_variance': prob_variance})
            curr_stats.update({'hist_prob_std': prob_std})
            curr_stats.update({'hist_prob_std_avg': prob_std_avg})
            wandb.log(curr_stats, step=it)
            
    dpmm.train()



def sample_from_model(channels, data_orig, dpmm, S=100, take_max=True, seed=None):
    # S: number of samples given the same data
    # data_orig: [1, N_sampling, ..data_dim..]
    
    if seed:
        np.random.seed(seed=seed)
    
    if S == 1:
        data = data_orig
    else:
        if channels == 0:
            data = data_orig.repeat(S, 1, 1)  # [S, N, 2]. This is only one data point repeated S times.
        elif channels == 1:
            data = data_orig.repeat(S, 1, 1, 1)  # [S, N, 28, 28]. This is only one data point repeated S times.
        elif channels == 3:
            data = data_orig.repeat(S, 1, 1, 1, 1)  # [S, N, 3, 28, 28]. This is only one data point repeated S times.
        
    # Get the cs sample for data:
    css, probs, ll, mc_loss = dpmm.module.sample(data, take_max=take_max)  # css (cs test): [S, N]; probs: [S,]
    return css, probs, ll, mc_loss, data 
    
