import torch




def kl_loss_func(E, E_mask):
    E = E * E_mask
    m, _ = torch.min(E, 1, keepdim=True)    # [B, 1]                  
    logprobs_kl = ((-E + m) * E_mask) - torch.log((torch.exp(-E + m) * E_mask).sum(dim=1, keepdim=True))
    return logprobs_kl  # [B, K+1]


def j_loss_func(dpmm, E, N, n, data, cs, it):
    # E is [B, K + 1]. This is the unnormalized logprob of p(c_{0:n} | x).

    # Compute entropy:
    m, _ = torch.min(E, 1, keepdim=True)    # [B, 1]                  
    logprobs = ((-E + m)) - torch.log((torch.exp(-E + m)).sum(dim=1, keepdim=True))
    probs = torch.exp(logprobs)
    entrpy = torch.distributions.Categorical(probs).entropy()  # [B,]
    entrpy = entrpy.mean()
    
    # Compute J loss:
    data_E = E[:, cs[n]].mean()  # scalar. E[:, cs[n]] is [B, 1], it's the unnormalized logprob of p(c_{0:N} | x) using the ground-truth label for the N-th point.         
    fake_E = dpmm.module.sample(data, it)
    fake_E = fake_E.mean()  # scalar. Mean over the minibatch
    j_loss = data_E - fake_E
        
    return j_loss, data_E, entrpy
    
    
def mc_loss_func(E, E_mask, log_pn, n, N, cs, batch_size, it, device, epsilon=1e-5):
    MC_n_term = torch.zeros(1).to(E.device)
    if n == 1:
        m, _ = torch.min(E, 1, keepdim=True)    # [B, 1] 
        log_pn = - torch.unsqueeze(E[:, cs[n]], 1) + m    # [B, 1], unnormalized logprob of p(c_{0:n} | x)
        
    elif n >= 2:
        m, _ = torch.min(E, 1, keepdim=True)    # [B, 1]
        # MC_n_term_0 = (log_pn - torch.log((torch.exp(- E + m) * E_mask).sum(1))) ** 2  # { unnormalized logprob(c_{0:n-1}|x) - log(sum_{c_n}(exp(unnormalized logprob(c_{0:n}|x)))) }^2                
        in_edge = torch.exp(log_pn)  # [B, 1]
        out_edges = torch.exp(- E + m) * E_mask
        out_edges_sum = out_edges.sum(dim=1, keepdim=True)  # [B, 1]
        MC_n_term = (torch.log(epsilon + in_edge) - torch.log(epsilon + out_edges_sum)) ** 2   # [B, 1]
        MC_n_term = MC_n_term.mean()
        log_pn = - torch.unsqueeze(E[:, cs[n]], 1) + m  # [B, 1], unnormalized logprob of p(c_{0:n} | x)
        
        # Compute the last MC term:
        if n == N - 1:
            in_edge_last = torch.exp(log_pn)
            reward = torch.tensor(10000).repeat(batch_size).to(E.device)  # [B,]. Fixed value used in the second term of the MC objective.
            last_MC_term = ((torch.log(epsilon + in_edge_last) - torch.log(epsilon + reward)) ** 2).mean() 
            MC_n_term = MC_n_term + last_MC_term
        
        # if it == 200 or it == 400 or it == 1200 or it == 1800:
        #     print('n: ', n)
        #     print('cs[n]: ', cs[n])
        #     print('E: ', E)
        #     print('in edge (prob): ', in_edge)
        #     print('out edges (probs): ', out_edges)
        #     if n == N - 1:
        #         print('#######################')
            
            # if n == N - 1:
            #     print('pseudo_LL:', pseudo_LL)
            #     print('out edges for N (probs): ', torch.exp(- E + m) * E_mask)
            # #     print('in_edge_last: ', in_edge_last)
            # #     print('reward: ', reward)
            
    return MC_n_term, log_pn   # MC_n_term is scalar, log_pn is [B, 1]