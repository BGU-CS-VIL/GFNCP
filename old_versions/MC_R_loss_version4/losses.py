import torch




def kl_loss_func(E):
    m, _ = torch.min(E, 1, keepdim=True)    # [B, 1]                  
    logprobs_kl = (-E + m) - torch.log(torch.exp(-E + m).sum(dim=1, keepdim=True))
    return logprobs_kl  # [B, K+1]


def j_loss_func(dpmm, E, c, hs, qs):
    # E is [B, K + 1]. This is the unnormalized logprob of p(c_{0:n} | x).
    # c is the ground-truth label of the N'th point (= cs[0, n])

    data_E = E[:, c] # .mean()  # scalar. E[:, cs[n]] is [B, 1], it's the unnormalized logprob of p(c_{0:N} | x) using the ground-truth label for the N-th point.         
    fake_E = dpmm.sample_for_J_loss(hs, qs)
    fake_E = fake_E # .mean()  # scalar. Mean over the minibatch
    j_loss = data_E - fake_E
        
    return j_loss, data_E  # all in shape [B, 1]
    
    
def mc_loss_func(E, log_pn, n, N, cs, batch_size, epsilon=1e-5):
    MC_n_term = torch.zeros(1).to(E.device)
    if n == 1:
        m, _ = torch.min(E, 1, keepdim=True)    # [B, 1] 
        log_pn = - torch.unsqueeze(E[:, cs[n]], 1) + m    # [B, 1], unnormalized logprob of p(c_{0:n} | x)
        
    elif n >= 2:
        m, _ = torch.min(E, 1, keepdim=True)    # [B, 1]
        # MC_n_term_0 = (log_pn - torch.log((torch.exp(- E + m)).sum(1))) ** 2  # { unnormalized logprob(c_{0:n-1}|x) - log(sum_{c_n}(exp(unnormalized logprob(c_{0:n}|x)))) }^2                
        in_edge = torch.exp(log_pn)  # [B, 1]
        out_edges = torch.exp(- E + m)
        out_edges_sum = out_edges.sum(dim=1, keepdim=True)  # [B, 1]
        MC_n_term = (torch.log(epsilon + in_edge) - torch.log(epsilon + out_edges_sum)) ** 2   # [B, 1]
        MC_n_term = MC_n_term.mean()
        log_pn = - torch.unsqueeze(E[:, cs[n]], 1) + m  # [B, 1], unnormalized logprob of p(c_{0:n} | x)
        
        # Compute the last MC term:
        if n == N - 1:
            R = 1 # Reward
            in_edge_last = torch.exp(log_pn)  # [B, 1]
            reward = torch.unsqueeze(torch.tensor(R).repeat(batch_size), 1).to(E.device)  # [B, 1]. Fixed value used in the second term of the MC objective.
            last_MC_term = ((torch.log(epsilon + in_edge_last) - torch.log(epsilon + reward)) ** 2).mean() 
            MC_n_term = MC_n_term + last_MC_term

    return MC_n_term, log_pn   # MC_n_term is scalar, log_pn is [B, 1]



def mc_r_loss_func(E, log_pn, n, N, cs, batch_size, dpmm, hs, qs, epsilon=1e-5, lambda_r=0.6):
    MC_n_term = torch.zeros(1).to(E.device)
    if n == 1:
        m, _ = torch.min(E, 1, keepdim=True)    # [B, 1] 
        log_pn = - torch.unsqueeze(E[:, cs[n]], 1) + m    # [B, 1], unnormalized logprob of p(c_{0:n} | x)
        
    elif n >= 2:
        m, _ = torch.min(E, 1, keepdim=True)    # [B, 1]
        # MC_n_term_0 = (log_pn - torch.log((torch.exp(- E + m)).sum(1))) ** 2  # { unnormalized logprob(c_{0:n-1}|x) - log(sum_{c_n}(exp(unnormalized logprob(c_{0:n}|x)))) }^2                
        in_edge = torch.exp(log_pn)  # [B, 1]
        out_edges = torch.exp(- E + m)
        out_edges_sum = out_edges.sum(dim=1, keepdim=True)  # [B, 1]
        MC_n_term = (torch.log(epsilon + in_edge) - torch.log(epsilon + out_edges_sum)) ** 2   # [B, 1]
        MC_n_term = MC_n_term.mean()
        log_pn = - torch.unsqueeze(E[:, cs[n]], 1) + m  # [B, 1], unnormalized logprob of p(c_{0:n} | x)
        
        # Compute the last MC term: Here the reward is alo learned.
        if n == N - 1:
            R = 1 # Reward
            in_edge_last = torch.exp(log_pn)  # [B, 1]
            reward = torch.unsqueeze(torch.tensor(R).repeat(batch_size), 1).to(E.device)  # [B, 1]. Fixed value used in the second term of the MC objective.
            last_MC_term = ((torch.log(epsilon + in_edge_last) - torch.log(epsilon + reward)) ** 2).mean() 
            MC_n_term = MC_n_term + last_MC_term
            
            # Add the J loss:
            true_E = - (log_pn - m)   # This is E[:, cs[n]] where n is the last point, using the ground-truth c(n)
            fake_E = dpmm.sample_for_J_loss(hs, qs)  # This is E[:, cs_samples[n]] where n is the last point, using sampled c(n)
            J_term = (lambda_r * (true_E - fake_E)).mean() 
            MC_n_term = MC_n_term + J_term
            
    return MC_n_term, log_pn   # MC_n_term is scalar, log_pn is [B, 1]