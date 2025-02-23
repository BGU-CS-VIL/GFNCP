

def get_parameters(dataset_name):

    params = {}
    
    params['batch_size'] = 64
    params['max_it'] = 5000
    params['loss_str'] = 'J'  # Choose from: {'MC + J', 'MC', 'J', 'MC + J + KL', 'KL'}
    params['is_old_kl'] = False
    params['data_path'] = '/vildata/tohamy/CPAB_Activation/data'
    
    params['lambda_mc'] = 1
    params['lambda_j'] = 1
    params['lambda_entrpy'] = 0
    params['mc_weights'] = {}  #{200:2, 300:3, 400:4, 500:5, 600:6, 700:7, 1000:10}
    params['j_weights'] = {}  #{200:2, 300:3, 400:4, 500:5, 600:6, 700:7, 1000:10}
    params['plot_freq'] = 100
    
    # Optimation params:    
    params['lr'] = 0.0005
    params['min_lr'] = 1e-6
    params['sched_lr_update'] = 10  # update lr every x iterations.
    params['weight_decay'] = 0.0
    params['weight_decay_end'] = None
    
    if dataset_name == 'MNIST':
        params['alpha'] = .7
        params['Nmin'] = 5
        params['Nmax'] = 100
        params['h_dim'] = 256
        params['g_dim'] = 512
        params['H_dim'] = 128
        params['img_sz'] = 28
        params['channels'] = 1
    
    elif dataset_name == 'FASHIONMNIST':
        params['alpha'] = .7
        params['Nmin'] = 5
        params['Nmax'] = 100
        params['h_dim'] = 256
        params['g_dim'] = 512
        params['H_dim'] = 128
        params['img_sz'] = 28
        params['channels'] = 1
    
    elif dataset_name == 'CIFAR':
        params['alpha'] = .7
        params['Nmin'] = 5
        params['Nmax'] = 100
        params['h_dim'] = 256
        params['g_dim'] = 512
        params['H_dim'] = 128
        params['img_sz'] = 28
        params['channels'] = 3
                
    elif dataset_name == 'Gauss2D':         
        params['alpha'] = .7
        params['sigma'] = 1        # std for the Gaussian noise around the cluster mean 
        params['lambda'] = 10      # std for the Gaussian prior that generates de centers of the clusters
        params['Nmin'] = 5
        params['Nmax'] = 100
        params['x_dim'] = 2
        params['h_dim'] = 256
        params['g_dim'] = 512
        params['H_dim'] = 128
        params['img_sz'] = None
        params['channels'] = 0
        
    else:
        raise NameError('Unknown dataset_name: '+ dataset_name)
        
    return params