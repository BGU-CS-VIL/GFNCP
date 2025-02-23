

def get_parameters(dataset_name):

    params = {}
    
    params['batch_size'] = 64
    params['max_it'] = 5000
    params['loss_str'] = 'KL'  # Choose from: {'MC + J', 'MC', 'J', 'MC + J + KL', 'KL'}
    params['data_path'] = '/vildata/tohamy/CPAB_Activation/data'  # options: [/home/tohamy/Projects/data, /vildata/tohamy/CPAB_Activation/data'
    
    params['lambda_mc'] = 1
    params['lambda_j'] = 1
    params['lambda_entrpy'] = 0
    params['mc_weights'] = {}  #{200:2, 300:3, 400:4, 500:5, 600:6, 700:7, 1000:10}
    params['j_weights'] = {}  #{200:2, 300:3, 400:4, 500:5, 600:6, 700:7, 1000:10}
    params['plot_freq'] = 100
    params['class_split'] = False
    
    # Optimation params:    
    params['lr'] = 0.0005
    params['min_lr'] = 1e-5
    params['sched_lr_update'] = 10  # update lr every x iterations.
    params['weight_decay'] = 0.0
    params['weight_decay_end'] = None
    
    params['alpha'] = .7  # Dispersion parameter of the Chinese Restaurant Process
    
    params['Nmin'] = 5
    params['Nmax'] = 100
    params['N_sampling'] = 20
        
    if dataset_name == 'Gauss2D':         
        params['sigma'] = 1        # std for the Gaussian noise around the cluster mean 
        params['lambda'] = 10      # std for the Gaussian prior that generates de centers of the clusters
        params['x_dim'] = 2
        params['h_dim'] = 256
        params['g_dim'] = 512
        params['H_dim'] = 128
        params['img_sz'] = None
        params['channels'] = 0
        
    elif dataset_name == 'MNIST':
        params['h_dim'] = 256
        params['g_dim'] = 512
        params['H_dim'] = 128
        params['img_sz'] = 28
        params['channels'] = 1
        params['nlabels'] = 10
    
    elif dataset_name == 'FASHIONMNIST':
        params['h_dim'] = 256
        params['g_dim'] = 512
        params['H_dim'] = 128
        params['img_sz'] = 28
        params['channels'] = 1
        params['nlabels'] = 10
    
    elif dataset_name == 'CIFAR':
        params['h_dim'] = 256
        params['g_dim'] = 512
        params['H_dim'] = 128
        params['img_sz'] = 32
        params['channels'] = 3
        params['input_dim'] = 32 * 32 * 3
        params['nlabels'] = 10
        params['reduce_dim_type'] = 't-SNE'
        params['CIFAR100_TRAIN_MEAN'] = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)  # for data transform
        params['CIFAR100_TRAIN_STD'] = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)   # for data transform
        params['data_path'] = '/vildata/tohamy/CPAB_Activation/data/CIFAR10_imgs/'  # options: [/vildata/tohamy/CPAB_Activation/data/CIFAR10_imgs/, /home/tohamy/Projects/data/CIFAR-10-images/]
        
    elif dataset_name == 'Features':
        params['h_dim'] = 384
        params['g_dim'] = 512
        params['H_dim'] = 128
        params['img_sz'] = None
        params['channels'] = 0
        params['nlabels'] = 50
        params['data_path'] = '/vildata/tohamy/data/imagenet50_featutres/'  # options: [/vildata/tohamy/data/imagenet50_featutres/, /home/tohamy/Projects/data/imagenet50_featutres/]
        
    else:
        raise NameError('Unknown dataset_name: '+ dataset_name)
        
    return params