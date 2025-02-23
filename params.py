

def get_parameters(dataset_name):

    params = {}
    
    params['batch_size'] = 64
    params['max_it'] = 10000
    params['loss_str'] = 'MC_R'  # Choose from: {'MC + J', 'MC', 'MC_R', 'J', 'MC + J + KL', 'KL'}
    params['data_path'] = '/home/tohamy/Projects/data'  # options: [/home/tohamy/Projects/data, /vildata/tohamy/CPAB_Activation/data'
    
    params['lambda_cd'] = 0.2  # the weight for the contrastive-divergence loss (learned reward)
    params['lambda_reg'] = 2  # he weight for the regularization loss (fixed reward)
    params['beta_uniform'] = 0.9  # the probability for non-uniform run
    
    params['plot_freq'] = 100  # -1, 100..
    params['save_model_freq'] = 500
    params['iter_stats_avg'] = 1000  # from this iteration we start computing stats average (NMI, ARI, LL)
    params['class_split'] = False
    params['backbone'] = 'fc'  # options: [fc, vit]  # this is the backbone of E and G (where its input data is already encoded by "encoder_type")
    params['include_U'] = True   # True if we use the U (unclustered points) at each forward of the model
    params['unsup_flag'] = True   # If True, use pseudo-labels from augmentations during training. Otherwise, use ground-truth labels.
    params['tempr'] = 0.05  # also 0.4 is good.  # calibration temperature (used in the "sample" and "sample_for_J_loss" functions). Use 1 for no calibration
    params['K_fixed'] = -1
    
    params['eval_it'] = (3500, 4000, 4500, 5000, 5500)  # Iteration numbers to perform eval on. In case we want specific model, put -1 and use the regular checkpoint file. 
    
    # Optimation params:    
    params['lr'] = 0.0005
    params['min_lr'] = 1e-6
    params['sched_lr_update'] = 10  # update lr every x iterations.
    params['weight_decay'] = 0.0
    params['weight_decay_end'] = None
    
    params['alpha'] = 1.0  #.7  # Dispersion parameter of the Chinese Restaurant Process

    params['CIFAR100_TRAIN_MEAN'] = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)  # for data transform
    params['CIFAR100_TRAIN_STD'] = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)   # for data transform
    
    # Dimensions of G in the E network
    params['g_dim'] = 512
    
    # Params for Geweke's Test:
    params['N_Geweke'] = 30
    params['alpha_Geweke'] = 0.7
    params['M_Geweke'] = 3000
    params['N_range_Geweke'] = (5, 70)  # (5, 70)
    
    ''' Options for encoder types: [identity, fc, fc_and_attn, conv, conv_and_attn, attn, resnet18, resnet34]
        Required params for each encoder:
            fc: x_dim, H_dim, h_dim
            fc_and_attn: x_dim, H_dim, pre_attn_dim, h_dim
            conv: h_dim
            conv_and_attn: pre_attn_dim, h_dim
            attn: x_dim, h_dim
            resnet*: h_dim
    '''
    
    if dataset_name == 'Gauss2D':         
        params['x_dim'] = 2
        params['H_dim'] = 128
        params['pre_attn_dim'] = 256
        params['h_dim'] = 128
        params['sigma'] = 1        # std for the Gaussian noise around the cluster mean 
        params['lambda'] = 10      # std for the Gaussian prior that generates de centers of the clusters
        params['img_sz'] = None
        params['channels'] = 0
        params['encoder_type'] = 'fc'
        params['Nmin'] = 100
        params['Nmax'] = 1000
        params['Nmin_test'] = 300
        params['Nmax_test'] = 301
        
    elif dataset_name == 'MNIST':
        params['x_dim'] = 784  # when using fc_encoder directly on the images
        params['H_dim'] = 256
        params['pre_attn_dim'] = 256
        params['h_dim'] = 128  # what we usually use: conv: 256. conv_and_attn: 128
        params['img_sz'] = 28
        params['channels'] = 1
        params['nlabels'] = 10
        params['encoder_type'] = 'conv'
        params['Nmin'] = 100
        params['Nmax'] = 1000
        params['Nmin_test'] = 300
        params['Nmax_test'] = 301
    
    elif dataset_name == 'FASHIONMNIST':
        params['x_dim'] = 784  # when using fc_encoder directly on the images
        params['H_dim'] = 128
        params['pre_attn_dim'] = 256
        params['h_dim'] = 128 # 256
        params['img_sz'] = 28
        params['channels'] = 1
        params['nlabels'] = 10
        params['encoder_type'] = 'conv'
        params['Nmin'] = 100
        params['Nmax'] = 1000
        params['Nmin_test'] = 300
        params['Nmax_test'] = 301
        
    elif dataset_name == 'CIFAR':
        params['x_dim'] = 3072
        params['H_dim'] = 128
        params['pre_attn_dim'] = 256
        params['h_dim'] = 256
        params['img_sz'] = 32
        params['channels'] = 3
        params['input_dim'] = 32 * 32 * 3
        params['nlabels'] = 10
        params['reduce_dim_type'] = 't-SNE'
        params['data_path'] = params['data_path'] + '/CIFAR-10-images/' 
        params['encoder_type'] = 'resnet18'
        params['Nmin'] = 100
        params['Nmax'] = 1000
        params['Nmin_test'] = 20
        params['Nmax_test'] = 50

    elif dataset_name == 'STL':
        params['x_dim'] = 9216
        params['H_dim'] = 128
        params['pre_attn_dim'] = 256
        params['h_dim'] = 256
        params['img_sz'] = 96
        params['channels'] = 3
        params['input_dim'] = 96 * 96 * 3
        params['nlabels'] = 10
        params['reduce_dim_type'] = 't-SNE'
        params['data_path'] = params['data_path'] + '/CIFAR-10-images/' 
        params['encoder_type'] = 'resnet18'
        params['Nmin'] = 100
        params['Nmax'] = 1000
        params['Nmin_test'] = 20
        params['Nmax_test'] = 50
        
    elif dataset_name == 'IN50_ftrs':
        params['x_dim'] = 384
        params['H_dim'] = 128
        params['pre_attn_dim'] = 256
        params['h_dim'] = 128
        params['img_sz'] = None
        params['channels'] = 0
        params['nlabels'] = 50
        params['data_path'] = params['data_path'] + '/imagenet50_ftrs/'  
        params['encoder_type'] = 'fc'
        params['Nmin'] = 100
        params['Nmax'] = 1000
        params['Nmin_test'] = 49
        params['Nmax_test'] = 50

    elif dataset_name == 'IN100_ftrs':
        params['x_dim'] = 384
        params['H_dim'] = 128
        params['pre_attn_dim'] = 256
        params['h_dim'] = 128
        params['img_sz'] = None
        params['channels'] = 0
        params['nlabels'] = 100
        params['data_path'] = params['data_path'] + '/imagenet100_ftrs/'  
        params['encoder_type'] = 'fc'
        params['Nmin'] = 100
        params['Nmax'] = 1000
        params['Nmin_test'] = 49
        params['Nmax_test'] = 50

    elif dataset_name == 'IN200_ftrs':
        params['x_dim'] = 384
        params['H_dim'] = 128
        params['pre_attn_dim'] = 256
        params['h_dim'] = 128
        params['img_sz'] = None
        params['channels'] = 0
        params['nlabels'] = 199
        params['data_path'] = params['data_path'] + '/imagenet200_ftrs/'  
        params['encoder_type'] = 'fc'
        params['Nmin'] = 100
        params['Nmax'] = 1000
        params['Nmin_test'] = 49
        params['Nmax_test'] = 50
                
    elif dataset_name == 'CIFAR_ftrs':
        params['x_dim'] = 384
        params['H_dim'] = 128
        params['pre_attn_dim'] = 256
        params['h_dim'] = 128
        params['img_sz'] = None
        params['channels'] = 0
        params['nlabels'] = 10
        params['data_path'] = params['data_path'] + '/cifar10_features/'  
        params['encoder_type'] = 'fc'
        params['Nmin'] = 100  # 100
        params['Nmax'] = 200 # 1000
        params['Nmin_test'] = 200 # 300
        params['Nmax_test'] = 201 # 301
        
    elif dataset_name == 'tinyimagenet':
        params['x_dim'] = None
        params['H_dim'] = 128
        params['pre_attn_dim'] = 256
        params['h_dim'] = 256
        params['img_sz'] = 64
        params['channels'] = 3
        params['input_dim'] = 64 * 64 * 3
        params['nlabels'] = 200
        params['reduce_dim_type'] = 't-SNE'
        params['data_path'] = params['data_path'] + '/tiny_imagenet/tiny-imagenet-200/'
        params['encoder_type'] = 'resnet18'
        params['Nmin'] = 50
        params['Nmax'] = 500
        params['Nmin_test'] = 20
        params['Nmax_test'] = 50
        
    else:
        raise NameError('Unknown dataset_name: '+ dataset_name)
        
    return params