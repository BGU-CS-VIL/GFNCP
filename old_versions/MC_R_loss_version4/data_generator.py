import numpy as np
import torch
from torchvision import datasets, transforms
from utils import relabel, GaussianBlur, Solarization, DropOutFeatures, GaussianNoise
from PIL import Image
import matplotlib.pyplot as plt



def get_generator(params):
    
    if params['dataset_name'] == 'Gauss2D':         
        return gauss2dGenerator(params)
    else:
        return dataGenerator(params) 
    
    
class dataGenerator():
    
    def __init__(self, params, train=True):
        self.img_sz = params['img_sz']
        self.channels = params['channels']
        self.nlabels = params['nlabels']
        self.x_dim = params['x_dim']
        self.params = params
        
        self.augment = DataAugmentation(self.img_sz, self.channels, self.x_dim, self.params['dataset_name'])
        
        # Extract the train and test data:
        self.dataset_train, _ = get_dataset(self.params, train=True, unsup=params['unsup_flag'])
        self.dataset_test, _ = get_dataset(self.params, train=False, unsup=False)
            # "dataset_train/test" is a list of tuples of (x, label) with shapes e.g.: ([1, 28, 28], scalar int) 
        
        self.dataset_train_size = len(self.dataset_train)
        self.dataset_test_size = len(self.dataset_test)
        
        # Prepare self.label_data_train_map and self.label_data_map_test:
        self.label_data_map_train, label_counts_train = self.prepare_label_data_map(train=True)
        self.label_data_map_test, label_counts_test = self.prepare_label_data_map(train=False)
            # Each map is a dictionary of groups of images per class. E.g: entry 0 in will be of shape (M, ..data_shape..) where M is the number of images from class 0.
        
        # Update the Nmax values based on the size of the smallest class:
        params['Nmax'] = int(min(params['Nmax'], np.min(label_counts_train)))
        params['Nmax_test'] = int(min(params['Nmax_test'], np.min(label_counts_test)))
        if params['Nmin'] >= params['Nmax'] or params['Nmin_test'] >= params['Nmax_test']:
            raise NameError('Nmin/Nmin_test are bigger than Nmax/Nmax_test')
        
        print('Actual N values used during training: ({0}, {1})'.format(params['Nmin'], params['Nmax']))
        print('Actual N values used during testing: ({0}, {1})'.format(params['Nmin_test'], params['Nmax_test']))

        # Perform class splitting if needed:
        if self.params['class_split']:
            print('Before class split (train.keys, test.keys):', self.label_data_map_train.keys(), self.label_data_map_test.keys())
            split_ind = self.nlabels // 2
            self.label_data_map_train = dict(list(self.label_data_map_train.items())[:split_ind])
            self.label_data_map_test = dict(list(self.label_data_map_test.items())[split_ind:])
            
            # relabel the test set to start from 0:
            for new_key, old_key in enumerate(range(split_ind, self.nlabels)):
                self.label_data_map_test[new_key] = self.label_data_map_test.pop(old_key)
            
            print('After class split (train.keys, test.keys):', self.label_data_map_train.keys(), self.label_data_map_test.keys())
            
            
    def prepare_label_data_map(self, train=True):
        
        if train:
            print('Preparing train data...')
            dataset = self.dataset_train
        else:
            print('Preparing test data...')
            dataset = self.dataset_test
            
        all_labels = np.zeros(len(dataset), dtype=np.int32)
        label_counts = np.zeros(self.nlabels)
        for i in range(len(dataset)):
            all_labels[i] = dataset[i][1]
                   
        # Create a list of groups of images per class.
        # E.g: entry 0 in "label_data_map" will be of shape (M, 28, 28) where M is the number of images from class 0.
        label_data_map = {}
        for i in range(self.nlabels):
            label_inds = np.nonzero(all_labels == i)[0]   # Returns all indices in "all_labels" that their label value is i         
            S = label_inds.shape[0]  # S is the number of data points assigned to label i
            label_counts[i] = S
            print('Processing label ', i, ' with ', S, ' data points.')
            
                # For extracted-features input
            if self.channels == 0 and self.params['dataset_name'] == 'Features': 
                label_data_map[i] = torch.zeros([S, self.x_dim])
                for s in range(S):
                    label_data_map[i][s, :] = dataset[label_inds[s]][0][:]
                    
                # for black and white images
            elif self.channels == 1: 
                label_data_map[i] = torch.zeros([S, self.img_sz, self.img_sz])
                for s in range(S):
                    label_data_map[i][s, :, :] = dataset[label_inds[s]][0][0, :, :]
                    
                # for RGB images 
            else: 
                label_data_map[i] = torch.zeros([S, self.channels, self.img_sz, self.img_sz])
                for s in range(S):
                    label_data_map[i][s, :, :, :] = dataset[label_inds[s]][0][:, :, :]

        return label_data_map, label_counts
    
    
    def generate(self, N=None, batch_size=1, train=True, unsup=True):
        '''
            Output:
                data: B groups of data points of the same mixture (but the data content is different). [B, N, data_shape]
                cs: ground-truth labels of the mixture, repeated B times. [B, N]
            
            If train == True and unsup == True: 
                we extract augmented images using the CRP mixture
            If train == True and unsup == False: 
                we extract data from label_data_map_train (based on ground-truth labels)
            If train == False: 
                we extract data from label_data_map_test (based on ground-truth labels)
        '''
        
        if train:
            if not unsup:
                label_data_map = self.label_data_map_train
            else:
                label_data_map = None
        else:
            label_data_map = self.label_data_map_test
            
        # Number of clusters to sample from:
        L = self.params['nlabels']
        K = L + 1
        while K > L:  # Generate clustering according to CRP with K < L.
            clusters, N, K = generate_CRP(self.params, N=N, train=train)  
            # "clusters": is in shape [N+2]. Entry i (from 1..N+1) holds the number of points assigned to label i. Sum over clusters should be equal to N.
        
        # Prepare "data" tensor:     
        if self.channels == 0 and self.params['dataset_name'] == 'Features':  # For extracted-features input
            data = torch.zeros([batch_size, N, self.x_dim])    
        elif self.channels == 1:  # for black and white images
            data = torch.zeros([batch_size, N, self.img_sz, self.img_sz])
        else: # for RGB images
            data = torch.zeros([batch_size, N, self.channels, self.img_sz, self.img_sz])     
            
        cumsum = np.cumsum(clusters)  # Cumulative sum over clusters. Shape: [N+2]
        
        # Fill in "data" and "cs": 
        #   "data": shape: [B, N, 28, 28]. Each point in B is a dataset of N images with the same mixture of K clusters (but with different classes).
        #   "cs": shape: [N]. This is a list of pseudo-labels of images from "data", which are relevant to all B point.
        # cmap = plt.cm.gray
        # print('K:', K)
        for i in range(batch_size):
            labels = np.random.choice(L, size=K, replace=False)  #this is a sample from the 'base measure' for each cluster (only for the supervised version and test time)
            for k in range(K):
                nk = clusters[k + 1]
                
                if train and unsup:
                    # sample image from the entire dataset:
                    ind = np.random.choice(self.dataset_train_size, size=1, replace=False)
                    img = self.dataset_train[ind[0]][0][:]
                    
                    # img_np = torch.clone(torch.squeeze(img)).cpu().detach().numpy()
                    # im = cmap(img_np)
                    # plt.imsave('tmp/orig_img_k' + str(k) + '.jpeg', im)
                    
                    # prepare m augmentations of the image (according to the mixture):
                    prepared_data = self.augment(img, aug_number=nk)   # shape: [nk, 3, 28, 28] or [nk, 28, 28] or [nk, h]
                
                else:  # during test time / train in supervised manner
                    l = labels[k]
                    inds = np.random.choice(label_data_map[l].shape[0], size=nk, replace=False) 
                    prepared_data = label_data_map[l][inds, :]   # shape: [nk, 3, 28, 28] or [nk, 28, 28] or [nk, h]
                
                # Insert the prepared data chunk to the "data" tensor in the right places
                data[i, cumsum[k]:cumsum[k + 1], :] = prepared_data
                

        cs = np.empty(N, dtype=np.int32)     
        for k in range(K):
            cs[cumsum[k]:cumsum[k + 1]]= k
        
        # Shuffle "data" and "cs" in the same way:
        arr = np.arange(N)
        np.random.shuffle(arr)
        cs = cs[arr]   
        data = data[:, arr, :]
        
        # Relabel cluster numbers so that they appear in order
        cs = relabel(cs)
        
        cs = torch.tensor(cs)
        cs = cs.repeat(data.shape[0], 1)  # [B, N] where all rows are the same
        return data, cs, clusters, K    # e.g.: data: [B, N, 28, 28], cs: [B, N], clusters: [N+2]
    

    def get_full_test_data(self):
        
        N = len(self.dataset_test)
        B = 1
        print('full test-data size:', N)
        
                    # For extracted-features input
        if self.channels == 0 and self.params['dataset_name'] == 'Features': 
            data = torch.zeros([B, N, self.x_dim])    
            # for black and white images
        elif self.channels == 1: 
            data = torch.zeros([B, N, self.img_sz, self.img_sz])
            # for RGB images
        else:
            data = torch.zeros([B, N, self.channels, self.img_sz, self.img_sz])  
    
        cs = np.zeros(N, dtype=np.int32)
        for i in range(N):
            data[0, i, :] = self.dataset_test[i][0]
            cs[i] = self.dataset_test[i][1]
            
        # Shuffle "data" and "cs" in the same way:
        arr = np.arange(N)
        np.random.shuffle(arr)
        cs = cs[arr]   
            # For extracted-features input
        if self.channels == 0 and self.params['dataset_name'] == 'Features': 
            data = data[:, arr, :]      
            # for black and white images
        elif self.channels == 1: 
            data = data[:, arr, :, :]
            # for RGB images
        else:
            data = data[:, arr, :, :, :]
        
        # Relabel cluster numbers so that they appear in order
        cs = relabel(cs)
        
        return data, cs
        
        
        
        
class gauss2dGenerator():
    
    def __init__(self, params):
        self.params = params
        

    def generate(self, N=None, batch_size=1, train=True):        
        
        lamb = self.params['lambda']
        sigma = self.params['sigma']
        x_dim = self.params['x_dim']    
        
        clusters, N, num_clusters = generate_CRP(self.params, N=N, train=train)
        
        cumsum = np.cumsum(clusters)  # Cumulative sum. Shape: [N+2]
        data = np.empty([batch_size, N, x_dim])
        cs =  np.empty(N, dtype=np.int32)
        
        for i in range(num_clusters):
            mu= np.random.normal(0,lamb, size = [x_dim*batch_size,1])
            samples= np.random.normal(mu,sigma, size=[x_dim*batch_size,clusters[i+1]] )
            
            samples = np.swapaxes(samples.reshape([batch_size, x_dim,clusters[i+1]]),1,2)        
            data[:,cumsum[i]:cumsum[i+1],:]  = samples
            cs[cumsum[i]:cumsum[i+1]]= i+1
            
        #%shuffle the assignment order
        arr = np.arange(N)
        np.random.shuffle(arr)
        cs = cs[arr]
        
        data = data[:,arr,:]
        
        # relabel cluster numbers so that they appear in order 
        cs = relabel(cs)
        cs = torch.tensor(cs)
        cs = cs.repeat(data.shape[0], 1)  # [B, N] where all rows are the same
        
        #normalize data 
        #means = np.expand_dims(data.mean(axis=1),1 )    
        medians = np.expand_dims(np.median(data,axis=1),1 )    
        
        data = data-medians
        data = torch.tensor(data).float()
        #data = 2*data/(maxs-mins)-1        #data point are now in [-1,1]
        
        # data = torch.tensor(data).float().to(torch.device('cuda'))
        # data = torch.tensor(data).float()

        return data, cs, clusters, num_clusters
    


def generate_CRP(params, N=None, train=True, no_ones=False):
    # Group the data according to CRP (assign each point to a cluster, N is random and K is sampled from the CRP)

    alpha = params['alpha']   # dispersion parameter of the Chinese Restaurant Process
    crp_not_done = True
    
    while crp_not_done:
        if N is None or N==0:
            if train:
                N = np.random.randint(params['Nmin'], params['Nmax'])
            else:
                N = np.random.randint(params['Nmin_test'], params['Nmax_test'])
                
        clusters = np.zeros(N+2)
        clusters[0] = 0
        clusters[1] = 1      # we start filling the array here in order to use cumsum below 
        clusters[2] = alpha
        index_new = 2
        for n in range(N-1):     #we loop over N-1 particles because the first particle was assigned already to cluster[1]
            p = clusters/clusters.sum()
            z = np.argmax(np.random.multinomial(1, p))
            if z < index_new:  # Assign point n to an existing cluster
                clusters[z] +=1
            else:              # Assign point n to a new cluster
                clusters[index_new] =1
                index_new +=1
                clusters[index_new] = alpha
        
        clusters[index_new] = 0 
        clusters = clusters.astype(np.int32)
        
        if no_ones:
            clusters= clusters[clusters!=1]
        N = int(np.sum(clusters))
        crp_not_done = N==0                       
        
        
    K = np.sum(clusters>0)

    return clusters, N, K
    
    

def get_dataset(params,
                train=True,
                lsun_categories=None,
                deterministic=False,
                transform=None,
                unsup=True):
    
    data_name = params['dataset_name']
    data_dir = params['data_path']
    size = params['img_sz']

    if data_name != 'Features':
        mnist_transform = transforms.Compose([
                                    transforms.Resize(size),
                                    transforms.CenterCrop(size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, ), (0.5, ))
                                    ])
        
        cifar_transform = cifar_transform = transforms.Compose([
                #transforms.ToPILImage(),
                # transforms.RandomCrop(32, padding=4),
                transforms.CenterCrop(size),
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation(15),
                transforms.ToTensor(),
                # transforms.Normalize((0.5, ), (0.5, ))
                transforms.Normalize(params['CIFAR100_TRAIN_MEAN'], params['CIFAR100_TRAIN_STD'])
            ])


    if data_name == 'MNIST':
        dataset = datasets.MNIST(data_dir,
                                 transform=mnist_transform,
                                 train=train,
                                 download=True)
        nlabels = 10
    
    elif data_name == 'FASHIONMNIST':
        dataset = datasets.FashionMNIST(data_dir,
                                    transform=mnist_transform,
                                    train=train,
                                    download=True)
        nlabels = 10
      
    elif data_name == 'CIFAR':
        print("reading from datapath ", data_dir)
        root = data_dir + 'train' if train else data_dir + 'test'
        dataset = datasets.ImageFolder(root, transform=cifar_transform)
        nlabels = params['nlabels']

        # dataset = datasets.CIFAR10(root=data_dir, train=train, download=True, transform=cifar_transform)
        # nlabels = 10
        
    elif data_name == 'CIFAR100':
        dataset = datasets.CIFAR100(data_dir, train=train, transform=cifar_transform, download=True)
        nlabels = 100

    elif data_name == 'tinyimagenet':
        print("reading from datapath ", data_dir)
        root = data_dir + 'train' if train else data_dir + 'val'
        dataset = datasets.ImageFolder(root, transform=cifar_transform)
        nlabels = params['nlabels']
        
    elif data_name == 'STL':
        if train:
            split = 'train'
        else:
            split = 'test'
            
        dataset = datasets.STL10(root=data_dir, split=split, download=True,
                                transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ]))
        nlabels = len(dataset.classes)
    
    elif data_name == 'Features':
        # Load data and labels from .pt files, it should be in shape: data=[len(dataset), x_dim], labels=[len(dataset),]
        # dataset object is a list of tuples of (x, c) which is data and label.
        
        dataset = {}
        if train:
            x = torch.load(data_dir + 'embeddings.pt') # [N, x_dim]
            c = torch.load(data_dir + 'label.pt')  # [N,]
        else:
            x = torch.load(data_dir + 'embeddings-test.pt') # [N, x_dim]
            c = torch.load(data_dir + 'label-test.pt')   # [N,]  
        
        for i in range(len(c)):
            dataset[i] = (x[i], c[i])

        nlabels = torch.unique(c)    
                   
    else:
        raise NameError('Unknown dataset_name ' + data_name)
    
    return dataset, nlabels




class DataAugmentation(object):
    def __init__(self, img_size, channels, x_dim, dataset_name, scale=(0.5, 1.0)):
        
        self.channels = channels
        self.img_size = img_size
        self.x_dim = x_dim
        self.dataset_name = dataset_name
        self.to_pil_image = transforms.ToPILImage()  

        # Several augmentation options:
        
        if channels == 0:
            self.augment = transforms.Compose([
                GaussianNoise(std=0.05),
                DropOutFeatures(p=0.2),   
            ])
        
        elif channels == 1:  # BW images
            normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, ), (0.5, )),
            ])
            
            self.augment = transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=scale, interpolation=Image.BICUBIC),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(0.1),
                normalize,
            ])
        
        else: # RGB images
            normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            
            flip_and_color_jitter = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
            ])
                    
            self.augmentation1 = transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                GaussianBlur(1.0),
                normalize,
            ])
            
            self.augmentation2 = transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                GaussianBlur(0.1),
                Solarization(0.2),
                normalize,
            ])
            
            self.augment = self.augmentation2
        

        
    def __call__(self, x, aug_number):  
           
        # Prepare "aug_data" tensor:     
        if self.channels == 0 and self.dataset_name == 'Features':  # For extracted-features input
            aug_data = torch.zeros([aug_number, self.x_dim])   
        elif self.channels == 1:  # for black and white images
            x = self.to_pil_image(x)   
            aug_data = torch.zeros([aug_number, self.img_size, self.img_size])
        else: # for RGB images
            x = self.to_pil_image(x)   
            aug_data = torch.zeros([aug_number, self.channels, self.img_size, self.img_size])     
        
        # cmap = plt.cm.gray
        for i in range(aug_number):
            x_aug = self.augment(x)

            # img_np = torch.clone(torch.squeeze(ag)).cpu().detach().numpy()
            # im = cmap(img_np)
            # plt.imsave('tmp/aug_img_i' + str(i) + '.jpeg', im)
            
            if self.channels == 1:
                x_aug = torch.squeeze(x_aug)
                
            aug_data[i, :] = x_aug
        
        return aug_data   # shape: [aug_number, 3, 28, 28] or [aug_number, 28, 28] or [aug_number, h]
