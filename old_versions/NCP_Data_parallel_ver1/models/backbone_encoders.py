
import torch.nn as nn
import torch
from models.resnet import resnet18, resnet34
import torch.nn.functional as F


def get_encoder(params):
    encoder = None
        
    if params['encoder_type'] == 'fc':
        encoder = fc_encoder(params)
        # encoder = encoder.to(torch.device('cuda'))
    elif params['encoder_type'] == 'mnist_encoder':
        encoder = MNIST_encoder(params)
        # encoder = encoder.to(torch.device('cuda'))
    elif params['encoder_type'] == 'resnet18':
        encoder = resnet18(params)   
        # encoder = encoder.to(torch.device('cuda'))
    elif params['encoder_type'] == 'resnet34':
        encoder = resnet34(params)
        # encoder = encoder.to(torch.device('cuda'))
    elif params['encoder_type'] == 'identity':
        encoder = nn.Identity()
        # encoder = encoder.to(torch.device('cuda'))
    else:
        raise NameError('Unknown encoder type ' + params['encoder_type'])
        
    return encoder
        

    
class fc_encoder(nn.Module):
    
    def __init__(self, params):
        super(fc_encoder, self).__init__()
        
        H = params['H_dim']
        self.h_dim = params['h_dim']        
        self.x_dim = params['x_dim']
        
        self.h = torch.nn.Sequential(
                torch.nn.Linear(self.x_dim, H),
                torch.nn.ReLU(),
                torch.nn.Linear(H, H),
                torch.nn.ReLU(),    
                torch.nn.Linear(H, H),
                torch.nn.ReLU(),
                torch.nn.Linear(H, H),
                torch.nn.ReLU(),
                torch.nn.Linear(H, self.h_dim),
                )

    def forward(self, x):
        return self.h(x)



class MNIST_encoder(nn.Module):
    
    def __init__(self, params):  
        super(MNIST_encoder, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 256)
        self.fc2 = nn.Linear(256, params['h_dim'])

    def forward(self, x):
        '''
            Input: [B * N, 28, 28]
            Output: [B * N, h]
        '''
        
        x = x.unsqueeze(1)   # add channel index
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
    


class CIFAR_encoder(nn.Module):
    # !!! NOT WORKING !!!   
    def __init__(self, params):
        super(CIFAR_encoder, self).__init__()
        
        self.conv1 = nn.Conv2d(params['channels'], 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 256)
        self.fc2 = nn.Linear(256, params['h_dim'])

    def forward(self, x):
        '''
            Input: [B * N, 3, 28, 28]
            Output: [B * N, h]
        '''

        # x = x.unsqueeze(1)   # add channel index
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)  # !! THIS SIZE IS WRONG !!
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
    
    

