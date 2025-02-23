
import torch.nn as nn
import torch
from models.resnet import resnet18, resnet34
from models.set_transformer import ISAB, SAB
import torch.nn.functional as F


def get_encoder(params):
    encoder = None
        
    if params['encoder_type'] == 'fc':
        encoder = fc_encoder(params)
        # encoder = encoder.to(torch.device('cuda'))
    elif params['encoder_type'] == 'fc_and_attn':
        encoder = fc_and_attn_encoder(params)
        # encoder = encoder.to(torch.device('cuda'))
    elif params['encoder_type'] == 'conv':
        encoder = conv_encoder(params)
        # encoder = encoder.to(torch.device('cuda'))
    elif params['encoder_type'] == 'conv_and_attn':
        encoder = conv_and_attn_encoder(params)
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
    elif params['encoder_type'] == 'attn':
        encoder = attn_encoder(params)
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
        
        self.enc = torch.nn.Sequential(
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
        ''' 
            x: [B, N, x_dim] 
            out: [B, N, h_dim]
        '''
        B = x.shape[0]
        N = x.shape[1]
        x = x.reshape(tuple((-1,)) + tuple(x.shape[2:])) # will be: [B*N, channels, img_sz, img_sz] or [B*N, img_sz, img_sz] or [B*N, h_dim]
        x = x.view(x.size(0), -1)
        out = self.enc(x).reshape([B, N, self.h_dim])  # [B, N, h_dim]  
        return out


class fc_and_attn_encoder(nn.Module):
    
    def __init__(self, params, num_heads=4, num_inds=32, ln=False):
        super(fc_and_attn_encoder, self).__init__()
        
        x_dim = params['x_dim']
        H = params['H_dim']
        pre_attn_dim = params['pre_attn_dim']
        h_dim = params['h_dim'] 
        
        self.enc = torch.nn.Sequential(
                torch.nn.Linear(x_dim, H),
                torch.nn.ReLU(),
                torch.nn.Linear(H, H),
                torch.nn.ReLU(),    
                torch.nn.Linear(H, H),
                torch.nn.ReLU(),
                torch.nn.Linear(H, H),
                torch.nn.ReLU(),
                torch.nn.Linear(H, pre_attn_dim),
                )
        
        if params['Nmin'] >= 100:
            self.attn_enc = nn.Sequential(
                ISAB(pre_attn_dim, h_dim, num_heads, num_inds, ln=ln),
                ISAB(h_dim, h_dim, num_heads, num_inds, ln=ln))
        else:
            self.attn_enc = nn.Sequential(
                SAB(pre_attn_dim, h_dim, num_heads, ln=ln),
                SAB(h_dim, h_dim, num_heads, ln=ln))
        
    def forward(self, x):
        ''' 
            x: [B, N, x_dim] 
            out: [B, N, h_dim]
        '''
        B = x.shape[0]
        N = x.shape[1]
        x = x.reshape(tuple((-1,)) + tuple(x.shape[2:])) # will be: [B*N, channels, img_sz, img_sz] or [B*N, img_sz, img_sz] or [B*N, h_dim]
        x = x.view(x.size(0), -1)
        x = self.enc(x)  # [B*N, pre_attn_dim]  
        x = x.reshape([B, N, -1])  # [B, N, pre_attn_dim]  
        out = self.attn_enc(x) # [B, N, h_dim]  
        return out
    

class conv_encoder(nn.Module):
    
    def __init__(self, params):  
        super(conv_encoder, self).__init__()
        
        fc_dim = 320
        if params['img_sz'] == 32:
            fc_dim = 500
            
        self.conv1 = nn.Conv2d(params['channels'], 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(fc_dim, 256)
        self.fc2 = nn.Linear(256, params['h_dim'])

    def forward(self, x):
        ''' 
            x: [B, N, x_dim] 
            out: [B, N, h_dim]
        '''
        
        B = x.shape[0]
        N = x.shape[1]
        x = x.reshape(tuple((-1,)) + tuple(x.shape[2:])) # will be: [B*N, channels, img_sz, img_sz] or [B*N, img_sz, img_sz] or [B*N, x_dim]
 
        if len(x.shape) < 4:
            x = x.unsqueeze(1)   # add channel index
        
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)  # [B*N, h_dim]  
        out = x.reshape([B, N, -1])  # [B, N, h_dim]  
        return out
    

class conv_and_attn_encoder(nn.Module):
    
    def __init__(self, params, num_heads=4, num_inds=32, ln=False):
        super(conv_and_attn_encoder, self).__init__()
        
        pre_attn_dim = params['pre_attn_dim']
        h_dim = params['h_dim']
        
        fc_dim = 320
        if params['img_sz'] == 32:
            fc_dim = 500
            
        self.conv1 = nn.Conv2d(params['channels'], 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(fc_dim, 256)
        self.fc2 = nn.Linear(256, pre_attn_dim)

        if params['Nmin'] >= 100:
            self.attn_enc = nn.Sequential(
                ISAB(pre_attn_dim, h_dim, num_heads, num_inds, ln=ln),
                ISAB(h_dim, h_dim, num_heads, num_inds, ln=ln))
        else:
            self.attn_enc = nn.Sequential(
                SAB(pre_attn_dim, h_dim, num_heads, ln=ln),
                SAB(h_dim, h_dim, num_heads, ln=ln))
            
        
        
    def forward(self, x):
        ''' 
            x: [B, N, x_dim] 
            out: [B, N, h_dim]
        '''
        
        B = x.shape[0]
        N = x.shape[1]
        x = x.reshape(tuple((-1,)) + tuple(x.shape[2:])) # will be: [B*N, channels, img_sz, img_sz] or [B*N, img_sz, img_sz] or [B*N, x_dim]
   
        if len(x.shape) < 4:
            x = x.unsqueeze(1)   # add channel index
        
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x).reshape([B, N, -1])  # [B, N, pre_attn_dim] 
        out = self.attn_enc(x)  # [B, N, h_dim] 
        return out
    

class attn_encoder(nn.Module):
    
    def __init__(self, params, num_heads=4, num_inds=32, ln=False):
        super(attn_encoder, self).__init__()
        
        x_dim = params['x_dim']
        h_dim = params['h_dim']

        if params['Nmin'] >= 100:
            self.attn_enc = nn.Sequential(
                ISAB(x_dim, h_dim, num_heads, num_inds, ln=ln),
                ISAB(h_dim, h_dim, num_heads, num_inds, ln=ln))
        else:
            self.attn_enc = nn.Sequential(
                SAB(x_dim, h_dim, num_heads, ln=ln),
                SAB(h_dim, h_dim, num_heads, ln=ln))

    def forward(self, x):
        ''' 
            x: [B, N, x_dim] 
            out: [B, N, h_dim]
        '''
        x = x.view(x.size(0), x.size(1), -1)
        out = self.attn_enc(x)
        return out