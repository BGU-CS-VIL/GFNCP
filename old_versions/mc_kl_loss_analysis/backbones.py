
import torch.nn as nn
import torch


class FC(nn.Module):
    
    def __init__(self, params, input_dim=10, output_dim=1, layers_num=6, last_layer_bias=True):    
        super(FC, self).__init__()
        
        H = params['H_dim']  
        self.backbone = torch.nn.Sequential(
                torch.nn.Linear(input_dim, H),
                torch.nn.ReLU(),    
                torch.nn.Linear(H, H),
                torch.nn.ReLU(),
                torch.nn.Linear(H, H),
                torch.nn.ReLU(),    
                torch.nn.Linear(H, H),
                torch.nn.ReLU(),
                torch.nn.Linear(H, H),
                torch.nn.ReLU(),
                torch.nn.Linear(H, output_dim, bias=last_layer_bias),
                )
        
    
    def forward(self, x):
        out = self.backbone(x)
        return out