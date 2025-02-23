'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class BasicBlock(nn.Module):
    expansion = 1
                
    def __init__(self, act_func, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
        
        if act_func == 'relu':
            self.act = F.relu
        elif act_func == 'leaky_relu':
            self.act = F.leaky_relu
            
        # self.conv_theta = nn.Conv2d(in_planes, 16, kernel_size=3, stride=2, padding=1, bias=False)
        # self.activation_func=nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out1 = self.act(out)    
        out = self.conv2(out1)
        out = self.bn2(out)
        out += self.shortcut(x)
        out1 = self.act(out)

        return out1


class ResNet(nn.Module):
    def __init__(self, params, block, num_blocks, num_classes=100):
        super(ResNet, self).__init__()
        
        self.img_nc = params['channels']
        self.img_sz = params['img_sz']
        self.act_func = 'relu'  # choose from ['relu', 'leaky_relu']
        self.in_planes = 64

        self.conv1 = nn.Conv2d(self.img_nc, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.act_func, self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        
        return mySequential(*layers)  # nn.Sequential(*layers)

    def forward(self, x):
        
        if len(x.shape) < 4:
            x = x.unsqueeze(1)   # add channel index, now x is in shape [B * N, channels, img_sz, img_sz]
                    
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.adaptive_avg_pool(out)
        # out = F.avg_pool2d(out, 4)
        out_emb = out.view(out.size(0), -1)
        out = self.linear(out_emb)
        return out
    
    

def resnet18(params):
    return ResNet(params, BasicBlock, [2, 2, 2, 2], num_classes=params['h_dim'])


def resnet34(params):
    return ResNet(params, BasicBlock, [3, 4, 6, 3], num_classes=params['h_dim'])


class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs