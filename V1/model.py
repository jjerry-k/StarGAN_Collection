# %%
import numpy as np
import torch 
from torch import nn

# %%

class Residual_block(nn.Module):
    def __init__(self, in_channel, output_channel, strides=1, use_branch=True):
        super(Residual_block, self).__init__()

        self.branch1 = lambda x: x
        if use_branch:
            self.branch1 = nn.Conv2d(in_channel, output_channel, 1, strides)
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channel, output_channel//4, 1, strides),
            nn.InstanceNorm2d(output_channel//4, affine=True, track_running_stats=True),
            nn.ReLU(True),
            nn.Conv2d(output_channel//4, output_channel//4, 3, 1, padding=1),
            nn.InstanceNorm2d(output_channel//4, affine=True, track_running_stats=True),
            nn.ReLU(True),
            nn.Conv2d(output_channel//4, output_channel, 1, 1),
            nn.InstanceNorm2d(output_channel, affine=True, track_running_stats=True),        
        )

        self.relu = nn.ReLU(True)

    def forward(self, x):
        out = self.branch2(x)
        out = self.relu(out + self.branch1(x))

        return out

class Generator(nn.Module):
    def __init__(self, in_channel, n_domains):
        super(Generator, self).__init__()

        layers = [
            nn.Conv2d(in_channel, 32, 1, 1),
            nn.ReLU(True)
        ]
        
        features = 32

        for _ in range(4):
            layers.append(Residual_block(features, features*2))
            layers.append(nn.AvgPool2d(2, 2))
            features *= 2

        layers.append(Residual_block(features, features))
        layers.append(Residual_block(features, features))
        layers.append(Residual_block(features, features)) # To-do : Instance --> Adaptive Instance
        layers.append(Residual_block(features, features))

        for i in range(4):
            layers.append(Residual_block(features, features//2))
            layers.append(nn.Upsample(scale_factor=(2, 2), mode='nearest'))
            features //= 2
            
        layers.append(nn.Conv2d(feature, in_channel, 1, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
        return out
    
class Mapping_Network(nn.Module):
    def __init__(self, in_channel, n_domains):
        super(Mapping_Network, self).__init__()

        layers = [
            nn.Linear(in_channel, 512), 
            nn.ReLU(True)]

        for _ in range(1, 6):
            layers.append(nn.Linear(in_channel, 512))
            layers.append(nn.ReLU(True))
        layers.append(nn.Linear(512, 64*n_domains))

        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.net(x)
        return out

class Style_Encoder(nn.Module):
    def __init__(self, in_channel, n_domains, features):
        super(Style_Encoder, self).__init__()

        layers = [
            nn.Conv2d(in_channel, features, 1, 1),
            nn.ReLU(True)
        ]

        for _ in range(6):
            layers.append(Residual_block(features, features*2))
            layers.append(nn.AvgPool2d(2, 2))
            features *= 2