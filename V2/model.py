# %%
import numpy as np
import torch 
from torch import nn

# %%

def Adaptive_Instance_Norm(content, style):
    assert content.shape[:2] == style.shape[:2], "Not equal shape of content feature and style feature"
    
    content_mean = content.mean((2, 3), keepdims=True)
    content_std = content.std((2, 3), keepdims=True)
    
    style_mean = style.mean((2, 3), keepdims=True)
    style_std = style.std((2, 3), keepdims=True)

    output = style_std * (content - content_mean) / content_std + style_mean

    return output

class Residual_block(nn.Module):
    def __init__(self, in_channel, output_channel, ksize=3, stride=1, padding=1, adain=True, use_branch=True):
        super(Residual_block, self).__init__()

        self.branch1 = lambda x: x
        if use_branch:
            self.branch1 = nn.Conv2d(in_channel, output_channel, 1, stride)
        
        self.branch2 = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channel, output_channel, ksize, stride, padding)     
        )

        self.branch3 = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channel, output_channel, ksize, stride, padding)     
        )

        if adain:
            self.norm1 = Adaptive_Instance_Norm
            self.norm2 = Adaptive_Instance_Norm
        else:
            self.norm1 = nn.InstanceNorm2d(output_channel, affine=True, track_running_stats=True)
            self.norm2 = nn.InstanceNorm2d(output_channel, affine=True, track_running_stats=True)

        self.relu = nn.ReLU(True)

    def forward(self, x, style):
        # To-do : Instance & Adaptive Instance
        out = self.norm1(x)
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
        layers.append(Residual_block(features, features)) 
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
    def __init__(self, in_channel, n_domains=3):
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

class Style_Encoder_N_Discriminator(nn.Module):
    def __init__(self, in_channel, n_domains=3, features=16, D = 64):
        super(Style_Encoder, self).__init__()

        layers = [
            nn.Conv2d(in_channel, features, 1, 1),
            nn.ReLU(True)
        ]

        for _ in range(5):
            layers.append(Residual_block(features, features*2))
            layers.append(nn.AvgPool2d(2, 2))
            features *= 2
        layers.append(Residual_block(features, features*2))
        layers.append(nn.AvgPool2d(2, 2))
        layers.append(nn.LeakyReLU(0.01, True))
        layers.append(nn.Conv2d(features, features, 4, 1))
        layers.append(nn.LeakyReLU(0.01, True))
        layers.append(nn.Flatten())
        layers.append(nn.Dense(features, n_domains*D))

        self.Net = nn.Sequential(*layers)
        
        def forward(self, x):
            return self.Net(x)

# %%
