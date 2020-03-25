# %%
import numpy as np
import torch 
from torch import nn

# %%

class Conv_Block(nn.Module):
    def __init__(self, in_channel, output_channel, ksize, stride, padding, upsample=False):
        super(Conv_Block, self).__init__()

        if upsample :
            layers = [nn.ConvTranspose2d(in_channel, output_channel, ksize, stride, padding)]
        else :
            layers = [nn.Conv2d(in_channel, output_channel, ksize, stride, padding)]
        
        layers.append(nn.InstanceNorm2d(output_channel, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(True))
        
        self.Block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.Block(x)

class Residual_block(nn.Module):
    def __init__(self, in_channel, output_channel):
        super(Residual_block, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channel, output_channel, 1, 1),
            nn.InstanceNorm2d(output_channel, affine=True, track_running_stats=True),
            nn.ReLU(True),
            nn.Conv2d(output_channel, output_channel, 3, 1, 1),
            nn.InstanceNorm2d(output_channel, affine=True, track_running_stats=True)       
        )

    def forward(self, x):
        out = x + self.branch1(block)
        return out

class Generator(nn.Module):
    def __init__(self, in_channel=3, n_domains=3, repeat_num=6):
        super(Generator, self).__init__()
        layers = [
            Conv_Block(in_channel, 64, 7, 1, 3),
            Conv_Block(64, 128, 4, 2, 1),
            Conv_Block(128, 256, 4, 2, 1)
        ]

        for _ in range(repeat_num):
            layers.append(Residual_block(256, 256))
                        
        layers.append(Conv_Block(256, 128, 4, 2, 1, True))
        layers.append(Conv_Block(128, 64, 4, 2, 1, True))
        
        layers.append(nn.Conv2d(64, in_channel, 7, 1, 3))
        layers.append(nn.Tanh())
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
        return out

class Discriminator(nn.Module):
    def __init__(self, img_size=256, in_channel=3, n_domains=3, repeat_num=6):
        super(Discriminator, self).__init__()

        layers = []

        curr_f, next_f = in_channel, 64
        for _ in range(repeat_num):
            layers.append(nn.Conv2d(curr_f, next_f, 4, 2, 1))
            layers.append(nn.LeakyReLU(0.01, True))
            curr_f, next_f = next_f, next_f*2
        
        k_size = int(img_size / (2**repeat_num))
        self.Stem = nn.Sequential(*layers)
        self.Src = nn.Conv2d(2048, 1, 3, 1, 1)
        self.Cls = nn.Conv2d(2048, n_domains, k_size)

    def forward(self, x):
        out = self.Stem(x)
        out_src = self.Src(out)
        out_cls = self.Cls(out)
        return out_src, out_cls.reshape(out_cls.shape[0], out_cls.shape[1])