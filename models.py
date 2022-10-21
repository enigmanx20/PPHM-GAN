import torch
import torch.nn as nn
import numpy as np
from torch.cuda.amp import autocast
from nnModules import conv2d, upscale2d_conv2d, conv2d_downscale2d, apply_bias

class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in: int, dim_out: int, act: str, inplace_act: bool=True, resblock_bias: bool=False, **kwargs):
        super(ResidualBlock, self).__init__()
        self.act, self.gain = {'relu': (nn.ReLU(inplace=inplace_act), np.sqrt(2)), 
                               'lrelu': (nn.LeakyReLU(inplace=inplace_act), np.sqrt(2))}[act]
        layers = []
        layers.append(conv2d(dim_in, dim_out, kernel=3, gain=self.gain, use_wscale=True, **kwargs) )
        if resblock_bias:
            layers.append(apply_bias(dim_out))
        layers.append(nn.InstanceNorm2d(dim_out, affine=False, track_running_stats=False)) # default pytorch setting: affine=False
        layers.append(self.act)
        layers.append(conv2d(dim_in, dim_out, kernel=3, gain=self.gain, use_wscale=True, **kwargs) )
        if resblock_bias:
            layers.append(apply_bias(dim_out))
        layers.append(nn.InstanceNorm2d(dim_out, affine=False, track_running_stats=False)) 
        
        self.main = nn.Sequential(*layers)
        
    def forward(self, x, **kwargs):
        return x + self.main(x)
    

class Generator(nn.Module):
    """
    conv_dim: base conv channels, doubled as the downsampling.
    c_dim:    number of transformation dimension. 
    downsample: number of downsampling. 
    repeat_num: number of residual blocks.
    inplace_act: inplace flag for activation functions.
    enable_autocast: flag to switch autocast.
    """
    def __init__(self, conv_dim: int=64, c_dim: int=4, downsample: int=2, repeat_num: int=6, inplace_act: bool=True, 
                 enable_autocast: bool=False, **kwargs):
        
        super(Generator, self).__init__()
        self.c_dim = c_dim
        self.enable_autocast = enable_autocast
        layers = []
        layers.append(conv2d(3+c_dim, conv_dim, kernel=3, stride=1, **kwargs))
        layers.append(conv2d(conv_dim, conv_dim, kernel=3, stride=1, **kwargs))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=False, track_running_stats=False))
        layers.append(nn.ReLU(inplace=inplace_act))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(downsample):
            layers.append(conv2d_downscale2d(curr_dim, curr_dim*2, kernel=5,  **kwargs))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=False, track_running_stats=False))
            layers.append(nn.ReLU(inplace=inplace_act))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim, act='relu', inplace_act = inplace_act))

        # Up-sampling layers.
        for i in range(downsample):
            layers.append(upscale2d_conv2d(curr_dim, curr_dim//2, kernel=5, **kwargs))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=False, track_running_stats=False))
            layers.append(nn.ReLU(inplace=inplace_act))
            curr_dim = curr_dim // 2

        layers.append(conv2d(curr_dim, 3, kernel=7, stride=1, **kwargs))
        layers.append(nn.Tanh())
        self.layers = nn.ModuleList(layers)

    def forward(self, x, c_to, **kwargs):
        # Replicate spatially and concatenate domain information.
        c_to = c_to.view(c_to.size(0), self.c_dim, 1, 1)
        c_to = c_to.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c_to], dim=1) #channel concatenation
        with autocast(self.enable_autocast): 
            for layer in self.layers:
                x = layer(x)
        return x

class Discriminator(nn.Module):
    """
    image_size: size of the input images.
    conv_dim: base conv channels, doubled as the downsampling.
    c_dim:    number of transformation dimension.
    downsample_d: number of downsampling.
    patchGAN:     flag to enable patchGAN. If False, the last conv2d has the same kernel size as the last feature maps.
    enable_autocast: flag to switch autocast.
    """
    def __init__(self, image_size: int=128, conv_dim: int=64, c_dim: int=3, downsample_d: int=6, patchGAN: bool=True, 
                 enable_autocast: bool=False, **kwargs):
        #downsampled by 2**(repeat_num +1)
        self.enable_autocast = enable_autocast
        self.patchGAN = patchGAN
        super(Discriminator, self).__init__()
        layers = []
        layers.append(conv2d_downscale2d(3, conv_dim, kernel=5,  **kwargs))
        layers.append(apply_bias(conv_dim, **kwargs))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, downsample_d):  
            layers.append(conv2d_downscale2d(curr_dim, curr_dim*2, kernel=5, **kwargs))
            layers.append(apply_bias(curr_dim*2, **kwargs))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2
        kernel_size = int(image_size / np.power(2, downsample_d))
        self.main = nn.Sequential(*layers)
        if self.patchGAN:
            self.conv1 = conv2d(curr_dim, 1, kernel=3, **kwargs)
        else:
            self.conv1 = conv2d(curr_dim, 1, kernel=kernel_size, **kwargs)
        self.conv2 = conv2d(curr_dim, c_dim, kernel=kernel_size, **kwargs)
            
        
    def forward(self, x, **kwargs):
        with autocast(self.enable_autocast): 
            h = self.main(x)
            out_src = self.conv1(h).sum(dim=[2, 3])  # fake or genuine
            out_cls = self.conv2(h).sum(dim=[2, 3])    # classes such PAS, PAM, H&E, or MT
        return torch.cat([out_src, out_cls], dim=1)