import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm, trange

import numpy as np
from torchvision import models
from scipy import linalg
import torch.nn.functional as F
from torch.nn import Parameter as P
from torchvision.models.inception import inception_v3


#=========================================================================================
# FID from starganv2
class InceptionV3(nn.Module):
    def __init__(self):
        super().__init__()
        inception = models.inception_v3(pretrained=True)
        self.mean = P(torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1),
                  requires_grad=False)
        self.std = P(torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1),
                 requires_grad=False)
        self.block1 = nn.Sequential(
            inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.block2 = nn.Sequential(
            inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.block3 = nn.Sequential(
            inception.Mixed_5b, inception.Mixed_5c,
            inception.Mixed_5d, inception.Mixed_6a,
            inception.Mixed_6b, inception.Mixed_6c,
            inception.Mixed_6d, inception.Mixed_6e)
        self.block4 = nn.Sequential(
            inception.Mixed_7a, inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1)))

    def forward(self, x):
        x = (x + 1.) / 2.0
        x = (x - self.mean) / self.std
        # Upsample if necessary
        if x.shape[2] != 299 or x.shape[3] != 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=True)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x.view(x.size(0), -1)
#-----------------------------------------------------------------------
# from BigGAN-pytorch
# A pytorch implementation of cov, from Modar M. Alfadly
# https://discuss.pytorch.org/t/covariance-and-gradient-support/16217/2
def torch_cov(m, rowvar=False):
    '''Estimate a covariance matrix given data.
    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()

def frechet_distance(mu, cov, mu2, cov2):
    cc, _ = linalg.sqrtm(np.dot(cov, cov2), disp=False)
    dist = np.sum((mu -mu2)**2) + np.trace(cov + cov2 - 2*cc)
    return np.real(dist)
#-----------------------------------------------------------------------

@torch.no_grad()
def calculate_inception_moments_given_loader(loader, device, stain, mode='train', image_size=512, num_inception_images=50000, **kwargs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inception = InceptionV3().eval().to(device)
    mu, cov = [], []
    pools = []
    imgs_loaded = 0
    for batch in tqdm(loader, total=len(loader)):
        x, _ = batch
        pool = inception(x.to(device))
        pools.append(pool)
        imgs_loaded += x.size(0)
        if imgs_loaded > num_inception_images:
            break
    pools = torch.cat(pools, dim=0).cpu().detach()
    mu = torch.mean(pools, dim=0).detach().cpu().numpy()
    cov = torch_cov(pools, rowvar=False).detach().cpu().numpy()
    print('Saving ...', 'PPHM_{}_{}_{}_inception_moments.npz'.format(mode, stain, image_size))
    np.savez('PPHM_{}_{}_{}_inception_moments.npz'.format(mode, stain, image_size), **{'mu' : mu, 'sigma' : cov})
    
@torch.no_grad()
def calculate_fid_given_loader(mu_data, cov_data, loader, device, num_inception_images=50000, **kwargs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inception = InceptionV3().eval().to(device)
    mu, cov = [], []
    pools = []
    imgs_loaded = 0
    for x in tqdm(loader, total=num_inception_images):
        pool = inception(x.to(device))
        pools.append(pool)
        imgs_loaded += x.size(0)
        if imgs_loaded > num_inception_images:
            break
    pools = torch.cat(pools, dim=0).cpu().detach()
    mu = torch.mean(pools, dim=0)
    cov = torch_cov(pools, rowvar=False)
    fid = frechet_distance(mu.numpy(), cov.numpy(), mu_data, cov_data)
    return fid
    
#-----------------------------------------------------------------------
@torch.no_grad()
def PSNR(img_1, img_2, data_range=1.0):
    """batch capable PSNR"""
    mse = nn.functional.mse_loss(denorm(img_1), denorm(img_2), reduction='none').mean(dim=[1, 2, 3])
    return 10.0 *torch.log10(data_range / mse)