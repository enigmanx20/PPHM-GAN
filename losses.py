import torch
import torch.nn.functional as F
import utils
from logger import *
from torch.cuda.amp import autocast

#--from styleGAN ---------------------------------------------
def gradient_penalty(D, x_hat, target =1.0, l2=True):
    x_hat.requires_grad = True
    y = D(x_hat)[:, 0].mean()
    dydx = torch.autograd.grad(outputs=y,
                               inputs=x_hat,
                               create_graph=True,
                              )[0]
    dydx = dydx.view(dydx.size(0), -1)
    if l2:
        dydx = torch.sqrt(torch.sum(dydx**2, dim=[i for i in range(1, len(dydx.size()))]))
    gp = (dydx-target).pow(2).sum(1)
    return gp #[minibatch, 1]

#---------from BigGAN---------------
# DCGAN loss
def loss_dcgan_dis(dis_fake, dis_real, logger=None, itr=0, **kwargs):
    L1 = torch.mean(utils.softplus(-dis_real))
    L2 = torch.mean(utils.softplus(dis_fake))
    if logger is not None:
        item = {}
        item['D/loss_dc_gan_L1'] = L1.item()
        item['D/loss_dc_gan_L2'] = L2.item()
        for tag, value in item.items():
            logger.scalar_summary(tag, value, itr+1)
    return L1 + L2

def loss_dcgan_gen(dis_fake, logger=None, itr=0, **kwargs):
    loss = torch.mean(utils.softplus(-dis_fake))
    if logger is not None:
        item = {}
        item['G/loss_dc_gan'] = loss.item()
        for tag, value in item.items():
            logger.scalar_summary(tag, value, itr+1)
    return loss

#---------from BigGAN---------------
# Hinge Loss
def loss_hinge_dis(dis_fake, dis_real, logger=None, itr=0, **kwargs):
    loss_real = torch.mean(F.relu(1. - dis_real))
    loss_fake = torch.mean(F.relu(1. + dis_fake))
    if logger is not None:
        item = {}
        item['D/loss_hinge_real'] = loss_real.item()
        item['D/loss_hinge_fake'] = loss_real.item()
        for tag, value in item.items():
            logger.scalar_summary(tag, value, itr+1)
    return loss_real + loss_fake

def loss_hinge_gen(dis_fake, logger=None, itr=0, **kwargs):
    loss = -torch.mean(dis_fake)
    if logger is not None:
        item = {}
        item['G/loss_hinge'] = loss.item()
        for tag, value in item.items():
            logger.scalar_summary(tag, value, itr+1)
    return loss

#--------------------from stylegan-------------------------------------------
# WGAN & WGAN-GP loss functions.
def loss_wgan_dis(dis_fake, dis_real, wgan_epsilon = 0.001,
                   logger=None, itr=0, **kwargs):
    loss = dis_fake - dis_real
    epsilon_penalty = dis_real ** 2
    loss += epsilon_penalty * wgan_epsilon
    loss = loss.mean()
    if logger is not None:
        item = {}
        item['D/loss_wgan'] = loss.mean().item()
        item['D/epsilon_penalty'] = epsilon_penalty.mean().item()
        for tag, value in item.items():
            logger.scalar_summary(tag, value, itr+1)
    return loss

def loss_wgan_gen(dis_fake, logger=None, itr=0, **kwargs):
    loss = -torch.mean(dis_fake)
    if logger is not None:
        item = {}
        item['G/loss_wgan'] = loss.item()
        for tag, value in item.items():
            logger.scalar_summary(tag, value, itr+1)
    return loss

#------------from stylegan----------------------------------------
# Loss functions advocated by the paper  "Which Training Methods for GANs do actually Converge?"
def loss_logistic_saturating_gen(dis_fake, logger=None, itr=0, **kwargs): 
    loss = -utils.softplus(dis_fake, beta=1.0)  
    if logger is not None:
        item = {}
        item['G/loss_logistic_saturating'] = loss.mean().item()
        for tag, value in item.items():
            logger.scalar_summary(tag, value, itr+1)
    return torch.mean( loss )

def loss_logistic_nonsaturating_gen(dis_fake, logger=None, itr=0, **kwargs):
    loss = utils.softplus(-dis_fake, beta=1.0) 
    if logger is not None:
        item = {}
        item['G/loss_logistic_nonsaturating'] = loss.mean().item()
        for tag, value in item.items():
            logger.scalar_summary(tag, value, itr+1)
    return torch.mean( loss )

def loss_logistic_dis(dis_fake, dis_real,logger=None, itr=0, **kwargs):
    loss = utils.softplus(dis_fake, beta=1.0) + utils.softplus(-dis_real, beta=1.0) 
    if logger is not None:
        item = {}
        item['D/loss_logistic'] = loss.mean().item()
        for tag, value in item.items():
            logger.scalar_summary(tag, value, itr+1)
    return torch.mean( loss ) # [minibatch, loss]

def loss_logistic_simplegp_dis(dis_fake, dis_real, D, fake, real,
                               logger=None, itr=0, 
                                     r1_gamma=1.0, 
                                     r2_gamma=0.0, **kwargs): 
    
    loss = utils.softplus(dis_fake, beta=1.0) + utils.softplus(-dis_real, beta=1.0) 
    if logger is not None:
        item = {}
        item['D/loss_logistic'] = loss.mean().item()
    if r1_gamma != 0.0:
        R1 = gradient_penalty(D, real, target=0.0, l2=False)
        loss += R1 * (r1_gamma * 0.5)
    if r2_gamma != 0.0:
        R2 = gradient_penalty(D, fake, target=0.0)
        loss += R2 * (r2_gamma * 0.5)
    if logger is not None:
        if r1_gamma != 0.0:
            item['D/simpleGP_R1'] = R1.mean().item()
        if r2_gamma != 0.0:
            item['D/simpleGP_R2'] = R2.mean().item()
        item['D/loss_logistic_simplegp'] = loss.mean().item()
        for tag, value in item.items():
            logger.scalar_summary(tag, value, itr+1)  
    
    return torch.mean( loss)  

#=======================================================================================
def loss_vanilla_gen(dis_fake, logger=None, itr=0, **kwargs):
    loss = F.binary_cross_entropy_with_logits(dis_fake, torch.full_like(dis_fake, fill_value=1))
    if logger is not None:
        item = {}
        item['G/loss_logistic_saturating'] = loss.item()
        for tag, value in item.items():
            logger.scalar_summary(tag, value, itr+1)
    return loss 

def loss_vanilla_dis(dis_fake, dis_real,logger=None, itr=0, **kwargs):
    loss = F.binary_cross_entropy_with_logits(dis_fake, torch.full_like(dis_fake, fill_value=0)) + F.binary_cross_entropy_with_logits(dis_real, torch.full_like(dis_real, fill_value=1))
    if logger is not None:
        item = {}
        item['D/loss_logistic'] = loss.item()
        for tag, value in item.items():
            logger.scalar_summary(tag, value, itr+1)
    return loss # [minibatch, loss]

def loss_vanilla_simplegp_dis(dis_fake, dis_real, D, fake, real,
                               logger=None, itr=0, 
                                     r1_gamma=1.0, 
                                     r2_gamma=0.0, **kwargs):
    
    loss = F.binary_cross_entropy_with_logits(dis_fake, torch.full_like(dis_fake, fill_value=0)) + F.binary_cross_entropy_with_logits(dis_real, torch.full_like(dis_real, fill_value=1))
    
    if logger is not None:
        item = {}
        item['D/loss_logistic'] = loss.item()
    
    if r1_gamma != 0.0:
        R1 = gradient_penalty(D, real, target=0.0, l2=False).mean()
        loss += R1 * (r1_gamma * 0.5)
    if r2_gamma != 0.0:
        R2 = gradient_penalty(D, fake, target=0.0).mean()
        loss += R2 * (r2_gamma * 0.5)
    if logger is not None:
        if r1_gamma != 0.0:
            item['D/simpleGP_R1'] = R1.mean().item()
        if r2_gamma != 0.0:
            item['D/simpleGP_R2'] = R2.mean().item()
        item['D/loss_logistic_simplegp'] = loss.mean().item()
        for tag, value in item.items():
            logger.scalar_summary(tag, value, itr+1)  
    return loss
