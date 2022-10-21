# EasyDict, print_network, build_tensorboard, lerp, softplus, set_moving_average, Adam16
#import tensorflow as tf
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

def strip_sd(state_dict):
    """remove module. from module names"""
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]  # remove 'module.' of dataparallel
        new_state_dict[name] = v
    return new_state_dict

class AbsLoss(object):
    def __init__(self):
        super().__init__()
    def __call__(self, target, fake):
        return torch.abs(target - fake).mean()

#from Big-GAN
# Utility file to seed rngs
def seed_rng(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  np.random.seed(seed)

# from BigGAN-pytorch
# This is partially based on https://discuss.pytorch.org/t/subclassing-torch-tensor/23754/2
class Distribution(torch.Tensor):
  # Init the params of the distribution
  def init_distribution(self, dist_type='normal', **kwargs):    
    self.dist_type = dist_type
    self.dist_kwargs = kwargs
    if self.dist_type == 'normal':
      self.mean, self.var = kwargs['mean'], kwargs['var']
    elif self.dist_type == 'categorical':
      self.num_categories = kwargs['num_categories']

  def sample_(self):
    if self.dist_type == 'normal':
      self.normal_(self.mean, self.var)
    elif self.dist_type == 'categorical':
      self.random_(0, self.num_categories)    

        
def lerp(a: torch.Tensor, b: torch.Tensor, t) -> torch.Tensor:
    """Linear interpolation."""
    assert a.device == b.device
    if isinstance(t, torch.Tensor):
        t = t.to(a.device)
        if a.dim()==1:
            t = t.squeeze()
    
    return a + (b - a) * t

def cos_similarity(x, y, eps=1e-6):
    x = x.view([-1])
    y = y.view([-1])
    return torch.dot(x, y)  *  torch.rsqrt(torch.sum(x**2) + eps ) * torch.rsqrt(torch.sum(y**2) + eps )

def softplus(x:torch.Tensor, beta=1.0):
    return torch.log1p((torch.exp(beta * x)))/ beta
    
def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1.) / 2.
    return out.clamp_(0, 1)
    
    
def print_network(model):
    """Print out the network information."""
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(model)
    print("The number of parameters: {}".format(num_params))
        
# Convenience utility to switch off requires_grad
def toggle_grad(model, on_or_off: bool):
    for param in model.parameters():
        param.requires_grad = on_or_off

# form BigGAN-pytorch
# Apply modified ortho reg to a model
def ortho(model, strength=1e-4, skip_module=[]):
  with torch.no_grad():
    for name, param in model.named_parameters():
      # Only apply this to parameters with at least 2 axes, and not in the blacklist
      if len(param.shape) < 2 or sum([m in name for m in skip_module]):
        continue
      w = param.reshape(param.shape[0], -1)
      grad = (2 * torch.mm(torch.mm(w, w.t()) 
              * (1. - torch.eye(w.shape[0], device=w.device)), w))
      param.grad.data += strength * grad.view(param.shape)
        
def ema(target_net:nn.Module, src_net: nn.Module, is_parallel=False, beta = 0.99, beta_nontrainable = 1.0) -> None:
    """updates the variables of this network
        to be slightly closer to those of the given network."""
    # target_net @cpu, src_net@any and maybe datapallael compatible, copy as torch.float32
    src_names         = []
    target_param_dict = {}

    for name, param in src_net.named_parameters():
        if is_parallel:
            name = name.replace('module.', '')
        src_names.append(name)
        target_param_dict[name]=param.requires_grad # name: True or False
    sd = target_net.state_dict()

    for name, param in target_net.named_parameters():
        with torch.no_grad():
            if name in src_names:   
                cur_beta = torch.Tensor([beta]) if target_param_dict[name] else torch.Tensor([beta_nontrainable])
                if is_parallel:
                    src_fullname = 'module.' + name
                    new_value = lerp(src_net.state_dict()[src_fullname].cpu().to(torch.float32), 
                                       target_net.state_dict()[name].cpu(), cur_beta)
                else:
                    src_fullname =  name
                    new_value = lerp(src_net.state_dict()[src_fullname].cpu().to(torch.float32), 
                                       target_net.state_dict()[name].cpu(), cur_beta)
                sd[name]   = new_value
            
    target_net.load_state_dict(sd)

def  lerp_buffer(target_net:nn.Module, src_net: nn.Module, is_parallel=False, beta = 1.0) -> None:
    """linearly interpolate buffers of two modules."""
    src_names         = []
    target_param_dict = {}

    for name, param in src_net.named_buffers():
        if is_parallel:
            name = name.replace('module.', '')
        src_names.append(name)
    sd = target_net.state_dict()

    for name, param in target_net.named_buffers():
        with torch.no_grad():
            if name in src_names:
                cur_beta = torch.Tensor([beta])
                if is_parallel:
                    src_fullname = 'module.' + name
                    new_value = lerp(src_net.state_dict()[src_fullname].cpu().to(torch.float32), 
                                       target_net.state_dict()[name].cpu(), cur_beta)
                else:
                    src_fullname =  name
                    new_value = lerp(src_net.state_dict()[src_fullname].cpu().to(torch.float32), 
                                       target_net.state_dict()[name].cpu(), cur_beta)
                sd[name]   = new_value
    target_net.load_state_dict(sd)
    
#---------------------------------------------------------------------------------
def log_weight(net, skip_module=[], logger=None, itr=0, **kwargs):
    sd = net.state_dict()
    skip_module = skip_module
    if logger is None:
        return None
    for name, param in net.named_parameters():
        if not sum([m in name for m in skip_module]):
            if 'w' or 'weight' in name:
                logger.histogram_summary('weights/%s'%name, sd[name] , itr+1)
            
def log_sv(net, skip_module=[], logger=None, itr=0, **kwargs):
    sd = net.state_dict()
    skip_module = skip_module
    #SVs = []
    if logger is None:
        return None                
    for name, buff in net.named_buffers():
        if 'sv' in name:
            logger.scalar_summary('sv/%s'%name, buff[0].item() , itr+1)
            
 #---------------------------------------------------------------------------------       
 # Sample function for use with inception metrics
def sample(G, dataloader, y_, cls_onehot, stats):
    with torch.no_grad():
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        reals, cls =next(iter(dataloader))
        blured_reals = blur_lod(reals, stats['lod_in'])
        y_.sample_()
        cls_onehot.zero_()
        cls_onehot.scatter_(1, y_.view([-1, 1]).to(device, torch.long), 1)
        G_z =  G(reals, cls_onehot)
        return G_z.to(device), y_.cuda(device)       

#----------------------------------------------------------------------------------