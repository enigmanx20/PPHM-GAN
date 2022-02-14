import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.nn import ReLU, LeakyReLU
import torch.nn.functional as F
import numpy as np

#===========================================================================================================================
# from BigGAN-PyTorch: https://github.com/ajbrock/BigGAN-PyTorch
# Projection of x onto y
def proj(x, y):
  return torch.mm(y, x.t()) * y / torch.mm(y, y.t())

# Orthogonalize x wrt list of vectors ys
def gram_schmidt(x, ys):
  for y in ys:
    x = x - proj(x, y)
  return x

# Apply num_itrs steps of the power method to estimate top N singular values.
def power_iteration(W, u_, update=True, eps=1e-12):
    # Lists holding singular vectors and values
    dtype = W.dtype
    W = W.to(torch.float32)
    us, vs, svs = [], [], []

    for i, u in enumerate(u_):
    # Run one step of the power iteration
        with torch.no_grad():
            u = u.to(torch.float32)
            v = torch.matmul(u, W)
            # Run Gram-Schmidt to subtract components of all other singular vectors
            v = F.normalize(gram_schmidt(v, vs), eps=eps)
            # Add to the list
            vs += [v]
            # Update the other singular vector
            u = torch.matmul(v, W.t())
            # Run Gram-Schmidt to subtract components of all other singular vectors
            u = F.normalize(gram_schmidt(u, us), eps=eps)
            # Add to the list
            us += [u]
            if update:
                u_[i][:] = u.to(dtype)
    # Compute this singular value and add it to the list
    svs += [torch.squeeze(torch.matmul(torch.matmul(v, W.t()), u.t())).to(dtype)]
    #svs += [torch.sum(F.linear(u, W.transpose(0, 1)) * v)]
    return svs, us, vs

# spectral normalization base class
class SN(nn.Module):
  def __init__(self, num_svs, num_itrs, num_outputs, transpose=False, eps=1e-12):
    # Number of power iterations per step
    super().__init__()
    self.num_itrs = num_itrs
    # Number of singular values
    self.num_svs = num_svs
    #
    self.num_outputs = num_outputs
    # Transposed?
    self.transpose = transpose
    # Epsilon value for avoiding divide-by-0
    self.eps = eps
    # Register a singular vector for each sv
    for i in range(self.num_svs):
      self.register_buffer('u%d' % i, torch.randn(1, num_outputs))
      self.register_buffer('sv%d' % i, torch.ones(1))
  
  # Singular vectors (u side)
  @property
  def u(self):
    return [getattr(self, 'u%d' % i) for i in range(self.num_svs)]

  # Singular values; 
  # note that these buffers are just for logging and are not used in training. 
  @property
  def sv(self):
   return [getattr(self, 'sv%d' % i) for i in range(self.num_svs)]
   
  # Compute the spectrally-normalized weight
  def W_(self):    
    W_mat = self.w.reshape(self.num_outputs, -1)
    
    if self.transpose:
      W_mat = W_mat.t()
    # Apply num_itrs power iterations
    for _ in range(self.num_itrs):
      svs, us, vs = power_iteration(W_mat, self.u, update=self.training, eps=self.eps)  #in fp32
    # Update the svs
    if self.training:
      with torch.no_grad(): # Make sure to do this in a no_grad() context or you'll get memory leaks!
        for i, sv in enumerate(svs):
          self.sv[i][:] = sv     
    return self.w / svs[0].item()   



# -------- modules for progressive growing --------------------------------------------------------------------------------
# from StyleGAN: https://github.com/NVlabs/stylegan translated to pytorch
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved. modified by Masataka Kawai
# original liscence is CC BY-NC
class _blur2d(object):
    def __init__(self,  f=[1,2,1], normalize=True, flip=False, stride=1):
        assert isinstance(stride, int) and stride >= 1
        self.stride = stride
        f = np.array(f, dtype=np.float32)
        self.padding =  (len(f) - 1) //2
        if f.ndim == 1:
            f = f[:, np.newaxis] * f[np.newaxis, :]
        assert f.ndim == 2
        if normalize:
            f /= np.sum(f)
        if flip:
            f = f[::-1, ::-1]
        f = f[np.newaxis, np.newaxis, :, : ]
        
        self.f = f
    
    def __call__(self, x: torch.Tensor):
        assert len(x.size()) == 4 and all(dim is not None for dim in x.shape[1:])
        if tuple(self.f.shape) == (1, 1) and f[0,0] == 1:
            return x
        device = x.device
        f = torch.from_numpy(np.tile(self.f, [int(x.shape[1]), 1, 1, 1] )) # e.g. [5, 1, 3, 3] for 5 channels
        f = f.to(x.device)

        x = F.conv2d(x, f, stride=(self.stride, self.stride),  groups = int(x.shape[1]), padding=self.padding)
        return x
 
class blur2d(nn.Module):
    def __init__(self,  f=[1,2,1], normalize=True, flip=False, stride=1):
        super(blur2d, self).__init__()
        assert isinstance(stride, int) and stride >= 1
        self._blur2d = _blur2d(f, normalize, flip, stride)
        self._blur2d_b = _blur2d(f, normalize, flip=True, stride=stride)
    
    def forward(self, x: torch.Tensor):
        assert len(x.size()) == 4 and all(dim is not None for dim in x.shape[1:])
        return self._blur2d(x)
    
    def backward(self, grad_out):
        return self._blur2d_b(grad_out)
    
class _upscale2d(object):
    def __init__(self, factor=2, gain=1) -> torch.Tensor:
        super(_upscale2d, self).__init__()
        assert isinstance(factor, int) and factor >= 1
        self.factor = factor
        self.gain = gain
    
    def __call__(self, x: torch.Tensor):
        assert len(x.size()) == 4 and all(dim is not None for dim in x.shape[1:])
        if self.gain != 1:
            x *= self.gain
        if self.factor == 1:
            return x
        s = x.shape
        x = x.view([-1, s[1], s[2], 1, s[3], 1])
        x = x.repeat(1, 1, 1 , self.factor, 1, self.factor)
        x = x.view([-1, s[1], s[2] * self.factor, s[3] * self.factor])
        return x
    
class upscale2d(nn.Module):
    def __init__(self, factor=2, gain=1) -> torch.Tensor:
        super(upscale2d, self).__init__()
        assert isinstance(factor, int) and factor >= 1
        self._upscale2d = _upscale2d(factor=factor,  gain=gain)
        self._downscale2d = _downscale2d(factor, gain=float(factor)**2)        

    def forward(self, x: torch.Tensor):
        assert len(x.size()) == 4 and all(dim is not None for dim in x.shape[1:])

        return self._upscale2d(x)
    
    def backward(self, grad_out):
        return self._downscale2d(grad_out)

class _downscale2d(object):
    def __init__(self, factor=2, gain=1):
        super(_downscale2d, self).__init__()
        assert isinstance(factor, int) and factor >= 1
        self.factor = factor
        self.gain = gain
        if factor == 2:
            f = [np.sqrt(gain) / factor] * factor
            self.blur = _blur2d(f=f, normalize=False, stride=factor)
    def __call__(self, x:torch.Tensor):
        assert len(x.size()) == 4 and all(dim is not None for dim in x.shape[1:])

        # 2x2, float32 => downscale using _blur2d().
        if self.factor == 2 and x.dtype == torch.float32:
            return self.blur(x)

        if self.gain != 1:
            x *= self.gain
        if self.factor == 1:
            return x

        return F.avg_pool2d(x, kernel_size=self.factor, stride=None, padding=0)
    
class downscale2d(nn.Module):
    def __init__(self, factor=2, gain=1):
        super(downscale2d, self).__init__()
        assert isinstance(factor, int) and factor >= 1
        self._downscale2d = _downscale2d(factor=factor, gain=gain)
        self._upscale2d = _upscale2d(factor=factor, gain=1/float(factor)**2)
        
    def forward(self, x:torch.Tensor):
        assert len(x.size()) == 4 and all(dim is not None for dim in x.shape[1:])
        return self._downscale2d(x)
    
    def backward(self, grad_out):
        return self._upscale2d(grad_out)

# Get/create weight tensor for a convolutional or fully-connected layer. He initialization
def get_weight(shape: tuple, gain=np.sqrt(2), use_wscale=False, lrmul=1, **kwargs) ->torch.Tensor:
    fan_in = np.prod(shape[1:]) # [fmaps_out, fmaps_in, kernel, kernel] or [out, in]
    he_std = gain / np.sqrt(fan_in) # He init

    # Equalized learning rate and custom learning rate multiplier.
    if use_wscale:
        init_std = 1.0 / lrmul
        runtime_coef = he_std * lrmul
    else:
        init_std = he_std / lrmul
        runtime_coef = lrmul

    # Create variable.
    n = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([init_std]))
    init = n.sample(shape)
    init = (init * runtime_coef).view(shape)
    return init

# dense or linear layer with SN
class dense(SN):
    def __init__(self, shape_in, fmaps_out, SN=False, num_svs=1, num_itrs=1,eps=1e-12, **kwargs):
        self.SN = SN
        if SN:
            super(dense, self).__init__(num_svs=num_svs, num_itrs=num_itrs, num_outputs=fmaps_out, eps=eps) 
        else:
            nn.Module.__init__(self)
        if type(shape_in) in [int, np.int16, np.int32, np.int64]:
            fmaps_in = int( shape_in )
        elif len(shape_in) > 2:
            fmaps_in = np.prod([d for d in shape_in[:]]) 
        else:
            fmaps_in = shape_in[1]
        w = get_weight([fmaps_out, fmaps_in], **kwargs).permute([1, 0])
        self.w = torch.nn.Parameter(data=w.data, requires_grad = True)
    @property    
    def W(self):
        if self.SN:
            return self.W_()
        else:
            return self.w        
        
    def forward(self, x: torch.Tensor, **kwargs)->torch.Tensor:
        if len(x.shape) > 2:
            x = x.view([-1, np.prod([d for d in x.shape[1:]])])
        return torch.mm(x, self.W)
    
# Convolutional layer with SN

class conv2d(SN):
    def __init__(self, fmaps_in, fmaps_out, kernel, SN=False, num_svs=1, num_itrs=1,eps=1e-12, **kwargs):
        #assert kernel >= 1 and kernel % 2 == 1       
        self.SN = SN
        if SN:
            super(conv2d, self).__init__(num_svs=num_svs, num_itrs=num_itrs, num_outputs=fmaps_out, eps=eps) 
        else:
            nn.Module.__init__(self)
        self.kernel = kernel
        w = get_weight([fmaps_out, fmaps_in, kernel, kernel ], **kwargs)
        self.w = torch.nn.Parameter(data=w.data, requires_grad = True)
        self.padding = self.kernel//2
        
    @property  
    def W(self):
        if self.SN:
            return self.W_()
        else:
            return self.w      
        
    def forward(self, x,  **kwargs): 
        return F.conv2d(x, self.W, stride=1, padding=(self.padding, self.padding)) #same padding

# Fused convolution + scaling.
# Faster and uses less memory than performing the operations separately.

class upscale2d_conv2d(SN):
    def __init__(self, fmaps_in, fmaps_out, kernel,  fused_scale='auto',SN=False, num_svs=1, num_itrs=1, eps=1e-12,  **kwargs):
        assert kernel >= 1 and kernel % 2 == 1
        assert fused_scale in [True, False, 'auto']
        self.SN = SN
        if SN:
            super(upscale2d_conv2d, self).__init__(num_svs=num_svs, num_itrs=num_itrs, num_outputs=fmaps_out, eps=eps) 
        else:
            nn.Module.__init__(self) 
        self.fused_scale = fused_scale
        self.kwargs = kwargs
        self.kernel = kernel
        self.upscale2d = upscale2d()
        
        w = get_weight([fmaps_out, fmaps_in, kernel, kernel ], **kwargs).permute([1, 0, 2, 3]) # [fmaps_in, fmaps_out, kernel, kernel]
        
        self.w = torch.nn.Parameter(data=w.data, requires_grad = True)
    @property    
    def W(self):
        return self.W_() if self.SN else self.w  
             
    def forward(self, x:torch.Tensor) ->torch.Tensor:
        # Not fused => call the individual ops directly.
        self.fused_scale = min(x.shape[2:]) * 2 >= 128 if  self.fused_scale == 'auto' else self.fused_scale
        if not self.fused_scale:
            return F.conv2d(self.upscale2d(x), torch.transpose(self.W, 0, 1), stride=1, padding=(self.kernel//2, self.kernel//2))
        tmp = nn.ZeroPad2d((1, 1, 1, 1))(self.W)
        tmp= tmp[:, :, 1:, 1:] + tmp[:, :, :-1, 1:] +  tmp[:, :, 1:, :-1] + tmp[:, :, :-1, :-1]

        # Fused => perform both ops simultaneously using conv_transpose2d().
        return F.conv_transpose2d(x, tmp, bias=None, stride=2, padding= self.kernel//2, dilation=1)
    
class conv2d_downscale2d(SN):
    def __init__(self, fmaps_in, fmaps_out, kernel, fused_scale='auto',  
                       SN=False, num_svs=1, num_itrs=1,eps=1e-12, **kwargs):
        assert kernel >= 1 and kernel % 2 == 1
        assert fused_scale in [True, False, 'auto']
        self.SN = SN
        if SN:
            super(conv2d_downscale2d, self).__init__(num_svs=num_svs, num_itrs=num_itrs, num_outputs=fmaps_out, eps=eps) 
        else:
            nn.Module.__init__(self)
        self.fused_scale = fused_scale
        self.kwargs = kwargs
        self.kernel = kernel
        self.downscale2d = downscale2d()
        
        w = get_weight([fmaps_out, fmaps_in, kernel, kernel ], **kwargs) # [fmaps_out, fmaps_in, kernel, kernel]
        self.w = torch.nn.Parameter(data=w.data, requires_grad = True)
        
    @property    
    def W(self):
        return self.W_() if self.SN else self.w  
        
    def forward(self, x:torch.Tensor)->torch.Tensor:
            #rewritten for pytorch
        self.fused_scale = min(x.shape[2:])  >= 128 if  self.fused_scale == 'auto' else self.fused_scale
       
        # Not fused => call the individual ops directly.
        if not self.fused_scale:
            #return self.downscale2d(F.conv2d(x, torch.transpose(self.w, 0, 1), stride=1, padding=(self.kernel//2, self.kernel//2)))
            return self.downscale2d(F.conv2d(x, self.W, stride=1, padding=(self.kernel//2, self.kernel//2)))

        tmp = nn.ZeroPad2d((1, 1, 1, 1))(self.W)
        tmp= tmp[:, :, 1:, 1:] + tmp[:, :, :-1, 1:] +  tmp[:, :, 1:, :-1] + tmp[:, :, :-1, :-1] * 0.25
        # Fused => perform both ops simultaneously using tf.nn.conv2d().
        return F.conv2d(x, tmp, stride=2, padding=(self.kernel-1)//2 )

class apply_bias(nn.Module):
    def __init__(self, dims, **kwargs):
        super(apply_bias, self).__init__()
        bias = torch.zeros([dims])
        self.bias =  torch.nn.Parameter(data=bias, requires_grad = True)
        
    def forward(self, x: torch.Tensor, lrmul=1)->torch.Tensor:
        """bias is shared"""
        #b = torch.zeros(x.shape[1], dtype=x.dtype)
        b = self.bias * lrmul
        #b = torch.nn.Parameter(data=b, requires_grad = True)
        if len(x.shape) == 2:
            return x + b
        return x + b.view([1, -1, 1, 1])

#=====================================================================================================
# from StarGAN: https://github.com/yunjey/stargan
class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out, act, inplace_act=True, resblock_bias=False, **kwargs):
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


class G(nn.Module):
    """
    Stargan-based PPHM generator network.
    conv_dim: channel size of the first layer
    c_dim: class size (e.g. 4 for PPHM)
    downsample: # of downsampling, default 2
    repeat_num: # of ResidualBlock in downsampling and upsampling layers
    enable_autocast: enable automatic mixed precision calculation with torch.cuda.amp.autocast
    """
    def __init__(self, conv_dim=64, c_dim=4, downsample=2, repeat_num=6, 
                  enable_autocast=False, **kwargs):
        
        super(G, self).__init__()
        self.c_dim = c_dim
        self.enable_autocast = enable_autocast
        layers = []
        layers.append(conv2d(3+c_dim, conv_dim, kernel=3, stride=1, **kwargs))
        layers.append(conv2d(conv_dim, conv_dim, kernel=3, stride=1, **kwargs))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=False, track_running_stats=False)) # default pytorch setting: affine=False
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(downsample):
            layers.append(conv2d_downscale2d(curr_dim, curr_dim*2, kernel=5,  **kwargs))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=False, track_running_stats=False))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim, act='relu', inplace_act = True))

        # Up-sampling layers.
        for i in range(downsample):
            #layers.append(SNConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False))
            layers.append(upscale2d_conv2d(curr_dim, curr_dim//2, kernel=5, **kwargs))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=False, track_running_stats=False))
            layers.append(nn.ReLU(inplace=True))
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

class D(nn.Module):
    """
    Stargan-based PPHM discriminator network.
    image_size: input images thought to have same width and height.
    conv_dim: channel size of the first layer
    c_dim: class size (e.g. 4 for PPHM)
    downsample: # of downsampling, default 
    repeat_num_d: # of downsampling layers
    enable_autocast: enable automatic mixed precision calculation with torch.cuda.amp.autocast
    """
    def __init__(self, image_size=512, conv_dim=64, c_dim=4, repeat_num_d=6, 
                 enable_autocast=False, **kwargs):
        #downsampled by 2**(repeat_num +1)
        self.enable_autocast = enable_autocast
        super(D, self).__init__()
        layers = []
        layers.append(conv2d_downscale2d(3, conv_dim, kernel=5,  **kwargs))
        layers.append(apply_bias(conv_dim, **kwargs))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num_d):  
            layers.append(conv2d_downscale2d(curr_dim, curr_dim*2, kernel=5, **kwargs))
            layers.append(apply_bias(curr_dim*2, **kwargs))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2
        kernel_size = int(image_size / np.power(2, repeat_num_d))
        self.main = nn.Sequential(*layers)
        self.conv1 = conv2d(curr_dim, 1, kernel=kernel_size, **kwargs)
        self.conv2 = conv2d(curr_dim, c_dim, kernel=kernel_size, **kwargs)
        
    def forward(self, x, **kwargs):
        with autocast(self.enable_autocast): 
            h = self.main(x)
            out_src = self.conv1(h).sum(dim=[2, 3])  # fake or genuine
            out_cls = self.conv2(h).sum(dim=[2, 3])    # classes such as sad, happy , .. 
        return torch.cat([out_src, out_cls], dim=1)


 
