import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import ReLU, LeakyReLU
import numpy as np
from typing import *

#----------------------Spectral normalization --------
# from BigGAN
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

 
#------------------------------------------------------------------------------------------
#from BigGAN
# Spectral normalization base class 
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
    
# -------- modules from styleGAN --------------------------------------------------------------------------------

class _blur2d(object):
    def __init__(self,  f=[1,2,1], normalize=True, flip=False, stride=1):
        assert isinstance(stride, int) and stride >= 1
        self.stride = stride
        # Finalize filter kernel.

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

        # No-op => early exit.
        if tuple(self.f.shape) == (1, 1) and f[0,0] == 1:
            return x

        # Convolve using depthwise_conv2d.
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

        # Apply gain.
        if self.gain != 1:
            x *= self.gain

        # No-op => early exit.
        if self.factor == 1:
            return x

        # Upscale using tf.tile().
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

        # Upscale using tf.tile().
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

        # Apply gain.
        if self.gain != 1:
            x *= self.gain

        # No-op => early exit.
        if self.factor == 1:
            return x

        # Large factor => downscale using tf.nn.avg_pool().
        # NOTE: Requires tf_config['graph_options.place_pruned_graph']=True to work.
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
#----------------------------------------------------------------------------
# rewritten for pytorch
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
    

#----------------------------------------------------------------------------
# rewritten for pytorch
# Convolutional layer.

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

#----------------------------------------------------------------------------
# rewritten for pytorch
# Fused convolution + scaling.

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
        b = self.bias * lrmul
        if len(x.shape) == 2:
            return x + b
        return x + b.view([1, -1, 1, 1])
