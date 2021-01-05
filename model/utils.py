import torch
import torch.nn.functional as F
from torch.nn import Parameter as P
from torch import nn

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
  us, vs, svs = [], [], []
  for i, u in enumerate(u_):
    # Run one step of the power iteration
    with torch.no_grad():
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
        u_[i][:] = u
    # Compute this singular value and add it to the list
    svs += [torch.squeeze(torch.matmul(torch.matmul(v, W.t()), u.t()))]
    #svs += [torch.sum(F.linear(u, W.transpose(0, 1)) * v)]
  return svs, us, vs


# Spectral normalization base class 
class SN(object):
  def __init__(self, num_svs, num_itrs, num_outputs, transpose=False, eps=1e-12):
    # Number of power iterations per step
    self.num_itrs = num_itrs
    # Number of singular values
    self.num_svs = num_svs
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
    W_mat = self.weight.view(self.weight.size(0), -1)
    if self.transpose:
      W_mat = W_mat.t()
    # Apply num_itrs power iterations
    for _ in range(self.num_itrs):
      svs, us, vs = power_iteration(W_mat, self.u, update=self.training, eps=self.eps) 
    # Update the svs
    if self.training:
      with torch.no_grad(): # Make sure to do this in a no_grad() context or you'll get memory leaks!
        for i, sv in enumerate(svs):
          self.sv[i][:] = sv     
    return self.weight / svs[0]


# 2D Conv layer with spectral norm
class SNConv2d(nn.Conv2d, SN):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1,
             padding=0, dilation=1, groups=1, bias=True, 
             num_svs=1, num_itrs=1, eps=1e-12):
    nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride, 
                     padding, dilation, groups, bias)
    SN.__init__(self, num_svs, num_itrs, out_channels, eps=eps)    
  def forward(self, x):
    return F.conv2d(x, self.W_(), self.bias, self.stride, 
                    self.padding, self.dilation, self.groups)


class Attention(nn.Module):
  def __init__(self, ch, which_conv=SNConv2d, name='attention'):
    super().__init__()
    # Channel multiplier
    self.ch = ch
    self.which_conv = which_conv
    self.theta = self.which_conv(
        self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
    self.phi = self.which_conv(
        self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
    self.g = self.which_conv(self.ch, self.ch // 2,
                             kernel_size=1, padding=0, bias=False)
    self.o = self.which_conv(self.ch // 2, self.ch,
                             kernel_size=1, padding=0, bias=False)
    # Learnable gain parameter
    self.gamma = P(torch.tensor(0.), requires_grad=True)

  def forward(self, x, y=None):
    # Apply convs
    theta = self.theta(x)
    phi = F.max_pool2d(self.phi(x), [2, 2])
    g = F.max_pool2d(self.g(x), [2, 2])
    # Perform reshapes
    theta = theta.view(-1, self. ch // 8, x.shape[2] * x.shape[3])
    phi = phi.view(-1, self. ch // 8, x.shape[2] * x.shape[3] // 4)
    g = g.view(-1, self. ch // 2, x.shape[2] * x.shape[3] // 4)
    # Matmul and softmax to get attention maps
    beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
    # Attention map times g path
    o = self.o(torch.bmm(g, beta.transpose(1, 2)).view(-1,
                                                       self.ch // 2, x.shape[2], x.shape[3]))
    return self.gamma * o + x


class LinearSVDO(nn.Module):
    def __init__(self, in_features, out_features, threshold, bias=True):
        super(LinearSVDO, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.threshold = threshold

        self.W = P(torch.Tensor(out_features, in_features))
        self.log_sigma = P(torch.Tensor(out_features, in_features))
        self.bias = P(torch.Tensor(1, out_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        self.bias.data.zero_()
        self.W.data.normal_(0, 0.02)
        self.log_sigma.data.fill_(-5)        
        
    def forward(self, x):
        self.log_alpha = self.log_sigma * 2.0 - 2.0 * torch.log(1e-16 + torch.abs(self.W))
        self.log_alpha = torch.clamp(self.log_alpha, -10, 10) 
        
        if self.training:
            lrt_mean =  F.linear(x, self.W) + self.bias
            lrt_std = torch.sqrt(F.linear(x * x, torch.exp(self.log_sigma * 2.0)) + 1e-8)
            eps = lrt_std.data.new(lrt_std.size()).normal_()
            return lrt_mean + lrt_std * eps
    
        return F.linear(x, self.W * (self.log_alpha < 3).float()) + self.bias
        
    def kl_reg(self):
        # Return KL here -- a scalar 
        k1, k2, k3 = torch.Tensor([0.63576]).cuda(), torch.Tensor([1.8732]).cuda(), torch.Tensor([1.48695]).cuda()
        kl = k1 * torch.sigmoid(k2 + k3 * self.log_alpha) - 0.5 * torch.log1p(torch.exp(-self.log_alpha))
        a = - torch.sum(kl)
        return a