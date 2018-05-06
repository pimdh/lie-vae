
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Normal

from pytorch_util import logsumexp, n2p, t2p

class Nreparameterize(nn.Module):

    def __init__(self, input_dim, z_dim):
        super(Nreparameterize, self).__init__()

        self.input_dim = input_dim
        self.z_dim = z_dim
        self.sigma_linear = nn.Linear(input_dim, z_dim)
        self.mu_linear = nn.Linear(input_dim, z_dim)

    def forward(self, x, n=1):
        self.mu = self.mu_linear(x)
        self.sigma = F.softplus(self.sigma_linear(x))
        self.z = self.nsample(n=n)
        return self.z

    def kl(self):
        return -0.5 * torch.sum(1 + 2 * self.sigma.log() - self.mu.pow(2) - self.sigma ** 2, -1)
    
    def log_posterior(self):
        return self._log_posterior(self.z)

    def _log_posterior(self, z):
        return Normal(self.mu, self.sigma).log_prob(z)

    def log_prior(self):
        return Normal(torch.zeros_like(self.mu), torch.ones_like(self.sigma)).log_prob(self.z)
   
    def nsample(self, n=1):
        eps = Normal(torch.zeros_like(self.mu), torch.ones_like(self.mu)).sample_n(n)
        return self.mu + eps * self.sigma


class SO3reparameterize(nn.Module):
    def __init__(self, reparameterize, k=10):
        super(SO3reparameterize, self).__init__()
            
        self.reparameterize = reparameterize
        self.input_dim = self.reparameterize.input_dim
        assert self.reparameterize.z_dim == 3
        self.k = k
        
        self.mu_linear = nn.Linear(self.input_dim, 3)
       
          
    @staticmethod
    def _lieAlgebra(v):
        """Map a point in R^N to the tangent space at the identity, i.e. 
        to the Lie Algebra
        Arg:
            v = vector in R^N, (..., 3) in our case
        Return:
            R = v converted to Lie Algebra element, (3,3) in our case"""
        is_cuda = v.is_cuda
        R_x = n2p(np.array([[ 0., 0., 0.],[ 0., 0.,-1.],[ 0., 1., 0.]]), cuda = is_cuda)
        R_y = n2p(np.array([[ 0., 0., 1.],[ 0., 0., 0.],[-1., 0., 0.]]), cuda = is_cuda)
        R_z = n2p(np.array([[ 0.,-1., 0.],[ 1., 0., 0.],[ 0., 0., 0.]]), cuda = is_cuda)

        R = R_x * v[..., 0, None, None] + R_y * v[..., 1, None, None] + \
            R_z * v[..., 2, None, None]
        return R
    
    @staticmethod
    def _expmap_rodrigues(v):
        is_cuda = v.is_cuda
        theta = v.norm(p=2,dim=-1, keepdim=True)
        K = SO3reparameterize._lieAlgebra(v/theta)
        I = Variable(torch.eye(3))
        I = I.cuda() if is_cuda else I
        R = I + torch.sin(theta)[...,None]*K + \
                (1. - torch.cos(theta))[...,None]*(K@K)
        a = torch.sin(theta)[...,None]
        return R
    
    def forward(self, x, n=1):
        self.mu = self.mu_linear(x)
        self.v = self.reparameterize(x, n)
        
        self.z = self.nsample(n = n)
        return self.z.view(*self.z.size()[:-2],-1)
    
    def kl(self):
        log_q_z_x = self.log_posterior()
        log_p_z = self.log_prior()
        kl = log_q_z_x - log_p_z
        return kl.mean(0)
            
    def log_posterior(self):
        
        theta = self.v.norm(p=2,dim=-1, keepdim=True) #[n,B,1]
    
        u = self.v / theta #[n,B,3]
        angles = Variable(torch.arange(-self.k, self.k+1) * 2 * math.pi) #[2k+1]
        angles = angles.cuda() if self.v.is_cuda else angles
         
        theta_hat = theta[..., None, :] + angles[:,None] #[n,B,2k+1,1]
        
        x = u[...,None,:] * theta_hat #[n,B,2k+1,3]
              
        log_p = self.reparameterize._log_posterior(x.permute([0,2,1,3])) #[n,(2k+1),B,3] or [n,(2k+1),B]
        # maybe reduce last dimension
        if len(log_p.size()) == 4:

            log_p = log_p.sum(-1) # [n,(2k+1),B]
            
        log_p = log_p.permute([0,2,1]) # [n,B,(2k+1)]
        
        log_vol = 2 * torch.log(theta_hat.abs()) - torch.log(2 - 2 * torch.cos(theta_hat)) #[n,B,(2k+1),1]
        
        log_p = log_p*log_vol.sum(-1)
        
        log_p = logsumexp(log_p, -1)
       
        return log_p
        
    def log_prior(self):
        is_cuda = self.v.is_cuda
        prior = t2p(torch.Tensor([1 / (8 * math.pi ** 2)]), cuda=is_cuda)
        return (prior.log()).expand_as(self.z[...,0,0])
        

    def nsample(self, n=1):
        # reproduce the decomposition of L-D we make
        
        mu_lie = SO3reparameterize._expmap_rodrigues(self.mu)
        v_lie = SO3reparameterize._expmap_rodrigues(self.v)
        return mu_lie @ v_lie
    

