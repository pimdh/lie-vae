import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

from .utils import logsumexp, n2p, t2p
from .lie_tools import rodrigues, map2LieAlgebra, quaternions_to_group_matrix, \
    s2s1rodrigues, s2s2_gram_schmidt
from .threevariate_normal import ThreevariateNormal
from .functions.qr import qr_batched3

import hyperspherical_vae_pytorch
from hyperspherical_vae_pytorch.distributions import VonMisesFisher, HypersphericalUniform

class Nreparameterize(nn.Module):

    def __init__(self, input_dim, z_dim):
        super(Nreparameterize, self).__init__()

        self.input_dim = input_dim
        self.z_dim = z_dim
        self.sigma_linear = nn.Linear(input_dim, z_dim)
        self.mu_linear = nn.Linear(input_dim, z_dim)

    def forward(self, x, n=1):
        #print(x.max().data.cpu().numpy(),x.min().data.cpu().numpy())
        self.mu = self.mu_linear(x)
        self.sigma = F.softplus(self.sigma_linear(x))
        self.z = self.nsample(n=n)
        return self.z

    def kl(self):
        return -0.5 * torch.sum(1 + 2 * self.sigma.log() - self.mu.pow(2) - self.sigma ** 2, -1)
    
    #def kl(self):
    #    log_q_z_x = self.log_posterior()
    #    log_p_z = self.log_prior()
    #   kl = log_q_z_x - log_p_z
    #    return kl
    
    def log_posterior(self):
        return self._log_posterior(self.z)

    def _log_posterior(self, z):
        return Normal(self.mu, self.sigma).log_prob(z).sum(-1)

    def log_prior(self):
        return Normal(torch.zeros_like(self.mu), torch.ones_like(self.sigma)).log_prob(self.z).sum(-1)
   
    def nsample(self, n=1):
        eps = Normal(torch.zeros_like(self.mu), torch.ones_like(self.mu)).sample((n,))
        return self.mu + eps * self.sigma

class Sreparameterize(nn.Module):

    def __init__(self, input_dim, z_dim):
        super(Sreparameterize, self).__init__()

        self.input_dim = input_dim
        self.z_dim = z_dim
        self.k_linear = nn.Linear(input_dim, 1)
        self.mu_linear = nn.Linear(input_dim, z_dim)

    def forward(self, x, n=1):
        self.mu = self.mu_linear(x)
        self.k = F.softplus(self.k_linear(x)) + 1 #-nn.Threshold(-2000, -2000)( - (F.softplus(self.k_linear(x)) + 1) )
        self.z = self.nsample(n=n)
        return self.z

    def kl(self):
        return - VonMisesFisher(self.mu, self.k).entropy() + HypersphericalUniform(self.z_dim - 1).entropy().to(self.mu.device) 
    
    def log_posterior(self):
        return VonMisesFisher(self.mu, self.k).log_prob(self.z)
    
    def log_prior(self):
        return HypersphericalUniform(self.z_dim - 1).log_prob(self.z)
   
    def nsample(self, n=1):
        return VonMisesFisher(self.mu, self.k).rsample(n)
    
class N0reparameterize(nn.Module):

    def __init__(self, input_dim, z_dim):
        super(N0reparameterize, self).__init__()

        self.input_dim = input_dim
        self.z_dim = z_dim
        self.sigma_linear = nn.Linear(input_dim, z_dim)
       

    def forward(self, x, n=1):
        
        self.sigma = F.softplus(self.sigma_linear(x)) 
        self.z = self.nsample(n=n)
        return self.z

    def kl(self):
        return -0.5 * torch.sum(1 + 2 * self.sigma.log() - self.sigma ** 2, -1)
    
    def log_posterior(self):
        return self._log_posterior(self.z)

    def _log_posterior(self, z):
        return Normal(torch.zeros_like(self.sigma), self.sigma).log_prob(z).sum(-1)

    def log_prior(self):
        return Normal(torch.zeros_like(self.sigma), torch.ones_like(self.sigma)).log_prob(self.z).sum(-1)
   
    def nsample(self, n=1):
        eps = Normal(torch.zeros_like(self.sigma), torch.ones_like(self.sigma)).sample((n,))
        return eps * self.sigma


class N0Fullreparameterize(nn.Module):
    """Zero mean Gaussian with full covariance matrix."""
    def __init__(self, input_dim, z_dim):
        super().__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.scale_linear = nn.Linear(input_dim, z_dim*z_dim)

    def forward(self, x, n=1):
        scale = F.softplus(self.scale_linear(x)) \
            .view(-1, self.z_dim, self.z_dim)
        # Make lower triangular
        scale = scale * x.new_ones(self.z_dim, self.z_dim).tril()

        zero_mean = x.new_zeros((x.shape[0], self.z_dim))
        self.distr = ThreevariateNormal(zero_mean, scale_tril=scale)
        self.z = self.nsample(n=n)
        return self.z

    def kl(self):
        return kl_divergence(self.distr, self.prior_distr())

    def log_posterior(self):
        return self._log_posterior(self.z)

    def _log_posterior(self, z):
        return self.distr.log_prob(z)

    def log_prior(self):
        return self.prior_distr().log_prob(self.z)

    def prior_distr(self):
        prior_scale = torch.eye(self.z_dim, device=self.z.device)[None]
        zero_mean = self.zeros_like(self.z)
        return ThreevariateNormal(zero_mean, scale_tril=prior_scale)

    def nsample(self, n=1):
        return self.distr.rsample((n,))


class AlgebraMean(nn.Module):
    def __init__(self, input_dims):
        super().__init__()
        self.map = nn.Linear(input_dims, 3)

    def forward(self, x):
        return rodrigues(self.map(x))


class QuaternionMean(nn.Module):
    def __init__(self, input_dims):
        super().__init__()
        self.map = nn.Linear(input_dims, 4)

    def forward(self, x):
        return quaternions_to_group_matrix(self.map(x))


class S2S1Mean(nn.Module):
    def __init__(self, input_dims):
        super().__init__()
        self.s2_map = nn.Linear(input_dims, 3)
        self.s1_map = nn.Linear(input_dims, 2)

    def forward(self, x):
        s2_el = self.s2_map(x)
        s2_el = s2_el/s2_el.norm(p=2, dim=-1, keepdim=True)
        
        s1_el = self.s1_map(x)
        s1_el = s1_el/s1_el.norm(p=2, dim=-1, keepdim=True)
        
        return s2s1rodrigues(s2_el, s1_el)


class S2S2Mean(nn.Module):
    def __init__(self, input_dims):
        super().__init__()
        self.map = nn.Linear(input_dims, 6)

        # Start with big outputs
        self.map.weight.data.uniform_(-10, 10)
        self.map.bias.data.uniform_(-10, 10)

    def forward(self, x):
        v = self.map(x).double().view(-1, 2, 3)
        v1, v2 = v[:, 0], v[:, 1]
        return s2s2_gram_schmidt(v1, v2).float()


class QRMean(nn.Module):
    def __init__(self, input_dims):
        super().__init__()
        self.map = nn.Linear(input_dims, 9)

    def forward(self, x):
        return qr_batched3(self.map(x).view(-1, 3, 3))[0]


class SO3reparameterize(nn.Module):
    def __init__(self, reparameterize, mean_module, k=10):
        super(SO3reparameterize, self).__init__()

        self.mean_module = mean_module
        self.reparameterize = reparameterize
        self.input_dim = self.reparameterize.input_dim
        assert self.reparameterize.z_dim == 3
        self.k = k

    @staticmethod
    def _lieAlgebra(v):
        """Map a point in R^N to the tangent space at the identity, i.e. 
        to the Lie Algebra
        Arg:
            v = vector in R^N, (..., 3) in our case
        Return:
            R = v converted to Lie Algebra element, (3,3) in our case"""
        return map2LieAlgebra(v)

    @staticmethod
    def _expmap_rodrigues(v):
        return rodrigues(v)

    def forward(self, x, n=1):
        self.mu_lie = self.mean_module(x)
        self.v = self.reparameterize(x, n)
        
        self.z = self.nsample(n = n)
        return self.z
    
    def kl(self):
        log_q_z_x = self.log_posterior()
        log_p_z = self.log_prior()
        kl = log_q_z_x - log_p_z
        return kl.mean(0)
            
    def log_posterior(self):
        
        theta = self.v.norm(p=2,dim=-1, keepdim=True) #[n,B,1]
        u = self.v / theta #[n,B,3]
        
        angles = torch.arange(-self.k, self.k+1, device=u.device, dtype=self.v.dtype) * 2 * math.pi #[2k+1]

        theta_hat = theta[..., None, :] + angles[:,None] #[n,B,2k+1,1]

        clamp = 1e-3
        
        #CLAMP FOR NUMERICAL STABILITY
        
        x = u[...,None,:] * theta_hat #[n,B,2k+1,3]
        
        log_p = self.reparameterize._log_posterior(x.permute([0,2,1,3]).contiguous()) #[n,(2k+1),B,3] or [n,(2k+1),B]
        
        # maybe reduce last dimension
        if len(log_p.size()) == 4:
            log_p = log_p.sum(-1) # [n,(2k+1),B]
        
        log_p = log_p.permute([0,2,1]) # [n,B,(2k+1)]
        
        
        
        theta_hat_squared = torch.clamp(theta_hat ** 2, min=clamp)
        
        log_p.contiguous()
        cos_theta_hat = torch.cos(theta_hat)
        
        log_vol =  torch.log(theta_hat_squared / torch.clamp(2 - 2 * cos_theta_hat, min=clamp) ) #[n,B,(2k+1),1]
        
        #print (log_vol.max())
        #print(log_vol.size())
        
        log_p = log_p + log_vol.sum(-1)
        
        log_p = logsumexp(log_p,-1) #- np.log(8 * (np.pi ** 2)) #- (2 - (2) * torch.cos(theta)).log().sum(-1)
       
        return log_p
      
    def log_prior(self):
        prior = torch.tensor([- np.log(8 * (np.pi ** 2))], device=self.z.device)
        return prior.expand_as(self.z[...,0,0])
        

    def nsample(self, n=1):
        # reproduce the decomposition of L-D we make
        v_lie = SO3reparameterize._expmap_rodrigues(self.v)
        return self.mu_lie @ v_lie
    

