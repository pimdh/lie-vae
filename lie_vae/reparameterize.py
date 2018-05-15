import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Normal

from .utils import logsumexp, n2p, t2p
from .lie_tools import rodrigues, map2LieAlgebra

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
        return map2LieAlgebra(v)

    @staticmethod
    def _expmap_rodrigues(v):
        return rodrigues(v)

    def forward(self, x, n=1):
        self.mu = self.mu_linear(x)
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
        
        angles = torch.arange(-self.k, self.k+1, device=u.device) * 2 * math.pi #[2k+1]

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
        
        mu_lie = SO3reparameterize._expmap_rodrigues(self.mu)
        v_lie = SO3reparameterize._expmap_rodrigues(self.v)
        return mu_lie @ v_lie
    

