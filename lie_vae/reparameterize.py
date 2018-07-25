import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .utils import logsumexp
from .lie_tools import rodrigues, quaternions_to_group_matrix, \
    s2s1rodrigues, s2s2_gram_schmidt

from hyperspherical_vae_pytorch.distributions import VonMisesFisher, HypersphericalUniform


class Nreparameterize(nn.Module):
    """Reparametrize Gaussian variable."""
    def __init__(self, input_dim, z_dim):
        super().__init__()

        self.input_dim = input_dim
        self.z_dim = z_dim
        self.sigma_linear = nn.Linear(input_dim, z_dim)
        self.mu_linear = nn.Linear(input_dim, z_dim)
        self.return_means = False

        self.mu, self.sigma, self.z = None, None, None

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
        return Normal(self.mu, self.sigma).log_prob(z).sum(-1)

    def log_prior(self):
        return Normal(torch.zeros_like(self.mu), torch.ones_like(self.sigma)).log_prob(self.z).sum(-1)
   
    def nsample(self, n=1):
        if self.return_means:
            return self.mu.expand(n, -1, -1)
        eps = Normal(torch.zeros_like(self.mu), torch.ones_like(self.mu)).sample((n,))
        return self.mu + eps * self.sigma

    def deterministic(self):
        """Set to return means."""
        self.return_means = True


class Sreparameterize(nn.Module):
    """Reparametrize VMF latent variable."""

    def __init__(self, input_dim, z_dim):
        super().__init__()

        self.input_dim = input_dim
        self.z_dim = z_dim
        self.k_linear = nn.Linear(input_dim, 1)
        self.mu_linear = nn.Linear(input_dim, z_dim)
        self.return_means = False

        self.mu, self.k, self.z = None, None, None

    def forward(self, x, n=1):
        self.mu = self.mu_linear(x)
        self.mu = self.mu / self.mu.norm(p=2, dim=-1, keepdim=True)
        self.k = F.softplus(self.k_linear(x)) + 1
        self.z = self.nsample(n=n)
        return self.z

    def kl(self):
        return (-VonMisesFisher(self.mu, self.k).entropy() +
                HypersphericalUniform(self.z_dim - 1).entropy()
                .to(self.mu.device))
    
    def log_posterior(self):
        return VonMisesFisher(self.mu, self.k).log_prob(self.z)
    
    def log_prior(self):
        return HypersphericalUniform(self.z_dim - 1).log_prob(self.z)
   
    def nsample(self, n=1):
        if self.return_means:
            return self.mu.expand(n, -1, -1)
        return VonMisesFisher(self.mu, self.k).rsample(n)

    def deterministic(self):
        """Set to return means."""
        self.return_means = True


class N0reparameterize(nn.Module):
    """Reparametrize zero mean Gaussian Variable."""
    def __init__(self, input_dim, z_dim, fixed_sigma=None):
        super().__init__()

        self.input_dim = input_dim
        self.z_dim = z_dim
        self.sigma_linear = nn.Linear(input_dim, z_dim)
        self.return_means = False
        if fixed_sigma is not None:
            self.register_buffer('fixed_sigma', torch.tensor(fixed_sigma))
        else:
            self.fixed_sigma = None

        self.sigma = None
        self.z = None

    def forward(self, x, n=1):
        if self.fixed_sigma is not None:
            self.sigma = x.new_full((x.shape[0], self.z_dim), self.fixed_sigma)
        else:
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
        if self.return_means:
            return torch.zeros_like(self.sigma).expand(n, -1, -1)
        eps = Normal(torch.zeros_like(self.sigma), torch.ones_like(self.sigma)).sample((n,))
        return eps * self.sigma

    def deterministic(self):
        """Set to return means."""
        self.return_means = True


class AlgebraMean(nn.Module):
    """Module to map R^3 -> SO(3) with Algebra method."""
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
    """Module to map R^5 -> SO(3) with S2S1 method."""
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
    """Module to map R^6 -> SO(3) with S2S2 method."""
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


class SO3reparameterize(nn.Module):
    """Reparametrize SO(3) latent variable.

    It uses an inner zero mean Gaussian reparametrization module, which it
    exp-maps to a identity centered random SO(3) variable. The mean_module
    deterministically outputs a mean.
    """

    def __init__(self, reparameterize, mean_module, k=10):
        super().__init__()

        self.mean_module = mean_module
        self.reparameterize = reparameterize
        self.input_dim = self.reparameterize.input_dim
        assert self.reparameterize.z_dim == 3
        self.k = k
        self.return_means = False

        self.mu_lie, self.v, self.z = None, None, None

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
        theta = self.v.norm(p=2,dim=-1, keepdim=True)  # [n,B,1]
        u = self.v / theta  # [n,B,3]
        
        angles = 2 * math.pi * torch.arange(
            -self.k, self.k+1, device=u.device, dtype=self.v.dtype)  # [2k+1]

        theta_hat = theta[..., None, :] + angles[:, None]  # [n,B,2k+1,1]

        clamp = 1e-3
        x = u[..., None, :] * theta_hat  # [n,B,2k+1,3]

        # [n,(2k+1),B,3] or [n,(2k+1),B]
        log_p = self.reparameterize._log_posterior(x.permute([0, 2, 1, 3]).contiguous())
        
        if len(log_p.size()) == 4:
            log_p = log_p.sum(-1)  # [n,(2k+1),B]
        
        log_p = log_p.permute([0, 2, 1])  # [n,B,(2k+1)]

        theta_hat_squared = torch.clamp(theta_hat ** 2, min=clamp)
        
        log_p.contiguous()
        cos_theta_hat = torch.cos(theta_hat)

        # [n,B,(2k+1),1]
        log_vol = torch.log(theta_hat_squared / torch.clamp(2 - 2 * cos_theta_hat, min=clamp))
        log_p = log_p + log_vol.sum(-1)
        log_p = logsumexp(log_p, -1)
       
        return log_p
      
    def log_prior(self):
        prior = torch.tensor([- np.log(8 * (np.pi ** 2))], device=self.z.device)
        return prior.expand_as(self.z[..., 0, 0])

    def nsample(self, n=1):
        if self.return_means:
            return self.mu_lie.expand(n, *[-1]*len(self.mu_lie.shape))
        v_lie = rodrigues(self.v)
        return self.mu_lie @ v_lie

    def deterministic(self):
        """Set to return means."""
        self.return_means = True
        self.reparameterize.deterministic()
