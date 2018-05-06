
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
        self.z = self.nsample(self.mu, self.sigma, n=n)
        return self.z

    def kl(self):
        return -0.5 * torch.sum(1 + 2 * self.sigma.log() - self.mu.pow(2) - self.sigma ** 2, -1)

    def log_posterior(self):
        return Normal(self.mu, self.sigma).log_prob(self.z)

    def log_prior(self):
        return Normal(torch.zeros_like(self.mu), torch.ones_like(self.sigma)).log_prob(self.z)
   
    @staticmethod
    def nsample(mu, sigma, n=1):
        eps = Normal(torch.zeros_like(mu), torch.ones_like(mu)).sample_n(n)
        return mu + eps * sigma


class SO3reparameterize(nn.Module):

    def __init__(self, input_dim, z_dim=3, k=3):
        super(SO3reparameterize, self).__init__()

        self.input_dim = input_dim
        self.z_dim = 3
        self.k = k

        self.mu_linear = nn.Linear(input_dim, 3)
        self.Ldiag_linear = nn.Linear(input_dim, 3)
        self.Lnondiag_linear = nn.Linear(input_dim, 3)

    @staticmethod
    def _lieAlgebra(v):
        """Map a point in R^N to the tangent space at the identity, i.e. 
        to the Lie Algebra
        Arg:
            v = vector in R^N, (..., 3) in our case
        Return:
            R = v converted to Lie Algebra element, (3,3) in our case"""
        is_cuda = v.is_cuda
        
        R_x = n2p(np.array([[0., 0., 0.], [0., 0., -1.], [0., 1., 0.]]), cuda=is_cuda)
        R_y = n2p(np.array([[0., 0., 1.], [0., 0., 0.], [-1., 0., 0.]]), cuda=is_cuda)
        R_z = n2p(np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 0.]]), cuda=is_cuda)

        R = R_x * v[..., 0, None, None] + R_y * v[..., 1, None, None] + \
            R_z * v[..., 2, None, None]
        return R

    @staticmethod
    def _expmap_rodrigues(v):
        is_cuda = v.is_cuda
        theta = v.norm(p=2, dim=-1, keepdim=True)
        K = SO3reparameterize._lieAlgebra(v / theta)
        I = t2p(torch.eye(3), cuda=is_cuda)
        R = I + torch.sin(theta)[..., None] * K + \
            (1. - torch.cos(theta))[..., None] * (K @ K)
        a = torch.sin(theta)[..., None]
        return R

    def forward(self, x, n=1):
        self.mu = self.mu_linear(x)
        self.D = F.softplus(self.Ldiag_linear(x))
        L = self.Lnondiag_linear(x)

        is_cuda = L.is_cuda
        self.L = torch.cat((t2p(torch.ones(torch.Size((*self.D.size()[:-1], 1))), cuda=is_cuda),
                            t2p(torch.zeros(torch.Size((*self.D.size()[:-1], 2))), cuda=is_cuda),
                            L[..., 0].unsqueeze(-1),
                            t2p(torch.ones(torch.Size((*self.D.size()[:-1], 1))), cuda=is_cuda),
                            t2p(torch.zeros(torch.Size((*self.D.size()[:-1], 1))), cuda=is_cuda),
                            L[..., 1:],
                            t2p(torch.ones(torch.Size((*self.D.size()[:-1], 1))), cuda=is_cuda)), -1).view(
            torch.Size((*self.D.size()[:-1], 3, 3)))

        self.v, self.z = self.nsample(self.mu, self.L, self.D, n=n)

        return self.z.view(*self.z.size()[:-2],-1)

    def kl(self):
        log_q_z_x = self.log_posterior()
        log_p_z = self.log_prior()
        
        kl = log_q_z_x - log_p_z
        return kl.mean(0)

    def log_posterior(self):
        is_cuda = self.v.is_cuda
        
        theta = self.v.norm(p=2, dim=-1, keepdim=True)
#         print ('\ntheta\n')
#         print (theta)
        u = self.v / theta
#         print ('u\n')
#         print (u)
        angles = t2p(torch.arange(-self.k, self.k + 1) * 2 * math.pi, cuda=is_cuda)
#         print ('angles \n')
#         print (angles)
        theta_hat = theta[..., None] + angles
#         print ('theta_hat\n')
#         print (theta_hat)
        x = u[..., None] * theta_hat
#         print ('x\n')
#         print (x)
        L_hat = self.L - t2p(torch.eye(3), cuda=is_cuda)
#         print ('l hat \n')
#         print (L_hat)
        L_inv = t2p(torch.eye(3), cuda=is_cuda) - L_hat + L_hat @ L_hat
#         print ('l inv \n')
#         print (L_inv)
        D_inv = 1. / self.D
#         print ('d inv \n')
#         print (D_inv)
        A = L_inv @ x
#         print ('a \n')
#         print (A)
#         print ("\np\n")
        p = -0.5 * (A * D_inv[..., None] * A + 2 * torch.log(theta_hat.abs()) - \
                    torch.log(2 - 2 * torch.cos(theta_hat))).sum(-2)
#         print (p)
        p = logsumexp(p, -1)
#         print (p)
        p += -0.5 * (torch.log(self.D.prod(-1)) + self.v.size()[-1] * math.log(2. * math.pi))
#         print (p)
        return p

    def log_prior(self):
        # To DO :
        is_cuda = self.v.is_cuda
        prior = t2p(torch.Tensor([1 / (8 * math.pi ** 2)]), cuda=is_cuda)
        return (prior.log()).expand_as(self.z[...,0,0])

    @staticmethod
    def nsample(mu, L, D, n=1):
        # reproduce the decomposition of L-D we make
        eps = Normal(torch.zeros_like(mu), torch.ones_like(mu)).sample_n(n)
        v = (L @ (D.pow(0.5) * eps)[..., None]).squeeze(-1)
        mu_lie = SO3reparameterize._expmap_rodrigues(mu)
        v_lie = SO3reparameterize._expmap_rodrigues(v)
        return v, mu_lie @ v_lie
