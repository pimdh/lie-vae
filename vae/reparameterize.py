
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Normal

from pytorch_util import logsumexp, n2p


class Nreparametrize(nn.Module):

    def __init__(self, input_dim, z_dim):
        super(Nreparametrize, self).__init__()

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


class SO3Reparameterize(nn.Module):

    def __init__(self, input_dim, k=10):
        super(SO3Reparameterize, self).__init__()

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
        R_x = n2p(np.array([[0., 0., 0.], [0., 0., -1.], [0., 1., 0.]]))
        R_y = n2p(np.array([[0., 0., 1.], [0., 0., 0.], [-1., 0., 0.]]))
        R_z = n2p(np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 0.]]))

        R = R_x * v[..., 0, None, None] + R_y * v[..., 1, None, None] + \
            R_z * v[..., 2, None, None]
        return R

    @staticmethod
    def _expmap_rodrigues(v):
        theta = v.norm(p=2, dim=-1, keepdim=True)
        K = SO3Reparameterize._lieAlgebra(v / theta)
        I = Variable(torch.eye(3))
        R = I + torch.sin(theta)[..., None] * K + \
            (1. - torch.cos(theta))[..., None] * (K @ K)
        a = torch.sin(theta)[..., None]
        return R

    def forward(self, x, n=1):
        self.mu = self.mu_linear(x)
        self.D = F.softplus(self.Ldiag_linear(x))
        L = self.Lnondiag_linear(x)

        self.L = torch.cat((Variable(torch.ones(torch.Size((*self.D.size()[:-1], 1)))),
                            Variable(torch.zeros(torch.Size((*self.D.size()[:-1], 2)))),
                            L[..., 0].unsqueeze(-1),
                            Variable(torch.ones(torch.Size((*self.D.size()[:-1], 1)))),
                            Variable(torch.zeros(torch.Size((*self.D.size()[:-1], 1)))),
                            L[..., 1:],
                            Variable(torch.ones(torch.Size((*self.D.size()[:-1], 1))))), -1).view(
            torch.Size((*self.D.size()[:-1], 3, 3)))

        self.v, self.z = self.nsample(self.mu, self.L, self.D, n=n)

        return self.z

    def kl(self):
        kl = 0
        return kl

    def log_posterior(self):
        theta = self.v.norm(p=2, dim=-1, keepdim=True)
        u = self.v / theta
        angles = Variable(torch.arange(-self.k, self.k + 1) * 2 * math.pi)
        theta_hat = theta[..., None] + angles
        x = u[..., None] * theta_hat

        L_hat = self.L - Variable(torch.eye(3))
        L_inv = Variable(torch.eye(3)) - L_hat + L_hat @ L_hat
        D_inv = 1. / self.D
        A = L_inv @ x

        p = -0.5 * (A * D_inv[..., None] * A + 2 * torch.log(theta_hat.abs()) - \
                    torch.log(2 - 2 * torch.cos(theta_hat))).sum(-2)
        p = logsumexp(p, -1)
        p += -0.5 * (torch.log(self.D.prod(-1)) + self.v.size()[-1] * math.log(2. * math.pi))

        return p

    def log_prior(self):
        # To DO :
        return 1 / (8 * math.pi ** 2)

    @staticmethod
    def nsample(mu, L, D, n=1):
        # reproduce the decomposition of L-D we make
        eps = Normal(torch.zeros_like(mu), torch.ones_like(mu)).sample_n(n)
        v = (L @ (D.pow(0.5) * eps)[..., None]).squeeze(-1)
        mu_lie = SO3Reparameterize._expmap_rodrigues(mu)
        v_lie = SO3Reparameterize._expmap_rodrigues(v)
        return v, mu_lie @ v_lie
