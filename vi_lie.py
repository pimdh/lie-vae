import numpy as np
import math

import torch
from torch import nn
from torch.autograd import Variable
from torch import Tensor as t
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam

from utils import *
from lie_tools import map2LieVector, map2LieAlgebra, rodrigues, log_map



def log_density(v, L, D, k = 10):
    theta = v.norm(p=2,dim=-1, keepdim=True)
    u = v / theta
    angles = Variable(torch.arange(-k, k+1) * 2 * math.pi)
    theta_hat = theta[...,None] + angles
    x = u[...,None] * theta_hat

    L_hat = L - Variable(torch.eye(3))
    L_inv = Variable(torch.eye(3)) - L_hat + L_hat@L_hat
    D_inv = 1. / D
    A = L_inv @ x

    p = -0.5*(A * D_inv[...,None] * A + 2 * torch.log(theta_hat.abs()) -\
                      torch.log(2 - 2 * torch.cos(theta_hat)) ).sum(-2)
    p = logsumexp(p, -1)
    p += -0.5*(torch.log(D.prod(-1)) + v.size()[-1]*math.log(2.*math.pi))

    return p

def get_sample(N, x_mb, xo, var_noise=True):
    mu, L, D = N(n2p(x_mb))
    noise = Variable(Normal(t(np.zeros(3)), t(np.ones(3))).sample_n(1))
    v = (L @ (D.pow(0.5)*noise)[..., None]).squeeze()
    mu_lie = rodrigues(mu)
    v_lie = rodrigues(v)
    g_lie = mu_lie
    if var_noise:
        g_lie = g_lie @ v_lie

    xrot_recon = (n2p(xo) @ g_lie).data.numpy()
    return xrot_recon
