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

def map2LieAlgebra(v):
    """Map a point in R^N to the tangent space at the identity, i.e.
    to the Lie Algebra
    Arg:
        v = vector in R^N, (..., 3) in our case
    Return:
        R = v converted to Lie Algebra element, (3,3) in our case"""

    # make sure this is a sample from R^3
    assert v.size()[-1] == 3

    R_x = n2p(np.array([[ 0., 0., 0.],
                        [ 0., 0.,-1.],
                        [ 0., 1., 0.]]))

    R_y = n2p(np.array([[ 0., 0., 1.],
                        [ 0., 0., 0.],
                        [-1., 0., 0.]]))

    R_z = n2p(np.array([[ 0.,-1., 0.],
                        [ 1., 0., 0.],
                        [ 0., 0., 0.]]))

    R = R_x * v[..., 0, None, None] + \
        R_y * v[..., 1, None, None] + \
        R_z * v[..., 2, None, None]
    return R


def map2LieVector(X):
    """Map Lie algebra in ordinary (3, 3) matrix rep to vector.

    In literature known as 'vee' map.

    inverse of map2LieAlgebra
    """
    return torch.stack((-X[..., 1, 2], X[..., 0, 2], -X[..., 0, 1]), -1)


def rodrigues(v):
    theta = v.norm(p=2, dim=-1, keepdim=True)
    # normalize K
    K = map2LieAlgebra(v/theta)

    I = Variable(torch.eye(3))
    R = I + torch.sin(theta)[...,None]*K + (1. - torch.cos(theta))[...,None]*(K@K)
    a = torch.sin(theta)[...,None]
    return R


def log_map(R):
    """Map Matrix SO(3) element to Algebra element.

    Input is taken to be 3x3 matrices of ordinary representation.
    Output algebra element in 3x3 L_i representation.
    Uses https://en.wikipedia.org/wiki/Rotation_group_SO(3)#Logarithm_map
    """
    anti_sym = .5 * (R - R.transpose(-1, -2))
    theta = torch.acos(.5 * (torch.trace(R)-1))
    return theta / torch.sin(theta) * anti_sym


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


def test_algebra_maps():
    vs = torch.randn(100, 3)
    matrices = map2LieAlgebra(vs)
    vs_prime = map2LieVector(matrices)
    matrices_prime = map2LieAlgebra(vs_prime)

    np.testing.assert_allclose(vs_prime.detach().numpy(), vs.detach().numpy())
    np.testing.assert_allclose(matrices_prime.detach().numpy(), matrices.detach().numpy())


def test_log_exp(scale, error):
    for _ in range(50):
        v_start = torch.randn(3) * scale
        R = rodrigues(v_start)
        v = map2LieVector(log_map(R))
        R_prime = rodrigues(v)
        v_prime = map2LieVector(log_map(R_prime))
        np.testing.assert_allclose(R_prime.detach(), R.detach(), rtol=error)
        np.testing.assert_allclose(v_prime.detach(), v.detach(), rtol=error)


if __name__ == '__main__':
    test_algebra_maps()
    test_log_exp(0.1, 1E-5)
    test_log_exp(10, 2E-2)

