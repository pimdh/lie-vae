import numpy as np
import math

import torch
from torch import nn
from torch.autograd import Variable
from torch import Tensor as t
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam


def n2p(x, requires_grad = True):
    """converts numpy tensor to pytorch variable"""
    return Variable(t(x), requires_grad)

def t2c(x):
    return x.cuda()

# https://github.com/pytorch/pytorch/issues/2591
def logsumexp(inputs, dim=None, keepdim=False):
    """Numerically stable logsumexp.

    Args:
        inputs: A Variable with any shape.
        dim: An integer.
        keepdim: A boolean.

    Returns:
        Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim)).
    """
    # For a 1-D array x (any array along a single dimension),
    # log sum exp(x) = s + log sum exp(x - s)
    # with s = max(x) being a common choice.
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def deg2coo(x):
    theta = x.T[0]
    phi = x.T[1]

    xs = np.sin(phi) * np.sin(theta)
    ys = np.cos(phi) * np.sin(theta)
    zs = np.cos(theta)

    return np.vstack((xs, ys, zs)).T

def randomR():
    q, r = np.linalg.qr(np.random.normal(size=(3, 3)))
    r = np.diag(r)
    ret = q @ np.diag(r / np.abs(r))
    return ret * np.linalg.det(ret)

def canonicalShape(letter='L', size=6):
    a = np.pi / size
    canon = None
    if letter =='L':
        canon = deg2coo(np.array([[0, 0],
                               [a / 4, np.pi/2],
                               [a / 2, np.pi/2],
                               [3*a / 4, np.pi/2],
                               [a, np.pi/2],
                               [a / 2, 0],
                               [a / 4, 0]
                              ]))
    elif letter == '0':
        canon = deg2coo(np.array([[0, 0],
                                 [a / 2, 0],
                                 [a / 2, 1*np.pi/6],
                                 [a / 2, 2*np.pi/6],
                                 [a / 2, 3*np.pi/6],
                                 [a / 2, 4*np.pi/6],
                                 [a / 2, 5*np.pi/6]
                              ]))
    else:
        canon = None

    return canon

def next_batch(batch_dim, letter='L', size=6):

    canonical_s = canonicalShape(letter, size)

    originalL = np.stack([canonical_s @ randomR() for _ in range(batch_dim)])
    rotations = np.stack([randomR() for _ in range(batch_dim)])
    rotatedL = np.stack([oL @ rot for oL, rot in zip(originalL, rotations)])

    return originalL, rotatedL, rotations
