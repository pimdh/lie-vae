import numpy as np
from functools import partial

import torch
from torch import nn
from torch.autograd import Variable
from itertools import accumulate
from torch.utils.data import Dataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def n2p(x, requires_grad=True, cuda=False):
    """converts numpy tensor to pytorch variable"""
    x_pt = Variable(torch.Tensor(x), requires_grad)
    if cuda:
        x_pt = x_pt.cuda()

    return x_pt

def t2p(x, requires_grad=True, cuda=False):
    x_pt = Variable(x)
    if cuda:
        x_pt = x_pt.cuda()
    return x_pt


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
    """For proof, see https://math.stackexchange.com/a/138837/243884"""
    q, r = np.linalg.qr(np.random.normal(size=(3, 3)))
    r = np.diag(r)
    # TODO: What's going on here??
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


# Some code from master branch that allows for random dataset splits
class Subset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def random_split(dataset, lengths):
    assert sum(lengths) == len(dataset)
    seed = np.random.get_state()
    np.random.seed(0)
    indices = torch.tensor(np.random.permutation(sum(lengths)),
                           dtype=torch.long)
    np.random.set_state(seed)

    return [Subset(dataset, indices[offset - length:offset])
            for offset, length in zip(accumulate(lengths), lengths)]


class MLP(nn.Sequential):
    """Helper module to create MLPs."""
    def __init__(self, input_dims, output_dims, hidden_dims,
                 num_layers=1, activation=nn.ReLU):
        if num_layers == 0:
            super().__init__(nn.Linear(input_dims, output_dims))
        else:
            super().__init__(
                nn.Linear(input_dims, hidden_dims),
                activation(),
                *[l for _ in range(num_layers-1)
                  for l in [nn.Linear(hidden_dims, hidden_dims), activation()]],
                nn.Linear(hidden_dims, output_dims)
            )


class View(nn.Module):
    def __init__(self, *v):
        super(View, self).__init__()
        self.v = v

    def forward(self, x):
        return x.view(*self.v)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


def encode(encoder, single_id, item_label, rot_label, img_label):
    if encoder:
        rot, content_data = encoder(img_label)
    else:
        rot = rot_label
        if not single_id:
            content_data = torch.eye(10, device=device)[item_label]
        else:
            content_data = None
    return rot, content_data


def tensor_slice(start, end, x):
    """Slice last dimension of tensor."""
    return x[..., start:end]


def tensor_slicer(start, end):
    """Return function that slices tensor in last dimension."""
    return partial(tensor_slice, start, end)