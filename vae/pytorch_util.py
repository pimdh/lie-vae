
import gzip
import pickle

import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as data_utils
from torch.autograd import Variable
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, H=[1, 10, 1], activation=nn.ReLU, end_activation=False):
        super(MLP, self).__init__()
        self.H = H
        self.activation = activation
        modules = []
        for input_dim, output_dim in zip(H, H[1:-1]):
            modules.append(nn.Linear(input_dim, output_dim))
            modules.append(self.activation())
        modules.append(nn.Linear(H[-2], H[-1]))
        if end_activation:
            modules.append(self.activation())
        self.module = nn.Sequential(*modules)

    def forward(self, x):
        y = self.module(x)
        return y


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


def load_mnist_data(path, batch_size):
    with gzip.open(path, 'rb') as f:
        dataset = pickle.load(f)

    train_data = torch.from_numpy(
        dataset["train"]["images"][:, None, :, :].astype(np.float32))
    train_labels = torch.from_numpy(
        dataset["train"]["labels"].astype(np.int64))

    mean = train_data.mean()
    stdv = train_data.std()

    train_dataset = data_utils.TensorDataset(train_data, train_labels)
    train_loader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_data = torch.from_numpy(
        dataset["test"]["images"][:, None, :, :].astype(np.float32))
    test_labels = torch.from_numpy(
        dataset["test"]["labels"].astype(np.int64))

    test_dataset = data_utils.TensorDataset(test_data, test_labels)
    test_loader = data_utils.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader, train_dataset, test_dataset
