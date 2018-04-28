import numpy as np
import math

import torch
from torch import nn
from torch.autograd import Variable
from torch import Tensor as t
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam


class Net(nn.Module):
    def __init__(self, n_hidden):
        super(Net, self).__init__()
        self.hidden_1 = nn.Linear(2 * 5 * 3, n_hidden)
        self.hidden_2 = nn.Linear(n_hidden, n_hidden)
        self.hidden_3 = nn.Linear(n_hidden, n_hidden)
        self.hidden_4 = nn.Linear(n_hidden, n_hidden)
        self.hidden_mu = nn.Linear(n_hidden, 3)
        self.hidden_Ldiag = nn.Linear(n_hidden, 3)
        self.hidden_Lnondiag = nn.Linear(n_hidden, 3)
        self.hidden_random_matrix = nn.Linear(n_hidden, 9)

    def forward(self, x, random=False):

        h0 = F.relu(self.hidden_1(x))
        h01 = F.tanh(self.hidden_2(h0))
        h02 = F.relu(self.hidden_3(h01))
        h1 = F.tanh(self.hidden_4(h02))


        if random:
            M = self.hidden_random_matrix(h1)
            return 0., M.view(-1, 3, 3), 0.

        mu = self.hidden_mu(h1)
        D = F.softplus(self.hidden_Ldiag(h1))
        L = self.hidden_Lnondiag(h1)

        L = torch.cat((Variable(torch.ones(torch.Size((*D.size()[:-1], 1)))),
                  Variable(torch.zeros(torch.Size((*D.size()[:-1], 2)))),
                  L[...,0].unsqueeze(-1),
                  Variable(torch.ones(torch.Size((*D.size()[:-1], 1)))),
                  Variable(torch.zeros(torch.Size((*D.size()[:-1], 1)))),
                  L[...,1:],
                  Variable(torch.ones(torch.Size((*D.size()[:-1], 1))))),
                  -1).view(torch.Size((*D.size()[:-1], 3, 3)))

        return mu, L, D
