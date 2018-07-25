from torch import nn as nn, nn

from lie_vae.lie_tools import rodrigues
from lie_vae.experiments.utils import View, Flatten


class ConvNet(nn.Sequential):
    def __init__(self, out_dims, hidden_dims=50, rgb=False):
        in_dims = 3 if rgb else 1
        super().__init__(
            # input is (input_dims) x 64 x 64
            nn.Conv2d(in_dims, hidden_dims, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (hidden_dims) x 32 x 32
            nn.Conv2d(hidden_dims, hidden_dims * 2, 4, 2, 1),
            # nn.BatchNorm2d(hidden_dims * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (hidden_dims*2) x 16 x 16
            nn.Conv2d(hidden_dims * 2, hidden_dims * 4, 4, 2, 1),
            # nn.BatchNorm2d(hidden_dims * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (hidden_dims*4) x 8 x 8
            nn.Conv2d(hidden_dims * 4, hidden_dims * 8, 4, 2, 1),
            # nn.BatchNorm2d(hidden_dims * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (hidden_dims*8) x 4 x 4
            nn.Conv2d(hidden_dims * 8, out_dims, 4, 1, 0),
            # state size. out_dims x 1 x 1
            Flatten()
        )


class ConvNetBN(nn.Sequential):
    def __init__(self, out_dims, hidden_dims=50, rgb=False):
        in_dims = 3 if rgb else 1
        super().__init__(
            # input is (input_dims) x 64 x 64
            nn.Conv2d(in_dims, hidden_dims, 4, 2, 1),
            nn.BatchNorm2d(hidden_dims),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (hidden_dims) x 32 x 32
            nn.Conv2d(hidden_dims, hidden_dims * 2, 4, 2, 1),
            nn.BatchNorm2d(hidden_dims * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (hidden_dims*2) x 16 x 16
            nn.Conv2d(hidden_dims * 2, hidden_dims * 4, 4, 2, 1),
            nn.BatchNorm2d(hidden_dims * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (hidden_dims*4) x 8 x 8
            nn.Conv2d(hidden_dims * 4, hidden_dims * 8, 4, 2, 1),
            nn.BatchNorm2d(hidden_dims * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (hidden_dims*8) x 4 x 4
            nn.Conv2d(hidden_dims * 8, out_dims, 4, 1, 0),
            # state size. out_dims x 1 x 1
            Flatten()
        )


class DeconvNet(nn.Sequential):
    """1x1 to 64x64 deconvolutional stack."""
    def __init__(self, in_dims, hidden_dims, rgb=False):
        out_dims = 3 if rgb else 1
        super().__init__(
            View(-1, in_dims, 1, 1),
            nn.ConvTranspose2d(in_dims, hidden_dims, 4, 1, 0),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dims, hidden_dims, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dims, hidden_dims, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dims, hidden_dims, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dims, out_dims, 4, 2, 1),
        )


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