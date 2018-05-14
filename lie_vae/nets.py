from torch import nn as nn

from .lie_tools import rodrigues
from .utils import View, Flatten


class ChairsEncoder(nn.Module):
    def __init__(self, id_dims=0, hidden_dims=50):
        super().__init__()
        self.id_dims = id_dims
        out_dims = 3 + id_dims
        self.conv = ChairsConvNet(out_dims, hidden_dims)

    def forward(self, img):
        x = self.conv(img)
        algebra = x[:, :3]
        if self.id_dims:
            id_data = x[:, 3:]
        else:
            id_data = None
        return rodrigues(algebra), id_data


class ChairsConvNet(nn.Sequential):
    def __init__(self, out_dims, hidden_dims=50):
        input_dims = 1
        super().__init__(
            # input is (input_dims) x 64 x 64
            nn.Conv2d(input_dims, hidden_dims, 4, 2, 1),
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


class ChairsDeconvNet(nn.Sequential):
    """1x1 to 64x64 deconvolutional stack."""
    def __init__(self, in_dims, hidden_dims):
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
            nn.ConvTranspose2d(hidden_dims, 1, 4, 2, 1),
        )


class CubesConvNet(nn.Sequential):
    def __init__(self):
        ndf = 16
        super().__init__(
            # input is (nc) x 32 x 32
            nn.Conv2d(3, ndf, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, ndf * 8, 4, 1, 0, bias=False),
            Flatten(),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(ndf * 8, ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )


class CubesDeconvNet(nn.Sequential):
    """1x1 to 32x32 deconvolutional stack."""
    def __init__(self, in_dims, hidden_dims):
        super().__init__(
            nn.Linear(in_dims, 32 * 8 * 8 ),
            View(-1, 32, 8, 8),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, 3, padding=1)
        )