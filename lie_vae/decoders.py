"""Defines the MLP and action decoders."""
import torch
from torch import nn as nn

from .lie_tools import block_wigner_matrix_multiply
from lie_vae.experiments.nets import MLP


class ActionNet(nn.Module):
    """Uses proper group action."""
    def __init__(self, degrees, deconv, rep_copies=10,
                 with_mlp=False, item_rep=None, transpose=False):
        """Action decoder.

        Params:
        - degrees : max number of degrees of representation,
                    harmonics matrix has (degrees+1)^2 rows
        - deconv : deconvolutional network used after transformation
        - content_dims : content vector dimension
        - rep_copies : number of copies of representation / number of dimension
                       of signal on sphere / columns of harmonics matrix
        - harmonics_encoder_layers : number of layers of MLP that transforms
                                     content vector to harmonics matrix
        - with_mlp : route transformed harmonics through MLP before deconv
        - item_rep : optional fixed single item rep
        - transpose : Whether to take transpose of fourier matrices
        """
        super().__init__()
        self.degrees = degrees
        self.rep_copies = rep_copies
        self.matrix_dims = (degrees + 1) ** 2
        self.transpose = transpose

        if item_rep is None:
            self.item_rep = nn.Parameter(torch.randn((self.matrix_dims, rep_copies)))
        else:
            self.register_buffer('item_rep', item_rep)

        if with_mlp:
            self.mlp = MLP(self.matrix_dims * rep_copies,
                           self.matrix_dims * rep_copies, 50, 3)
        else:
            self.mlp = None

        self.deconv = deconv

    def forward(self, angles):
        """Input is ZYZ Euler angles and possibly content vector."""
        n, d = angles.shape

        assert d == 3, 'Input should be Euler angles.'

        harmonics = self.item_rep.expand(n, -1, -1)
        item = block_wigner_matrix_multiply(
            angles, harmonics, self.degrees, transpose=self.transpose) \
            .view(-1, self.matrix_dims * self.rep_copies)

        if self.mlp:
            item = self.mlp(item)

        return self.deconv(item)


class MLPNet(nn.Module):
    """Decoder that concatenates group and content vector and routes through MLP.

    Params:
    - degrees : max number of degrees of representation,
                harmonics matrix has (degrees+1)^2 rows
    - deconv : deconvolutional network used after transformation
    - in_dims : number of dimensions of (flattened) group input.
                9 for matrix, 3 for angles.
    - rep_copies : number of copies of representation / number of dimension
                   of signal on sphere / columns of harmonics matrix
    """
    def __init__(self, degrees, deconv, in_dims=9, rep_copies=10,
                 layers=3, hidden_dims=50, activation=nn.ReLU):
        super().__init__()
        matrix_dims = (degrees + 1) ** 2
        self.mlp = MLP(in_dims, matrix_dims * rep_copies, hidden_dims, layers,
                       activation)
        self.deconv = deconv

    def forward(self, x, content_data=None):
        n = x.size(0)
        x = x.view(n, -1)
        return self.deconv(self.mlp(x))
