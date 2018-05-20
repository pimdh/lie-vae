import torch
from torch import nn as nn

from .lie_tools import block_wigner_matrix_multiply, group_matrix_to_eazyz
from .utils import MLP


class ActionNet(nn.Module):
    """Uses proper group action."""
    def __init__(self, degrees, deconv, content_dims=10, rep_copies=10,
                 single_id=True, harmonics_encoder_layers=3,
                 with_mlp=False):
        """Action decoder.

        Params:
        - degrees : max number of degrees of representation,
                    harmonics matrix has (degrees+1)^2 rows
        - deconv : deconvolutional network used after transformation
        - single_id : whether to have single content vector or not
        - content_dims : content vector dimension if single_id=False
        - rep_copies : number of copies of representation / number of dimension
                       of signal on sphere / columns of harmonics matrix
        - harmonics_encoder_layers : number of layers of MLP that transforms
                                     content vector to harmonics matrix
        - with_mlp : route transformed harmonics through MLP before deconv
        """
        super().__init__()
        self.degrees = degrees
        self.rep_copies = rep_copies
        self.matrix_dims = (degrees + 1) ** 2

        if single_id:
            self.item_rep = nn.Parameter(torch.randn((self.matrix_dims, rep_copies)))
        else:
            self.item_rep = None
            self.harmonics_encoder = MLP(
                content_dims, self.matrix_dims * self.rep_copies,
                50, harmonics_encoder_layers)

        if with_mlp:
            self.mlp = MLP(self.matrix_dims * rep_copies,
                           self.matrix_dims * rep_copies, 50, 3)
        else:
            self.mlp = None

        self.deconv = deconv

    def forward(self, angles, content_data=None):
        """Input is ZYZ Euler angles and possibly content vector."""
        assert (content_data is not None) != (self.item_rep is not None), \
            'Either must be single id or provide content_data, not both.'
        n, d = angles.shape

        assert d == 3, 'Input should be Euler angles.'


        if content_data is None:
            harmonics = self.item_rep.expand(n, -1, -1)
        else:
            harmonics = self.harmonics_encoder(content_data)\
                .view(n, self.matrix_dims, self.rep_copies)

        item = block_wigner_matrix_multiply(angles, harmonics, self.degrees) \
            .view(-1, self.matrix_dims * self.rep_copies)

        if self.mlp:
            item = self.mlp(item)

        return self.deconv(item)


class ActionNetWrapper(ActionNet):
    """Wrapper to first map rotation matrix to Euler angles."""
    def forward(self, rot, content_data=None):
        angles = group_matrix_to_eazyz(rot)
        return super().forward(angles, content_data)


class MLPNet(nn.Module):
    """Decoder that concatenates group and content vector and routes through MLP.

    Params:
    - degrees : max number of degrees of representation,
                harmonics matrix has (degrees+1)^2 rows
    - deconv : deconvolutional network used after transformation
    - in_dims : number of dimensions of (flattened) group input.
                9 for matrix, 3 for angles.
    - single_id : whether to have single content vector or not
    - content_dims : content vector dimension if single_id=False
    - rep_copies : number of copies of representation / number of dimension
                   of signal on sphere / columns of harmonics matrix
    """
    def __init__(self, degrees, deconv, in_dims=9, rep_copies=10,
                 content_dims=10, single_id=True):
        super().__init__()
        matrix_dims = (degrees + 1) ** 2

        if not single_id:
            in_dims += content_dims

        self.mlp = MLP(in_dims, matrix_dims * rep_copies, 50, 3)
        self.deconv = deconv
        self.single_id = single_id

    def forward(self, x, content_data=None):
        assert (content_data is None) != (not self.single_id), \
            'Either must be single id or provide content_data, not both.'
        n = x.size(0)
        x = x.view(n, -1)

        if content_data is not None:
            x = torch.cat((x, content_data.view(n, -1)), 1)

        return self.deconv(self.mlp(x))