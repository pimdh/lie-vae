import torch
from torch import nn as nn
import torch.nn.functional as F
from s2cnn.soft.gpu.s2_fft import S2_ifft_real
from s2cnn.utils.complex import as_complex

from .lie_tools import block_wigner_matrix_multiply, group_matrix_to_eazyz, \
    complex_block_wigner_matrix_multiply
from .utils import MLP, orthographic_grid, expand_dim


class ActionNet(nn.Module):
    """Uses proper group action."""
    def __init__(self, degrees, deconv, content_dims=10, rep_copies=10,
                 single_id=True, harmonics_encoder_layers=3,
                 with_mlp=False, item_rep=None, transpose=False):
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
        - item_rep : optional fixed single item rep
        - transpose : Whether to take transpose of fourier matrices
        """
        super().__init__()
        self.degrees = degrees
        self.rep_copies = rep_copies
        self.matrix_dims = (degrees + 1) ** 2
        self.transpose = transpose

        if single_id:
            if item_rep is None:
                self.item_rep = nn.Parameter(torch.randn((self.matrix_dims, rep_copies)))
            else:
                self.register_buffer('item_rep', item_rep)
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

        item = block_wigner_matrix_multiply(
            angles, harmonics, self.degrees, transpose=self.transpose) \
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


class ProjectionDecoder(nn.Module):
    def __init__(self, degrees, deconv, rep_copies=10, projection_size=64,
                 item_rep=None, r=0.8):
        super().__init__()
        self.deconv = deconv
        self.degrees = degrees
        self.rep_copies = rep_copies
        self.matrix_dims = (degrees + 1) ** 2
        grid = orthographic_grid(projection_size, projection_size, r=r)
        self.register_buffer('grid', torch.tensor(grid, dtype=torch.float32))
        self.ifft = S2_ifft_real()

        if item_rep is None:
            self.spectrum = nn.Parameter(torch.randn((self.matrix_dims, rep_copies, 2)))
        else:
            if item_rep.dim() == 2:
                item_rep = as_complex(item_rep)
            self.register_buffer('spectrum', item_rep)

    def forward(self, angles, _=None):
        n = angles.shape[0]
        spectrum = expand_dim(self.spectrum, n)
        rotated_spectrum = complex_block_wigner_matrix_multiply(
            angles, spectrum, self.degrees)
        rotated_signal = self.ifft(rotated_spectrum.transpose(0, 1).contiguous())

        grid = expand_dim(self.grid, n)
        projection = F.grid_sample(rotated_signal, grid)

        return self.deconv(projection)
