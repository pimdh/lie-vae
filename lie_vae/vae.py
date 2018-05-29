import torch
import torch.nn as nn
import numpy as np

from .nets import CubesConvNet, CubesDeconvNet, ChairsConvNet, ChairsDeconvNet, \
    ChairsDeconvNetUpsample, CubesConvNetBN, ChairsConvNetBN, ChairsDeconvNet4, \
    ChairsDeconvNet8
from .decoders import MLPNet, ActionNet
from .reparameterize import  SO3reparameterize, N0reparameterize, \
    Nreparameterize, Sreparameterize, N0Fullreparameterize, \
    AlgebraMean, QuaternionMean, QRMean, S2S1Mean, S2S2Mean

from .lie_tools import group_matrix_to_eazyz, vector_to_eazyz, quaternions_to_eazyz
from .utils import logsumexp, tensor_slicer, MLP, Flatten


class VAE(nn.Module):
    """General VAE class."""
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = None
        self.decoder = None
        self.reparameterize = []
        self.r_callback = None

    def encode(self, x, n=1):
        h = self.encoder(x)

        if self.r_callback is not None:
            z = [r(f(h), n) for r, f in zip(self.reparameterize, self.r_callback)]
        else:
            z = [r(h, n) for r in self.reparameterize]

        return z

    def kl(self):
        # NOTE always call after encode
        # TODO make this bahaviour impossible
        kl = [r.kl() for r in self.reparameterize]
        return kl

    def decode(self, *z):
        return self.decoder(*z)

    def forward(self, x, n=1):
        z = self.encode(x, n=n)

        # we do not stack anymore to allow the decoder to use differently each input

        return self.decode(*z)

    def recon_loss(self, x_recon, x):
        raise NotImplemented

    def elbo(self, x, n=1):
        x_recon = self.forward(x, n)
        kl = self.kl()
        # TODO maybe sum directly  without stacking 
        kl_summed = torch.sum(torch.stack(kl, -1), -1)
        recon_loss = self.recon_loss(x_recon, x)
        return recon_loss, kl_summed, kl

    def log_likelihood(self, x, n=1):
        x_recon = self.forward(x, n)
  
        log_p_z = torch.cat([r.log_prior() for r in self.reparameterize], -1).to(x.device)
        log_q_z_x = torch.cat([r.log_posterior() for r in self.reparameterize], -1).to(x.device)
        log_p_x_z = - self.recon_loss(x_recon, x)

        return (logsumexp(log_p_x_z + log_p_z - log_q_z_x, dim=0) - np.log(n)).mean()


class CubeVAE(VAE):
    def __init__(self, decoder_mode, latent_mode, mean_mode='alg', batch_norm=False):
        super().__init__()
        self.decoder_mode = decoder_mode
        self.latent_mode = latent_mode

        ndf = 16
        if batch_norm:
            self.encoder = CubesConvNetBN()
        else:
            self.encoder = CubesConvNet()

        if self.decoder_mode == "mlp":
            deconv = CubesDeconvNet((6 + 1) ** 2 * 10, 50)
            if self.latent_mode == "so3":
                self.rep0 = SO3reparameterize(N0reparameterize(ndf * 4, z_dim=3), k=10)
                self.reparameterize = [self.rep0]
                self.decoder = MLPNet(in_dims=9, degrees=6, rep_copies=10, deconv=deconv)
            elif self.latent_mode == "normal":
                self.rep0 = Nreparameterize(ndf * 4, 3)
                self.reparameterize = [self.rep0]
                self.decoder = MLPNet(in_dims=3, degrees=6, rep_copies=10, deconv=deconv)
            elif self.latent_mode == "vmf":
                self.rep0 = Sreparameterize(ndf * 4, 4)
                self.reparameterize = [self.rep0]
                self.decoder = MLPNet(in_dims=4, degrees=6, rep_copies=10, deconv=deconv)
                
        elif self.decoder_mode == "action":
            if self.latent_mode == "so3":
                self.rep0 = SO3reparameterize(N0reparameterize(ndf * 4, z_dim=3), k=10)
                self.reparameterize = [self.rep0]
            elif self.latent_mode == "normal":
                self.rep0 = Nreparameterize(ndf * 4, 3)
                self.reparameterize = [self.rep0]
            elif self.latent_mode == "vmf":
                self.rep0 = Sreparameterize(ndf * 4, 4)
                self.reparameterize = [self.rep0]
                
        if self.latent_mode == 'so3' or self.latent_mode == 'so3f':
            if self.latent_mode == 'so3':
                normal = N0reparameterize(ndf * 4, z_dim=3)
            else:
                normal = N0Fullreparameterize(ndf * 4, z_dim=3)

            if mean_mode == 'alg':
                mean_module = AlgebraMean(ndf * 4)
            elif mean_mode == 'q':
                mean_module = QuaternionMean(ndf * 4)
            elif mean_mode == 's2s1':
                mean_module = S2S1Mean(ndf * 4)
            elif mean_mode == 'qr':
                mean_module = QRMean(ndf * 4)
            else:
                raise ValueError('Wrong mean mode')

            self.rep0 = SO3reparameterize(normal, mean_module, k=10)
            group_dims = 9
        elif self.latent_mode == 'normal':
            self.rep0 = Nreparameterize(ndf * 4, 3)
            group_dims = 3
        elif self.latent_mode == 'vmf':
            self.rep0 = Sreparameterize(ndf * 4, 4)
            group_dims = 4
        else:
            raise ValueError('Wrong latent mode')

        if self.decoder_mode == 'mlp':
            deconv = CubesDeconvNet((6 + 1) ** 2 * 100, 50)
            self.decoder = MLPNet(in_dims=group_dims, degrees=6, rep_copies=100, deconv=deconv)
        elif self.decoder_mode == 'action':
            deconv = CubesDeconvNet((6 + 1) ** 2 * 10, 50)
            self.decoder = ActionNet(6, rep_copies=10, with_mlp=True, deconv=deconv)
        else:
            raise ValueError('Wrong decoder mode')

        self.reparameterize = [self.rep0]

    def forward(self, x, n=1):
        z_list = self.encode(x, n=n)
        z_pose = z_list[0]
        z_pose_ = z_pose.view(-1, *z_pose.shape[2:])

        if self.decoder_mode == "action":
            if self.latent_mode == "so3" or self.latent_mode == 'so3f':
                angles = group_matrix_to_eazyz(z_pose_)
            elif self.latent_mode == "normal" or self.latent_mode == "vmf":
                angles = vector_to_eazyz(z_pose_)

            x_recon = self.decoder(angles).view(*z_pose.shape[:2], 3, 32, 32)

        elif self.decoder_mode == "mlp":
            x_recon = self.decode(z_pose_).view(*z_pose.shape[:2], 3, 32, 32)
        else:
            raise RuntimeError()

        return x_recon

    def recon_loss(self, x_recon, x):
        x = x.expand_as(x_recon)
#         max_val = (-x_recon).clamp(min=0)
#         loss = x_recon - x_recon * x + max_val + ((-max_val).exp() + (-x_recon - max_val).exp()).log()
        return ((x_recon - x) ** 2).sum(-1).sum(-1).sum(-1)
#         return loss.sum(-1).sum(-1).sum(-1)


class ChairsVAE(VAE):
    def __init__(
            self, *,
            latent_mode,
            decoder_mode,
            content_dims=10,
            degrees=6,
            deconv_hidden=50,
            encode_mode='conv',
            deconv_mode='deconv',
            rep_copies=10,
            batch_norm=True,
            rgb=False,
            single_id=False,
            mean_mode='alg',
            group_reparam_in_dims=10,
            normal_dims=3,
            deterministic=False,
            item_rep=None,
            wigner_transpose=False,
    ):
        """See lie_vae/decoders.py for explanation of params."""
        super().__init__()

        self.latent_mode = latent_mode
        self.decoder_mode = decoder_mode

        if deconv_mode == 'toy':
            self.out_shape = ((degrees+1)**2, rep_copies)
        else:
            self.out_shape = (3 if rgb else 1, 64, 64)

        if self.latent_mode == 'normal':
            if self.decoder_mode != 'mlp' and normal_dims != 3:
                raise ValueError('Normal Action must be 3 dim')
            # Make sure we don't have a bottleneck before
            group_reparam_in_dims = max(group_reparam_in_dims, normal_dims)

        content_reparam_in_dims = 0 if single_id else content_dims
        if encode_mode == 'conv':
            if batch_norm:
                self.encoder = ChairsConvNetBN(
                    group_reparam_in_dims + content_reparam_in_dims, rgb=rgb)
            else:
                self.encoder = ChairsConvNet(
                    group_reparam_in_dims + content_reparam_in_dims, rgb=rgb)
        elif encode_mode == 'toy':
            self.encoder = nn.Sequential(
                Flatten(),
                MLP((degrees+1)**2 * rep_copies, group_reparam_in_dims, 100, 2)
            )
        else:
            raise ValueError('Wrong encode mode')

        # Setup latent space
        if self.latent_mode == 'so3' or self.latent_mode == 'so3f':
            if self.latent_mode == 'so3':
                normal = N0reparameterize(group_reparam_in_dims, z_dim=3)
            else:
                normal = N0Fullreparameterize(group_reparam_in_dims, z_dim=3)

            if mean_mode == 'alg':
                mean_module = AlgebraMean(group_reparam_in_dims)
            elif mean_mode == 'q':
                mean_module = QuaternionMean(group_reparam_in_dims)
            elif mean_mode == 's2s1':
                mean_module = S2S1Mean(group_reparam_in_dims)
            elif mean_mode == 's2s2':
                mean_module = S2S2Mean(group_reparam_in_dims)
            elif mean_mode == 'qr':
                mean_module = QRMean(group_reparam_in_dims)
            else:
                raise ValueError('Wrong mean mode')

            self.rep_group = SO3reparameterize(normal, mean_module, k=10)
            group_dims = 9
        elif self.latent_mode == 'normal':
            self.rep_group = Nreparameterize(group_reparam_in_dims, normal_dims)
            group_dims = normal_dims
        elif self.latent_mode == 'vmf' or self.latent_mode == 'vmfq':
            self.rep_group = Sreparameterize(group_reparam_in_dims, 4)
            group_dims = 4
        else:
            raise ValueError('Wrong latent mode')

        if deterministic:
            self.rep_group.deterministic()

        if single_id:
            self.reparameterize = nn.ModuleList([self.rep_group])
        else:
            self.rep_content = Nreparameterize(
                content_reparam_in_dims, z_dim=content_dims)
            self.reparameterize = nn.ModuleList([self.rep_group, self.rep_content])

            # Split output of encoder
            self.r_callback = [tensor_slicer(0, group_reparam_in_dims),
                               tensor_slicer(group_reparam_in_dims, None)]

        # Setup decoder
        matrix_dims = (degrees + 1) ** 2
        if deconv_mode == 'deconv':
            deconv = ChairsDeconvNet(matrix_dims * rep_copies, deconv_hidden, rgb=rgb)
        elif deconv_mode == 'deconv4':
            deconv = ChairsDeconvNet4(matrix_dims * rep_copies, deconv_hidden, rgb=rgb)
        elif deconv_mode == 'deconv8':
            deconv = ChairsDeconvNet8(matrix_dims * rep_copies, deconv_hidden, rgb=rgb)
        elif deconv_mode == 'upsample':
            deconv = ChairsDeconvNetUpsample(matrix_dims * rep_copies, deconv_hidden, rgb=rgb)
        elif deconv_mode == 'toy':
            deconv = nn.Sequential()
        else:
            raise RuntimeError()

        if self.decoder_mode == 'action':
            self.decoder = ActionNet(
                degrees=degrees,
                deconv=deconv,
                content_dims=content_dims,
                rep_copies=rep_copies,
                single_id=single_id,
                item_rep=item_rep,
                transpose=wigner_transpose,
            )
        elif self.decoder_mode == 'mlp':
            self.decoder = MLPNet(
                degrees=degrees,
                in_dims=group_dims,
                deconv=deconv,
                content_dims=content_dims,
                rep_copies=rep_copies,
                single_id=single_id)
        else:
            raise RuntimeError()

    def decode(self, z_pose, z_content=None):
        # Group samples and batch into batch
        batch_dims = z_pose.shape[:2]
        z_pose = z_pose.view(-1, *z_pose.shape[2:])
        if z_content is not None:
            z_content = z_content.view(-1, *z_content.shape[2:])

        if self.decoder_mode == "action":
            if self.latent_mode == "so3" or self.latent_mode == 'so3f':
                angles = group_matrix_to_eazyz(z_pose)
            elif self.latent_mode == "normal" or self.latent_mode == "vmf":
                angles = vector_to_eazyz(z_pose)
            elif self.latent_mode == "vmfq":
                angles = quaternions_to_eazyz(z_pose)
            else:
                raise RuntimeError()

            x_recon = self.decoder(angles, z_content)

        elif self.decoder_mode == "mlp":
            x_recon = self.decoder(z_pose, z_content)
        else:
            raise RuntimeError()

        return x_recon.reshape(*batch_dims, *self.out_shape)

    def recon_loss(self, x_recon, x):
        x = x.expand_as(x_recon)
        l = ((x_recon - x) ** 2)
        for _ in range(len(self.out_shape)):
            l = l.sum(-1)
        return l
