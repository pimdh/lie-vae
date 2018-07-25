import torch
import torch.nn as nn
import numpy as np

from lie_vae.experiments.nets import ConvNet, DeconvNet, ConvNetBN, MLP
from lie_vae.decoders import MLPNet, ActionNet
from lie_vae.reparameterize import  SO3reparameterize, N0reparameterize, \
    Nreparameterize, Sreparameterize, \
    AlgebraMean, QuaternionMean, S2S1Mean, S2S2Mean

from lie_vae.lie_tools import group_matrix_to_eazyz, vector_to_eazyz, quaternions_to_eazyz
from lie_vae.utils import logsumexp
from lie_vae.experiments.utils import Flatten


class VAE(nn.Module):
    def __init__(
            self, *,
            latent_mode,
            decoder_mode,
            degrees=6,
            deconv_hidden=50,
            encode_mode='conv',
            deconv_mode='deconv',
            rep_copies=10,
            batch_norm=True,
            rgb=False,
            mean_mode='alg',
            group_reparam_in_dims=10,
            normal_dims=3,
            deterministic=False,
            item_rep=None,
            wigner_transpose=False,
            mlp_layers=3,
            mlp_hidden=50,
            mlp_activation=nn.ReLU,
            fixed_sigma=None,
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

        if encode_mode == 'conv':
            if batch_norm:
                self.encoder = ConvNetBN(
                    group_reparam_in_dims, rgb=rgb)
            else:
                self.encoder = ConvNet(
                    group_reparam_in_dims, rgb=rgb)
        elif encode_mode == 'toy':
            self.encoder = nn.Sequential(
                Flatten(),
                MLP((degrees+1)**2 * rep_copies, group_reparam_in_dims, 100, 2,
                    activation=mlp_activation)
            )
        else:
            raise ValueError('Wrong encode mode')

        # Setup latent space
        if self.latent_mode == 'so3':
            normal = N0reparameterize(group_reparam_in_dims, z_dim=3,
                                      fixed_sigma=fixed_sigma)

            if mean_mode == 'alg':
                mean_module = AlgebraMean(group_reparam_in_dims)
            elif mean_mode == 'q':
                mean_module = QuaternionMean(group_reparam_in_dims)
            elif mean_mode == 's2s1':
                mean_module = S2S1Mean(group_reparam_in_dims)
            elif mean_mode == 's2s2':
                mean_module = S2S2Mean(group_reparam_in_dims)
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

        self.reparameterize = nn.ModuleList([self.rep_group])

        # Setup decoder
        matrix_dims = (degrees + 1) ** 2
        if deconv_mode == 'deconv':
            deconv = DeconvNet(matrix_dims * rep_copies, deconv_hidden, rgb=rgb)
        elif deconv_mode == 'toy':
            deconv = nn.Sequential()
        else:
            raise RuntimeError()

        if self.decoder_mode == 'action':
            self.decoder = ActionNet(
                degrees=degrees,
                deconv=deconv,
                rep_copies=rep_copies,
                item_rep=item_rep,
                transpose=wigner_transpose,
            )
        elif self.decoder_mode == 'mlp':
            self.decoder = MLPNet(
                degrees=degrees,
                in_dims=group_dims,
                deconv=deconv,
                rep_copies=rep_copies,
                layers=mlp_layers,
                hidden_dims=mlp_hidden,
                activation=mlp_activation,
            )
        else:
            raise RuntimeError()

    def encode(self, x, n=1):
        h = self.encoder(x)

        if self.r_callback is not None:
            z = [r(f(h), n) for r, f in zip(self.reparameterize, self.r_callback)]
        else:
            z = [r(h, n) for r in self.reparameterize]

        return z

    def kl(self):
        kl = [r.kl() for r in self.reparameterize]
        return kl

    def forward(self, x, n=1):
        z = self.encode(x, n=n)
        self.z = z
        return self.decode(*z)

    def recon_loss(self, x_recon, x):
        raise NotImplementedError()

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

    def decode(self, z_pose, z_content=None):
        # Group samples and batch into batch
        batch_dims = z_pose.shape[:2]
        z_pose = z_pose.view(-1, *z_pose.shape[2:])
        if z_content is not None:
            z_content = z_content.view(-1, *z_content.shape[2:])

        if self.decoder_mode == "action" or self.decoder_mode == 'proj':
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
