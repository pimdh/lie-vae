import torch
import torch.nn as nn
import numpy as np

from .nets import CubesConvNet, CubesDeconvNet
from .decoders import MLPNet, ActionNet
from .reparameterize import  SO3reparameterize, N0reparameterize, Nreparameterize
from .lie_tools import group_matrix_to_eazyz, vector_to_eazyz
from .utils import logsumexp


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

    def decode(self, z):
        return self.decoder(z)

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
        return recon_loss, kl_summed

    def log_likelihood(self, x, n=1):
        x_recon = self.forward(x, n)
  
        log_p_z = torch.cat([r.log_prior() for r in self.reparameterize], -1)
        log_q_z_x = torch.cat([r.log_posterior() for r in self.reparameterize], -1)
        log_p_x_z = - self.recon_loss(x_recon, x)

        return (logsumexp(log_p_x_z + log_p_z - log_q_z_x, dim=0) - np.log(n)).mean()


class CubeVAE(VAE):
    def __init__(self, decoder_mode, latent_mode):
        super().__init__()
        self.decoder_mode = decoder_mode
        self.latent_mode = latent_mode

        ndf = 16
        self.encoder = CubesConvNet()

        if self.decoder_mode == "mlp":
            deconv = CubesDeconvNet((6 + 1) ** 2 * 100, 50)
            if self.latent_mode == "so3":
                self.rep0 = SO3reparameterize(N0reparameterize(ndf * 4, z_dim=3), k=10)
                self.reparameterize = [self.rep0]
                self.decoder = MLPNet(in_dims=9, degrees=6, rep_copies=100, deconv=deconv)
            elif self.latent_mode == "normal":
                self.rep0 = Nreparameterize(ndf * 4, 3)
                self.reparameterize = [self.rep0]
                self.decoder = MLPNet(in_dims=3, degrees=6, rep_copies=100, deconv=deconv)
        elif self.decoder_mode == "action":
            if self.latent_mode == "so3":
                self.rep0 = SO3reparameterize(N0reparameterize(ndf * 4, z_dim=3), k=10)
                self.reparameterize = [self.rep0]
            elif self.latent_mode == "normal":
                self.rep0 = Nreparameterize(ndf * 4, 3)
                self.reparameterize = [self.rep0]
            deconv = CubesDeconvNet((6 + 1) ** 2 * 10, 50)
            self.decoder = ActionNet(6, rep_copies=10, with_mlp=True, deconv=deconv)
        else:
            raise RuntimeError()

    def forward(self, x, n=1):
        z_list = self.encode(x)
        z_pose = z_list[0]
        z_pose_ = z_pose.view(-1, *z_pose.shape[2:])

        if self.decoder_mode == "action":
            if self.latent_mode == "so3":
                angles = group_matrix_to_eazyz(z_pose_)
            elif self.latent_mode == "normal":
                angles = vector_to_eazyz(z_pose_)

            x_recon = self.decoder(angles).view(*z_pose.shape[:2], 3, 32, 32)

        elif self.decoder_mode == "mlp":
            x_recon = self.decode(z_pose_).view(*z_pose.shape[:2], 3, 32, 32)
        else:
            raise RuntimeError()

        return x_recon

    def recon_loss(self, x_recon, x):
        x = x.expand_as(x_recon)
        max_val = (-x_recon).clamp(min=0)
        loss = x_recon - x_recon * x + max_val + ((-max_val).exp() + (-x_recon - max_val).exp()).log()

        return loss.sum(-1).sum(-1).sum(-1)