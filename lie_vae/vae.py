import torch
import torch.nn as nn
import numpy as np
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
