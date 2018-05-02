
import torch
import torch.nn as nn
import torch.nn.functional as F

from SNets import S2ConvNet, S2DeconvNet
from pytorch_util import MLP
from reparameterize import Nreparametrize


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = None
        self.decoder = None
        self.reparametrize = []
        self.r_callback = []

    def encode(self, x, n=1):
        h = self.encoder(x)
        z = [r(f(h), n) for r, f in zip(self.reparametrize, self.r_callback)]

        return z

    def kl(self):
        # NOTE always call after encode
        # TODO make this bahaviour impossible
        kl = [r.kl() for r in self.reparametrize]
        return kl

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, n=1):
        z = self.encode(x, n=n)

        # flatten and stack z
        z_cat = torch.cat([v.view(n, x.size()[0], -1) for v in z], -1)

        return self.decode(z_cat)

    def recon_loss(self, x, x_recon):
        raise NotImplemented

    def elbo(self, x):
        x_recon = self.forward(x)[0]
        kl = self.kl()
        # TODO maybe sum directly  without stacking 
        kl_summed = torch.sum(torch.stack(kl, -1), -1)
        recon_loss = self.recon_loss(x, x_recon)
        return recon_loss, kl_summed

    def log_likelihood(self, x, n=1):
        raise NotImplemented


class NS2VAE(VAE):
    def __init__(self, z_dim=10,
                 encoder_f=[1, 10, 10],
                 decoder_f=[10, 10, 1],
                 encoder_b=[30, 20, 6],
                 decoder_b=[5, 15, 30],
                 mlp_h=[100]):
        super(NS2VAE, self).__init__()

        self.encoder = S2ConvNet(f_list=encoder_f, b_list=encoder_b)
        self.decoder = S2DeconvNet(f_list=decoder_f, b_list=decoder_b, mlp_dim=[z_dim])

        self.mlp_h = mlp_h.copy()
        self.mlp_h.insert(0, encoder_f[-1] * (encoder_b[-1] * 2) ** 3)
        self.mlp = MLP(H=self.mlp_h, end_activation=True)

        self.repar1 = Nreparametrize(mlp_h[-1], z_dim)
        self.reparametrize = [self.repar1]
        self.r_callback = [lambda x: self.mlp(x.view(-1, self.mlp_h[0]))]

        self.bce = nn.BCELoss(size_average=False)

    def recon_loss(self, x, x_recon):
        x = x.expand_as(x_recon)
        # x_recon = F.sigmoid(x_recon)
        # b = x_recon.log() * x + (1 - x_recon).log() * (1 - x)
        b = F.binary_cross_entropy_with_logits(x_recon, x) * (x_recon.size()[-1] ** 2)
        return b
