
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from SNets import S2ConvNet, S2DeconvNet
from pytorch_util import MLP
from reparameterize import Nreparameterize
from reparameterize import SO3reparameterize
from pytorch_util import logsumexp

from s2cnn.nn.soft.so3_integrate import so3_integrate

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = None
        self.decoder = None
        self.reparameterize = []
        self.r_callback = []

    def encode(self, x, n=1):
        h = self.encoder(x)
        z = [r(f(h), n) for r, f in zip(self.reparameterize, self.r_callback)]

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

        # flatten and stack z
        z_cat = torch.cat([v.view(n, x.size()[0], -1) for v in z], -1)

        return self.decode(z_cat)

    def recon_loss(self, x, x_recon):
        raise NotImplemented

    def elbo(self, x, n=1):
        x_recon = self.forward(x, n)
        kl = self.kl()
        # TODO maybe sum directly  without stacking 
        kl_summed = torch.sum(torch.stack(kl, -1), -1)
        recon_loss = self.recon_loss(x_recon, x)
        return recon_loss, kl_summed

    def log_likelihood(self, x, n=1):
        raise NotImplemented


class equivariant_callback(nn.Module):
    def __init__(self, specs):
        super(equivariant_callback)
    
class NS2VAE(VAE):
    def __init__(self, z_dim=10,
                 encoder_f=[1, 10, 10],
                 decoder_f=[10, 10, 1],
                 encoder_b=[30, 20, 6],
                 decoder_b=[5, 15, 30],
                 encoder_mlp_h=[100],
                 decoder_mlp_h=[100]):
        super(NS2VAE, self).__init__()

        self.encoder = S2ConvNet(f_list=encoder_f, b_list=encoder_b)
        self.decoder_mlp_h = decoder_mlp_h.copy()
        self.decoder_mlp_h.insert(0, z_dim)
        self.decoder = S2DeconvNet(f_list=decoder_f, b_list=decoder_b, mlp_dim=self.decoder_mlp_h)

        # output of the encoder base
        encoder_final_dim = encoder_f[-1] * (encoder_b[-1] * 2) ** 3
        
        self.encoder_mlp_h = encoder_mlp_h.copy()
        self.encoder_mlp_h.insert(0, encoder_final_dim)
        self.encoder_mlp = MLP(H=self.encoder_mlp_h, end_activation=True)

#         self.repar1 = Nreparameterize(encoder_mlp_h[-1], z_dim)
        self.repar2 = SO3reparameterize(encoder_mlp_h[-1], z_dim)
        self.reparameterize = [self.repar2]
        
        callback1 = (lambda x: MLP(so3_integrate(x), H=self.encoder_mlp_h, end_activation=True))
        
        self.r_callback = [lambda x: self.encoder_mlp(x.view(-1, self.encoder_mlp_h[0]))]

        self.bce = nn.BCELoss(size_average=False)
        
        
    def recon_loss(self, x_recon, x):
        x = x.expand_as(x_recon)
        max_val = (-x_recon).clamp(min=0)
        loss = x_recon - x_recon * x + max_val + ((-max_val).exp() + \
                                               (-x_recon - max_val).exp()).log()
        
        return loss.sum(-1).sum(-1).sum(-1)
    
    
    def log_likelihood(self, x, n=1):
        x_recon = self.forward(x, n)
  
        log_p_z = torch.cat([r.log_prior() for r in self.reparameterize], -1).sum(-1)
        log_q_z_x = torch.cat([r.log_posterior() for r in self.reparameterize], -1).sum(-1)
        log_p_x_z = - self.recon_loss(x_recon, x)
        
        return (logsumexp(log_p_x_z + log_p_z - log_q_z_x, dim=0) - np.log(n)).mean()
