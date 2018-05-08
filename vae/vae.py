
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from SNets import S2ConvNet, S2DeconvNet
from pytorch_util import MLP
from reparameterize import Nreparameterize
from reparameterize import SO3reparameterize
from pytorch_util import logsumexp

from s2cnn.nn.soft.so3_conv import SO3Convolution
from s2cnn.nn.soft.so3_integrate import so3_integrate
from s2cnn.ops.so3_localft import near_identity_grid as so3_near_identity_grid

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
  
        log_p_z = torch.cat([r.log_prior() for r in self.reparameterize], -1).sum(-1)
        log_q_z_x = torch.cat([r.log_posterior() for r in self.reparameterize], -1).sum(-1)
        log_p_x_z = - self.recon_loss(x_recon, x)
        
        return (logsumexp(log_p_x_z + log_p_z - log_q_z_x, dim=0) - np.log(n)).mean()


class equivariant_callback(nn.Module):
    def __init__(self, f_list, b_list, mlp_h, activation=nn.ReLU):
        super(equivariant_callback, self).__init__()
        
        grid_so3 = so3_near_identity_grid()
        
        modules = []
        for f_in, f_out, b_in, b_out in zip(f_list[0:-1], f_list[1:], b_list[0:-1], b_list[1:]):
            conv = SO3Convolution(nfeature_in=f_in,
                                  nfeature_out=f_out,
                                  b_in=b_in,
                                  b_out=b_out,
                                  grid=grid_so3)
            modules.append(conv)
            modules.append(activation())
            
        self.conv_module = nn.Sequential(*modules) 
        
        conv_final_dim = f_list[-1] * (b_list[-1] * 2) ** 3
        mlp_h.insert(0, conv_final_dim)
        self.mlp_module = MLP(H = mlp_h, activation=activation, end_activation=False)
        
    def forward(self, x):
        x = self.conv_module(x)
        x = x.view(x.size(0), -1)
        x = self.mlp_module(x)
        
        return x
    
class invariant_callback(nn.Module):
    def __init__(self, input_dim, mlp_h, activation=nn.ReLU):
        super(invariant_callback, self).__init__()
        
        # num features left after integration over the channels
        mlp_h.insert(0, input_dim)
        self.mlp_module = MLP(H = mlp_h, activation=activation, end_activation=False)
        
    def forward(self, x):
        x = so3_integrate(x)
        x = x.view(x.size(0), -1)
        x = self.mlp_module(x)
        
        return x
    
class NS2VAE(VAE):
    def __init__(self, z_dims=[10, 9],
                 encoder_f=[1, 10, 10],
                 decoder_f=[10, 10, 1],
                 encoder_b=[30, 20, 6],
                 decoder_b=[5, 15, 30],
                 latents = ['gaussian', 'gaussian'],
                 callbacks = [{'type':'invariant', 
                               'mlp_h': [100]},
                              {'type':'equivariant', 
                               'f_list':[1],
                               'b_list':[10], 
                               'mlp_h': [100]}],
                 decoder_mlp_h=[100],
                 max_pooling=True):
        super(NS2VAE, self).__init__()

        self.encoder = S2ConvNet(f_list=encoder_f, b_list=encoder_b)
#         # output of the encoder base
#         encoder_final_dim = encoder_f[-1] * (encoder_b[-1] * 2) ** 3
     
        self.reparameterize = []
        self.r_callback = []
        for i, (l, cb, z) in enumerate(zip(latents, callbacks, z_dims)):
            if cb['type'] == 'invariant':
                cb_module = invariant_callback(input_dim=encoder_f[-1],
                                               mlp_h=cb['mlp_h'])
            elif cb['type'] == 'equivariant':
                f_list = cb['f_list']
                f_list.insert(0, encoder_f[-1])
                b_list = cb['b_list']
                b_list.insert(0, encoder_b[-1])
                cb_module = equivariant_callback(f_list = f_list, 
                                                 b_list = b_list,
                                                 mlp_h = cb['mlp_h'])
            else:
                print ('!!! please specifcy callback')
                raise RuntimeError 
            
            self.add_module(('cb%d' % i), cb_module)
            self.r_callback.append(cb_module)
            
            if l == 'gaussian':
                reparam = Nreparameterize(cb['mlp_h'][-1], z)
            elif l == 'so3':
                assert z == 9 #the 3x3 lie group element will be concatenated 
                reparam = Nreparameterize(cb['mlp_h'][-1], 3)
                reparam = SO3reparameterize(reparam)
                
            else:
                print ('!!! please specify latent')
                raise RuntimeError 
                
            self.add_module(('latent%d' % i),reparam)
            self.reparameterize.append(reparam)

        z_dim_out = sum(z_dims)
        self.decoder_mlp_h = decoder_mlp_h.copy()
        self.decoder_mlp_h.insert(0, z_dim_out)
        self.decoder = S2DeconvNet(f_list=decoder_f, b_list=decoder_b, 
                                   mlp_dim=self.decoder_mlp_h, max_pooling=max_pooling)
        
        
    def recon_loss(self, x_recon, x):
        x = x.expand_as(x_recon)
        max_val = (-x_recon).clamp(min=0)
        loss = x_recon - x_recon * x + max_val + ((-max_val).exp() + \
                                               (-x_recon - max_val).exp()).log()
        
        return loss.sum(-1).sum(-1).sum(-1)
