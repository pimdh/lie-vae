
# coding: utf-8

# In[ ]:


import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
import numpy as np
import matplotlib.pyplot as plt

from vae import VAE
from reparameterize import Nreparameterize, SO3reparameterize, N0reparameterize

from torch.utils.data.dataset import Dataset


# In[ ]:


class View(nn.Module):
    def __init__(self, *v):
        super(View, self).__init__()
        self.v = v
    
    def forward(self, x):
        return x.view(*self.v)
    
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    
    def forward(self, x):
        return x.view(x.size(0), -1)


# In[ ]:


class Decoder(nn.Module):
    
    def __init__(self):
        super(Decoder, self).__init__()
        
        self.__decoder_f = nn.Sequential(
            nn.Linear(9, 100),
            nn.ReLU(),
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 32 * 8 * 8),
            View(-1, 32, 8, 8),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReLU(),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        h = self.__decoder_f(z)
        return h.view(z.size()[0], z.size()[1], 3, 32, 32)

class ConvVAE(VAE):
    def __init__(self):
        super(ConvVAE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            Flatten(),
            nn.Linear(32 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 100),
            nn.ReLU()
        )

#         self.rep0 = Nreparameterize(100, z_dim=3)
        self.rep0 = N0reparameterize(100, z_dim=3)
        self.rep1 = SO3reparameterize(self.rep0)
        
        self.r_callback = [self.useless_f]
        
        self.reparameterize = [self.rep1] # [self.rep0]

        self.decoder = Decoder()
    
    def useless_f(self, x):
        return x
    
    def recon_loss(self, x_recon, x):
        return ((x_recon - x) ** 2).sum(-1).sum(-1).sum(-1)


# In[ ]:


model = ConvVAE()


# In[ ]:


# dir_ = 'imgs_jpg'
# images = np.array([plt.imread(os.path.join(dir_, filename)) for filename in os.listdir(dir_)])
# labels = np.load('cube-1.13.3.npy')
# np.save('images.npy', images)
# np.save('labels.npy', labels)

images = np.load('cubes_images.npy') / 255
labels = np.load('cubes_labels.npy') / 255

train_data = torch.from_numpy(images.transpose(0, 3, 1, 2).astype(np.float32))
train_labels = torch.from_numpy(labels.astype(np.int64))

batch_size = 32
train_dataset = data_utils.TensorDataset(train_data, train_labels)
train_loader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# In[ ]:


# decoder = nn.Sequential(
#     nn.Linear(100, 256),
#     View(4, 8, 8),
#     nn.Upsample(scale_factor=2, mode='nearest'),
#     nn.ReLU(),
#     nn.Conv2d(4, 8, 3, padding=1),
#     nn.Upsample(scale_factor=2, mode='nearest'),
#     nn.ReLU(),
#     nn.Conv2d(8, 3, 3, padding=1),
#     nn.Sigmoid()
# )


# In[ ]:


# images, labels = next(iter(train_loader))
# images = Variable(images)


# In[ ]:


# torch.save(model, 'filename.pt')
# model = torch.load('filename.pt')


# In[ ]:


optimizer = torch.optim.Adam(model.parameters())


# In[ ]:


for j in range(10):
    print(j)
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images)

        optimizer.zero_grad()
        
        recon, kl = model.elbo(images)
        loss = (recon + kl).mean()
        
        loss.backward()
        optimizer.step()
        print('\r', i, '/', len(train_loader), ':', loss.data.cpu().numpy()[0], '-', 
              recon.mean().data.cpu().numpy()[0], end='')

    torch.save(model, 'so3.pt')
    print(model.log_likelihood(Variable(train_data[:100]), n=5).data.cpu().numpy()[0])
    print()


# In[ ]:


# optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)


# In[ ]:

