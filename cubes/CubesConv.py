
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
import argparse

from vae import VAE
from reparameterize import Nreparameterize, SO3reparameterize, N0reparameterize

from torch.utils.data.dataset import Dataset

CUDA = torch.cuda.is_available()


# In[ ]:


parser = argparse.ArgumentParser()

parser.add_argument('--latent', '-z', type=str, default='normal', help='')
parser.add_argument('--epochs', '-e', type=int, default=10, help='')
parser.add_argument('--load', '-l', type=str, default='', help='')
parser.add_argument('--save', '-s', type=str, default='', help='')
parser.add_argument('--batch_dim', '-b', type=int, default=32, help='')

FLAGS, unparsed = parser.parse_known_args()


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

class Decoder(nn.Module):
    
    def __init__(self):
        super(Decoder, self).__init__()
        
        self.__decoder_f = nn.Sequential(
            nn.Linear(9 if FLAGS.latent == 'so3' else (3 if FLAGS.latent == 'normal' else None), 512),
            nn.ReLU(),
            nn.Linear(512, 32 * 8 * 8 ),
            View(-1, 32, 8, 8),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, 3, padding=1)
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
        
        self.rep0 = SO3reparameterize(self.N0reparameterize(100, z_dim=3), k=1000) if FLAGS.latent == 'so3' else(
            Nreparameterize(100, z_dim=3) if FLAGS.latent == 'normal' else None)
        
        self.r_callback = [self.useless_f]
        
        self.reparameterize = [self.rep0]

        self.decoder = Decoder()
    
    def useless_f(self, x):
        return x
    
    def recon_loss(self, x_recon, x):
        x = x.expand_as(x_recon)
        max_val = (-x_recon).clamp(min=0)
        loss = x_recon - x_recon * x + max_val + ((-max_val).exp() + (-x_recon - max_val).exp()).log()        
        return loss.sum(-1).sum(-1).sum(-1)


# In[ ]:


model = ConvVAE().cuda() if CUDA else ConvVAE()

if FLAGS.load != '':
    model = torch.load(FLAGS.load)
    
optimizer = torch.optim.Adam(model.parameters())


# In[ ]:


# dir_ = 'imgs_jpg'
# images = np.array([plt.imread(os.path.join(dir_, filename)) for filename in os.listdir(dir_)])
# labels = np.load('cube-1.13.3.npy')
# np.save('images.npy', images)
# np.save('labels.npy', labels)

# images = (np.load('cubes_images.npy') / 255).transpose(0, 3, 1, 2).astype(np.float32)
# labels = (np.load('cubes_labels.npy') / 255).astype(np.int64)

# idx = np.random.permutation(images.shape[0])
# train_data = images[idx[(images.shape[0] // 10) * 2:]]
# dev_data = images[idx[:images.shape[0] // 10]]
# test_data = images[idx[images.shape[0] // 10:(images.shape[0] // 10) * 2]]

# train_labels = labels[idx[(labels.shape[0] // 10) * 2:]]
# dev_labels = labels[idx[:labels.shape[0] // 10]]
# test_labels = labels[idx[labels.shape[0] // 10:(labels.shape[0] // 10) * 2]]

# np.save('train_data.npy', train_data)
# np.save('dev_data.npy', dev_data)
# np.save('test_data.npy', test_data)

# np.save('train_labels.npy', train_labels)
# np.save('dev_labels.npy', dev_labels)
# np.save('test_labels.npy', test_labels)


# In[ ]:


train_data = np.load('train_data.npy')
dev_data = np.load('dev_data.npy')
test_data = np.load('test_data.npy')

train_labels = np.load('train_labels.npy')
dev_labels = np.load('dev_labels.npy')
test_labels = np.load('test_labels.npy')

train_data = torch.from_numpy(train_data)
train_labels = torch.from_numpy(train_labels)

train_dataset = data_utils.TensorDataset(train_data, train_labels)
train_loader = data_utils.DataLoader(train_dataset, batch_size=FLAGS.batch_dim, shuffle=True)

dev_data = torch.from_numpy(dev_data)
dev_labels = torch.from_numpy(dev_labels)

dev_dataset = data_utils.TensorDataset(dev_data, dev_labels)
dev_loader = data_utils.DataLoader(dev_dataset, batch_size=FLAGS.batch_dim, shuffle=True)

test_data = torch.from_numpy(test_data)
test_labels = torch.from_numpy(test_labels)

test_dataset = data_utils.TensorDataset(test_data, test_labels)
test_loader = data_utils.DataLoader(test_dataset, batch_size=FLAGS.batch_dim, shuffle=True)


# In[ ]:


for j in range(FLAGS.epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images).cuda() if CUDA else Variable(images)

        optimizer.zero_grad()
        
        recon, kl = model.elbo(images)
        loss = (recon + kl).mean()
        
        loss.backward()
        optimizer.step()
        print('\r epoch: {:4}/{:4}, it: {:4}/{:4}: loss: {:.4f}, recon: {:.4f}, KL: {:.4f}'.format(
                j, FLAGS.epochs, i, len(train_loader),
                loss.data.cpu().numpy()[0],
                recon.mean().data.cpu().numpy()[0],
                kl.data.cpu().numpy()[0]),
            end='')

    if FLAGS.save != '':
        torch.save(model, FLAGS.save)
    
    images = Variable(dev_data).cuda() if CUDA else Variable(dev_data)
    
    recon, kl = model.elbo(images)
    loss = (recon + kl).mean()
    
    print('\r epoch: {:4}/{:4}, it: {:4}/{:4}: loss: {:.4f}, recon: {:.4f}, KL: {:.4f}, LL: {:.4f}'.format(
            j, FLAGS.epochs, i, len(train_loader),
            loss.data.cpu().numpy()[0],
            recon.mean().data.cpu().numpy()[0],
            kl.data.cpu().numpy()[0],
            model.log_likelihood(images, n=500).data.cpu().numpy()[0]))


# In[ ]:


images = Variable(test_data).cuda() if CUDA else Variable(test_data)

recon, kl = model.elbo(images)
loss = (recon + kl).mean()

print('TEST-> loss: {:.4f}, recon: {:.4f}, KL: {:.4f}, LL: {:.4f}'.format(
        loss.data.cpu().numpy()[0],
        recon.mean().data.cpu().numpy()[0],
        kl.data.cpu().numpy()[0],
        model.log_likelihood(images, n=5).data.cpu().numpy()[0]))


# In[ ]:


# optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)


# In[ ]:


# for j in range(10):
#     print(j)
#     for i, (images, labels) in enumerate(train_loader):
#         images = Variable(images)

#         optimizer.zero_grad()
#         recon = decoder(encoder(images))
#         loss = ((recon - images) ** 2).sum(-1).sum(-1).sum(-1).mean()
#         loss.backward()
#         optimizer.step()
#         print('\r', i, '/', len(train_loader), ':', loss.data.cpu().numpy()[0], end='')
#     print()


# In[ ]:


# img = Variable(next(iter(train_loader))[0][0:1])
# rec_img = decoder(encoder(img))

# plt.imshow(img.data.cpu().numpy()[0].transpose(1, 2, 0))
# plt.show()
# plt.imshow(rec_img.data.cpu().numpy()[0].transpose(1, 2, 0))
# plt.show()


# In[ ]:


# img = Variable(next(iter(train_loader))[0][0:1])
# rec_img = model(img)

# plt.imshow(img.data.cpu().numpy()[0].transpose(1, 2, 0))
# plt.show()
# plt.imshow(rec_img.data.cpu().numpy()[0].transpose(1, 2, 0))
# plt.show()

