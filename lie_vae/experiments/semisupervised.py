import numpy as np
import torch
from torch.utils.data import DataLoader
from ..utils import random_split, cycle

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class SemiSupervisedExperiment:
    """Experiment for semi-supervised pose learning.

    Some of the batch samples will be supervised. In addidion to KL and
    reconstruction loss, it will impose a supervised loss on the encoding.
    This loss is the Frobenius norm of the rotation matrices.
    """
    def __init__(self, *, log, model, optimizer, beta_schedule,
                 train_dataset, test_dataset, num_labelled,
                 lambda_supervised=1.0, elbo_samples=1,
                 report_freq=1250, clip_grads=None, selective_clip=False,
                 batch_size=64, labelled_batch_size=1):
        self.log = log
        self.model = model
        self.optimizer = optimizer
        self.beta_schedule = beta_schedule

        split = [len(train_dataset) - num_labelled, num_labelled]
        unlabelled_dataset, labelled_dataset = \
            random_split(train_dataset, split)

        self.unlabelled_loader = DataLoader(
            unlabelled_dataset, batch_size=batch_size-labelled_batch_size,
            shuffle=True, num_workers=5)

        # This will have many fewer samples, so we cycle the iterable
        self.labelled_loader = cycle(DataLoader(
            labelled_dataset, batch_size=labelled_batch_size,
            shuffle=True, num_workers=5))

        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True, num_workers=5)

        self.lambda_supervised = lambda_supervised
        self.elbo_samples = elbo_samples
        self.clip_grads = clip_grads
        self.selective_clip = selective_clip
        self.report_freq = report_freq

    def test(self):
        self.model.eval()
        losses = []
        for it, (item_label, rot_label, img_label) in \
                enumerate(self.test_loader):
            img_label = img_label.to(device)
            recon, kl, kls = self.model.elbo(img_label, n=self.elbo_samples)
            losses.append((recon.mean().item(), kl.mean().item(),
                           *[x.mean().item() for x in kls]))
        return np.mean(losses, 0)

    def train(self, epoch):
        losses = []
        iterable = enumerate(zip(self.unlabelled_loader, self.labelled_loader))
        for it, (ul_batch, l_batch) in iterable:
            ul_img = ul_batch[2].to(device)
            l_rot, l_img = l_batch[1].to(device), l_batch[2].to(device)
            img = torch.cat((l_img, ul_img), 0)

            global_it = epoch * len(self.unlabelled_loader) + it + 1
            self.model.train()

            # Unsupervised
            recon, kl, kls = self.model.elbo(img, n=self.elbo_samples)
            beta = self.beta_schedule(global_it)
            loss = (recon + beta * kl).mean()

            if torch.isnan(kl).sum():
                raise RuntimeError("NaN KL")

            # Supervised, Frobenius norm on rotation matrices
            encoding = self.model.encode(l_img)[0][0]
            sup_loss = (encoding - l_rot).pow(2).sum(-1).sum(-1).mean()

            loss = loss + sup_loss * self.lambda_supervised

            self.optimizer.zero_grad()
            loss.backward()
            if self.clip_grads:
                if self.selective_clip:
                    params = list(self.model.encoder.parameters()) \
                             + list(self.model.rep_group.parameters())
                else:
                    params = self.model.parameters()
                torch.nn.utils.clip_grad_norm_(params, self.clip_grads)
            self.optimizer.step()

            losses.append((recon.mean().item(), kl.mean().item(),
                           sup_loss.item(), *[x.mean().item() for x in kls]))

            if (it + 1) % self.report_freq == 0 or \
                    it + 1 == len(self.unlabelled_loader):
                train_recon, train_kl, train_sup, *train_kls = \
                    np.mean(losses[-self.report_freq:], 0)
                self.log.add_scalar('train_loss',
                                    train_recon + beta * train_kl + train_sup,
                                    global_it)
                self.log.add_scalar('train_recon', train_recon, global_it)
                self.log.add_scalar('train_kl', train_kl, global_it)
                self.log.add_scalar('train_sup', train_sup, global_it)
                self.log.add_scalars(
                    'train_kls',
                    {'kl%d' % i: x for i, x in enumerate(train_kls)},
                    global_it)

                test_recon, test_kl, *test_kls = self.test()
                self.log.add_scalar('test_loss', test_recon + beta * test_kl,
                                    global_it)
                self.log.add_scalar('test_recon', test_recon, global_it)
                self.log.add_scalar('test_kl', test_kl, global_it)
                self.log.add_scalars(
                    'test_kls',
                    {'kl%d' % i: x for i, x in enumerate(test_kls)},
                    global_it)

                self.log.add_scalar('beta', beta, global_it)
                print(('Epoch {} it {} train recon {:.4f} kl {:.4f}'
                       ' sup {:.4f} test recon {:.4f} kl {:.4f}')
                      .format(epoch, it+1, train_recon, train_kl,
                              train_sup, test_recon, test_kl))