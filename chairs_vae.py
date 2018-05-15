import torch
import numpy as np
from torch.utils.data import DataLoader
import os.path
from pprint import pprint
from tensorboardX import SummaryWriter
import argparse

from lie_vae.datasets import SelectedDataset
from lie_vae.vae import ChairsVAE
from lie_vae.utils import random_split

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def test(loader, model):
    model.eval()
    losses = []
    for it, (item_label, rot_label, img_label) in enumerate(loader):
        img_label = img_label.to(device)
        recon, kl = model.elbo(img_label)
        losses.append((recon.mean().item(), kl.mean().item()))
    return np.mean(losses, 0)


def train(epoch, train_loader, test_loader, model, optimizer, log,
          report_freq=1250, clip_grads=None, beta=1.0):
    losses = []
    for it, (item_label, rot_label, img_label) in enumerate(train_loader):
        model.train()
        img_label = img_label.to(device)
        recon, kl = model.elbo(img_label)

        loss = (recon + beta * kl).mean()

        if torch.isnan(kl).sum():
            raise RuntimeError("NaN KL")

        optimizer.zero_grad()
        loss.backward()
        if clip_grads:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grads)
        optimizer.step()

        losses.append((recon.mean().item(), kl.mean().item()))

        if (it + 1) % report_freq == 0 or it + 1 == len(train_loader):
            train_recon, train_kl = np.mean(losses[-report_freq:], 0)
            global_it = epoch * len(train_loader) + it + 1
            log.add_scalar('train_loss', train_recon + beta * train_kl, global_it)
            log.add_scalar('train_recon', train_recon, global_it)
            log.add_scalar('train_kl', train_kl, global_it)

            test_recon, test_kl = test(test_loader, model)
            log.add_scalar('test_loss', test_recon + beta * test_kl, global_it)
            log.add_scalar('test_recon', test_recon, global_it)
            log.add_scalar('test_kl', test_kl, global_it)
            print('Epoch {} it {} train recon {:.4f} kl {:.4f} test recon {:.4f} kl {:.4f}'
                  .format(epoch, it+1, train_recon, train_kl, test_recon, test_kl))


def main():
    args = parse_args()
    pprint(vars(args))
    log = SummaryWriter(args.log_dir)

    model = ChairsVAE(
        content_dims=args.content_dims,
        latent_mode=args.latent_mode,
        decoder_mode=args.decoder_mode,
        deconv_mode=args.deconv_mode,
        rep_copies=args.rep_copies,
        degrees=args.degrees,
        deconv_hidden=args.deconv_hidden,
    ).to(device)

    if args.continue_epoch > 0:
        model.load_state_dict(torch.load(os.path.join(
            args.save_dir, 'model.pickle')))

    dataset = SelectedDataset()
    if not len(dataset):
        raise RuntimeError('Dataset empty')

    num_test = min(int(len(dataset) * 0.2), 5000)
    split = [len(dataset)-num_test, num_test]
    train_dataset, test_dataset = random_split(dataset, split)
    train_loader = DataLoader(train_dataset, batch_size=64,
                              shuffle=True, num_workers=10)
    test_loader = DataLoader(test_dataset, batch_size=64,
                             shuffle=True, num_workers=10)

    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(args.continue_epoch, args.epochs):
        train(epoch, train_loader, test_loader, model, optimizer, log,
              report_freq=args.report_freq, clip_grads=args.clip_grads,
              beta=args.beta)
        if args.save_dir:
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            torch.save(model.state_dict(), os.path.join(
                args.save_dir, 'model.pickle'))
    log.close()


def parse_args():
    parser = argparse.ArgumentParser('Supervised experiment')
    parser.add_argument('--decoder_mode', required=True,
                        help='[action, mlp]')
    parser.add_argument('--latent_mode', required=True,
                        help='[so3, normal]')
    parser.add_argument('--deconv_mode', default='deconv',
                        help='Deconv mode [deconv, upsample]')
    parser.add_argument('--beta', type=float, default=1.)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--report_freq', type=int, default=1250)
    parser.add_argument('--degrees', type=int, default=3)
    parser.add_argument('--deconv_hidden', type=int, default=50)
    parser.add_argument('--content_dims', type=int, default=10,
                        help='The dims of the content latent code')
    parser.add_argument('--rep_copies', type=int, default=10,
                        help='The dims of the virtual signal on the Sphere, '
                             'i.e. the number of copies of the representation.')
    parser.add_argument('--clip_grads', type=float, default=1E-5)
    parser.add_argument('--log_dir')
    parser.add_argument('--save_dir')
    parser.add_argument('--continue_epoch', type=int, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    main()
