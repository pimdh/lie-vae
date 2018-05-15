"""
Supervised test setup.

For a single type of object and in many rotations, we try to reconstruct the
object from the given rotation. This is done so that we can later use the
decoder for the VAE.

We do this by learning a single Fourier spectrum of the spherical harmonics of
a signal on the sphere. This spectrum is spare, we just pick some of the
harmonics to be non-zero and construct the appropriate Wigner D matrices. We
transform the spectrum by matrix multiplication. Subsequently we map the
spectrum to 2D pixel space with the visualisation network.

The visualisation network and the original spectrum vector are learned through
MSE reconstruction loss.

The input data is stored as quaternions, but we convert them to rotation matrices.

Uses https://github.com/AMLab-Amsterdam/lie_learn
"""
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import os.path
from pprint import pprint
from tensorboardX import SummaryWriter
import argparse

from lie_vae.datasets import ShapeDataset, SelectedDataset
from lie_vae.decoders import ActionNetWrapper as ActionNet, MLPNet
from lie_vae.nets import ChairsEncoder, ChairsDeconvNet, ChairsDeconvNetUpsample
from lie_vae.utils import random_split, encode

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def test(loader, decoder, encoder=None, single_id=True):
    decoder.eval()
    if encoder:
        encoder.eval()
    losses = []
    for it, (item_label, rot_label, img_label) in enumerate(loader):
        rot_label, img_label = rot_label.to(device), img_label.to(device)

        rot, content_data = encode(encoder, single_id, item_label, rot_label, img_label)
        reconstruction = decoder(rot, content_data)
        loss = nn.MSELoss()(reconstruction, img_label)
        losses.append(loss.item())
    return np.mean(losses)


def train(epoch, train_loader, test_loader, decoder, optimizer, log, encoder=None,
          report_freq=1250, clip_grads=None, single_id=True):
    losses = []
    for it, (item_label, rot_label, img_label) in enumerate(train_loader):
        decoder.train()
        if encoder:
            encoder.train()

        rot_label, img_label = rot_label.to(device), img_label.to(device)

        rot, content_data = encode(encoder, single_id, item_label, rot_label, img_label)
        reconstruction = decoder(rot, content_data)

        loss = nn.MSELoss()(reconstruction, img_label)

        optimizer.zero_grad()
        loss.backward()
        if clip_grads and encoder is not None:
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip_grads)
        optimizer.step()

        losses.append(loss.item())

        if (it + 1) % report_freq == 0 or it + 1 == len(train_loader):
            train_loss = np.mean(losses[-report_freq:])
            global_it = epoch * len(train_loader) + it + 1
            log.add_scalar('train_loss', train_loss, global_it)

            test_loss = test(test_loader, decoder, encoder, single_id)
            log.add_scalar('test_loss', test_loss, global_it)
            print(global_it, train_loss, test_loss)


def main():
    args = parse_args()
    pprint(vars(args))
    log = SummaryWriter(args.log_dir)

    matrix_dims = (args.degrees + 1) ** 2
    if args.deconv == 'upsample':
        deconv = ChairsDeconvNetUpsample(matrix_dims * args.rep_copies, args.deconv_hidden)
    elif args.deconv == 'deconv':
        deconv = ChairsDeconvNet(matrix_dims * args.rep_copies, args.deconv_hidden)
    else:
        raise RuntimeError()

    if args.mode == 'action':
        net = ActionNet(args.degrees,
                        deconv=deconv,
                        content_dims=args.content_dims,
                        rep_copies=args.rep_copies,
                        harmonics_encoder_layers=args.harmonics_encoder_layers,
                        single_id=args.single_id).to(device)
    elif args.mode == 'mlp':
        net = MLPNet(args.degrees,
                     deconv=deconv,
                     content_dims=args.content_dims,
                     rep_copies=args.rep_copies,
                     single_id=args.single_id).to(device)
    else:
        raise RuntimeError('Mode {} not found'.format(args.mode))

    if args.ae:
        content_dims = args.content_dims if not args.single_id else 0
        encoder = ChairsEncoder(content_dims).to(device)
    else:
        encoder = None

    if args.continue_epoch > 0:
        net.load_state_dict(torch.load(os.path.join(
            args.save_dir, 'dec.pickle')))
        if encoder is not None:
            encoder.load_state_dict(torch.load(os.path.join(
                args.save_dir, 'enc.pickle')))

    if args.single_id:
        dataset = ShapeDataset('data/chairs/single')
    else:
        dataset = SelectedDataset()

    if not len(dataset):
        raise RuntimeError('Dataset empty')

    num_test = min(int(len(dataset) * 0.2), 5000)
    split = [len(dataset)-num_test, num_test]
    train_dataset, test_dataset = random_split(dataset, split)
    train_loader = DataLoader(train_dataset, batch_size=64,
                              shuffle=True, num_workers=5)
    test_loader = DataLoader(test_dataset, batch_size=64,
                             shuffle=True, num_workers=5)
    params = list(net.parameters())
    if encoder:
        params = params + list(encoder.parameters())
    optimizer = torch.optim.Adam(params)

    for epoch in range(args.continue_epoch, args.epochs):
        train(epoch, train_loader, test_loader, net, optimizer, log, encoder,
              single_id=args.single_id,
              report_freq=args.report_freq, clip_grads=args.clip_grads)
        if args.save_dir:
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            torch.save(net.state_dict(), os.path.join(
                args.save_dir, 'dec.pickle'))
            if encoder is not None:
                torch.save(encoder.state_dict(), os.path.join(
                    args.save_dir, 'enc.pickle'))
    log.close()


def parse_args():
    parser = argparse.ArgumentParser('Supervised experiment')
    parser.add_argument('--ae', type=int, default=0,
                        help='whether to auto-encode')
    parser.add_argument('--mode', required=True,
                        help='[action, mlp]')
    parser.add_argument('--deconv', default='deconv',
                        help='Deconv mode [deconv, upsample]')
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
    parser.add_argument('--single_id', type=int, default=1)
    parser.add_argument('--harmonics_encoder_layers', type=int, default=3)
    parser.add_argument('--log_dir')
    parser.add_argument('--save_dir')
    parser.add_argument('--continue_epoch', type=int, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    main()
