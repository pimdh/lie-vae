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
from PIL import Image
from tensorboardX import SummaryWriter
import argparse

from lie_vae.datasets import ShapeDataset, SelectedDataset
from lie_vae.nets import ChairsEncoder, ChairsDeconvNet
from lie_vae.utils import MLP, random_split
from lie_vae.lie_tools import group_matrix_to_eazyz, block_wigner_matrix_multiply, \
    group_matrix_to_quaternions

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class ActionNet(nn.Module):
    """Uses proper group action."""
    def __init__(self, degrees, deconv, id_dims=10, data_dims=10,
                 single_id=True, harmonics_encoder_layers=3):
        super().__init__()
        self.degrees = degrees
        self.data_dims = data_dims
        self.matrix_dims = (degrees + 1) ** 2

        if single_id:
            self.item_rep = nn.Parameter(torch.randn((self.matrix_dims, data_dims)))
        else:
            self.item_rep = None
            self.harmonics_encoder = MLP(
                id_dims, self.matrix_dims * self.data_dims, 50, harmonics_encoder_layers)

        self.deconv = deconv

    def forward(self, matrices, id_data=None):
        """Input dim is [batch, 3, 3]."""
        assert (id_data is not None) != (self.item_rep is not None), \
            'Either must be single id or provide id_data, not both.'

        n = matrices.size(0)

        if id_data is None:
            harmonics = self.item_rep.expand(n, -1, -1)
        else:
            harmonics = self.harmonics_encoder(id_data)\
                .view(n, self.matrix_dims, self.data_dims)

        angles = group_matrix_to_eazyz(matrices)
        item = block_wigner_matrix_multiply(angles, harmonics, self.degrees) \
            .view(-1, self.matrix_dims * self.data_dims)

        return self.deconv(item)


class MLPNet(nn.Module):
    """Uses MLP from group matrix."""
    def __init__(self, degrees, deconv, data_dims=10, mode='MAT',
                 id_dims=10, single_id=True):
        super().__init__()
        matrix_dims = (degrees + 1) ** 2
        self.mode = mode
        dims = {'MAT': 9, 'Q': 4, 'EA': 3}[mode]

        if not single_id:
            dims += id_dims

        self.mlp = MLP(dims, matrix_dims * data_dims, 50, 3)
        self.deconv = deconv
        self.single_id = single_id

    def forward(self, r, id_data=None):
        """Input dim is [batch, 3, 3]."""
        assert (id_data is None) != (not self.single_id), \
            'Either must be single id or provide id_data, not both.'
        n = r.size(0)

        if self.mode == 'MAT':
            x = r.view(-1, 9)
        elif self.mode == 'Q':
            x = group_matrix_to_quaternions(r)
        else:
            x = group_matrix_to_eazyz(r)

        if id_data is not None:
            x = torch.cat((x, id_data.view(n, -1)), 1)

        return self.deconv(self.mlp(x))


def encode(encoder, single_id, item_label, rot_label, img_label):
    if encoder:
        rot, id_data = encoder(img_label)
    else:
        rot = rot_label
        if not single_id:
            id_data = torch.eye(10, device=device)[item_label]
        else:
            id_data = None
    return rot, id_data


def test(loader, decoder, encoder=None, single_id=True):
    decoder.eval()
    if encoder:
        encoder.eval()
    losses = []
    for it, (item_label, rot_label, img_label) in enumerate(loader):
        rot_label, img_label = rot_label.to(device), img_label.to(device)

        rot, id_data = encode(encoder, single_id, item_label, rot_label, img_label)
        reconstruction = decoder(rot, id_data)
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

        rot, id_data = encode(encoder, single_id, item_label, rot_label, img_label)
        reconstruction = decoder(rot, id_data)

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


def generate_image(x, net, path):
    """Render image for certain quaternion and write to path."""
    reconstruction = net(x.to(device)[None])[0]
    image_data = (reconstruction * 255).byte()
    image_array = image_data.detach().to('cpu').numpy()
    im = Image.fromarray(image_array)
    im.convert('RGB').save(path)


def main():
    args = parse_args()
    pprint(vars(args))
    log = SummaryWriter(args.log_dir)

    matrix_dims = (args.degrees + 1) ** 2
    deconv = ChairsDeconvNet(matrix_dims * args.data_dims, args.deconv_hidden)
    if args.mode == 'action':
        net = ActionNet(args.degrees,
                        deconv=deconv,
                        id_dims=args.id_dims,
                        data_dims=args.data_dims,
                        harmonics_encoder_layers=args.harmonics_encoder_layers,
                        single_id=args.single_id).to(device)
    elif args.mode == 'mlp':
        net = MLPNet(args.degrees,
                     deconv=deconv,
                     id_dims=args.id_dims,
                     data_dims=args.data_dims,
                     mode=args.mlp_mode,
                     single_id=args.single_id).to(device)
    else:
        raise RuntimeError('Mode {} not found'.format(args.mode))

    if args.ae:
        id_dims = args.id_dims if not args.single_id else 0
        encoder = ChairsEncoder(id_dims).to(device)
    else:
        encoder = None

    if args.continue_epoch > 0:
        net.load_state_dict(torch.load(os.path.join(
            args.save_dir, 'dec.pickle')))
        if encoder is not None:
            encoder.load_state_dict(torch.load(os.path.join(
                args.save_dir, 'enc.pickle')))

    # Demo image
    # filename = './data/chairs/single/assets/chair.obj_0.0336_-0.1523_-0.5616_-0.8126.jpg'
    # q = ShapeDataset.filename_to_quaternion(filename)
    # x_demo = torch.tensor(SO3_coordinates(q, 'Q', 'MAT'), dtype=torch.float32)

    if args.single_id:
        dataset = ShapeDataset('data/chairs/single')
    else:
        dataset = SelectedDataset()

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
            # generate_image(x_demo, net, os.path.join(
            #     args.save_dir, '{}_{}.jpg'.format(args.mode, epoch+1)))

    log.close()


def parse_args():
    parser = argparse.ArgumentParser('Supervised experiment')
    parser.add_argument('--ae', type=int, default=0,
                        help='whether to auto-encode')
    parser.add_argument('--mode', required=True,
                        help='[action, mlp]')
    parser.add_argument('--mlp_mode', help='[MAT, Q, EA]', default='MAT')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--report_freq', type=int, default=1250)
    parser.add_argument('--degrees', type=int, default=3)
    parser.add_argument('--deconv_hidden', type=int, default=50)
    parser.add_argument('--id_dims', type=int, default=10,
                        help='The dims of the content latent code')
    parser.add_argument('--data_dims', type=int, default=10,
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
