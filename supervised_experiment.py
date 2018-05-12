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
from glob import glob
from torch.utils.data import Dataset, DataLoader
import os.path
from PIL import Image
from lie_learn.groups.SO3 import change_coordinates as SO3_coordinates
from tensorboardX import SummaryWriter
import argparse
from utils import MLP, random_split
from lie_tools import group_matrix_to_eazyz, block_wigner_matrix_multiply, \
    rodrigues


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class ShapeDataset(Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.files = glob(os.path.join(directory, '**/*.jpg'))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        image = Image.open(filename)
        image_tensor = torch.tensor(np.array(image), dtype=torch.float32) / 255
        quaternion = self.filename_to_quaternion(filename)
        image_tensor = image_tensor.mean(-1)

        group_el = torch.tensor(SO3_coordinates(quaternion, 'Q', 'MAT'),
                                dtype=torch.float32)
        return group_el, image_tensor

    @staticmethod
    def filename_to_quaternion(filename):
        """Remove extension, then retrieve _ separated floats"""
        return [float(x) for x in filename[:-len('.jpg')].split('_')[-4:]]


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        input_nc = 1
        ndf = 50
        nout = 3
        self.conv = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(input_nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, nout, 4, 1, 0, bias=False),
        )

    def forward(self, img):
        x = self.conv(img[:, None, :, :])
        return rodrigues(x[:, :, 0, 0])


class DeconvNet(nn.Sequential):
    """1x1 to 64x64 deconvolutional stack."""
    def __init__(self, in_dims, hidden_dims):
        super().__init__(
            nn.ConvTranspose2d(in_dims, hidden_dims, 4, 1, 0),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dims, hidden_dims, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dims, hidden_dims, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dims, hidden_dims, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dims, 1, 4, 2, 1),
        )


class ActionNet(nn.Module):
    """Uses proper group action."""
    def __init__(self, degrees, in_dims=10, with_mlp=False):
        super().__init__()
        self.degrees = degrees
        self.in_dims = in_dims
        self.matrix_dims = (degrees + 1) ** 2
        self.item_rep = nn.Parameter(torch.randn((self.matrix_dims, in_dims)))
        self.deconv = DeconvNet(self.matrix_dims * self.in_dims, 50)
        if with_mlp:
            self.mlp = MLP(self.matrix_dims * in_dims,
                           self.matrix_dims * in_dims, 50, 3)
        else:
            self.mlp = None

    def forward(self, matrices):
        """Input dim is [batch, 3, 3]."""
        n = matrices.size(0)
        item_expanded = self.item_rep.expand(n, -1, -1)

        angles = group_matrix_to_eazyz(matrices)
        item = block_wigner_matrix_multiply(angles, item_expanded, self.degrees) \
            .view(-1, self.matrix_dims * self.in_dims)

        if self.mlp:
            item = self.mlp(item)
        out = self.deconv(item[:, :, None, None])
        return out[:, 0, :, :]


class MLPNet(nn.Module):
    """Uses MLP from group matrix."""
    def __init__(self, degrees, in_dims=10):
        super().__init__()
        matrix_dims = (degrees + 1) ** 2
        self.mlp = MLP(3 * 3, matrix_dims * in_dims, 50, 3)
        self.deconv = DeconvNet(matrix_dims * in_dims, 50)

    def forward(self, x):
        """Input dim is [batch, 3, 3]."""
        x = self.mlp(x.view(-1, 9))
        return self.deconv(x[:, :, None, None])[:, 0, :, :]


def test(loader, net, encoder=None):
    net.eval()
    if encoder:
        encoder.eval()
    losses = []
    for it, (rot_label, img_label) in enumerate(loader):
        rot_label, img_label = rot_label.to(device), img_label.to(device)

        if encoder:
            rot = encoder(img_label)
        else:
            rot = rot_label
        reconstruction = net(rot)
        loss = nn.MSELoss()(reconstruction, img_label)
        losses.append(loss.item())
    return np.mean(losses)


def train(epoch, train_loader, test_loader, net, optimizer, log, encoder=None,
          report_freq=1250):
    losses = []
    for it, (rot_label, img_label) in enumerate(train_loader):
        net.train()
        rot_label, img_label = rot_label.to(device), img_label.to(device)
        if encoder:
            encoder.train()
            rot = encoder(img_label)
        else:
            rot = rot_label
        reconstruction = net(rot)

        loss = nn.MSELoss()(reconstruction, img_label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (it + 1) % report_freq == 0 or it + 1 == len(train_loader):
            train_loss = np.mean(losses[-report_freq:])
            global_it = epoch * len(train_loader) + it + 1
            log.add_scalar('train_loss', train_loss, global_it)
            test_loss = test(test_loader, net, encoder)
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
    log = SummaryWriter(args.log_dir)

    if args.mode == 'action':
        net = ActionNet(args.degrees).to(device)
    elif args.mode == 'mlp':
        net = MLPNet(args.degrees).to(device)
    else:
        raise RuntimeError('Mode {} not found'.format(args.mode))

    if args.ae:
        encoder = Encoder().to(device)
    else:
        encoder = None

    # Demo image
    filename = './shapes/assets/chair.obj_0.0336_-0.1523_-0.5616_-0.8126.jpg'
    q = ShapeDataset.filename_to_quaternion(filename)
    x_demo = torch.tensor(SO3_coordinates(q, 'Q', 'MAT'), dtype=torch.float32)

    dataset = ShapeDataset('./shapes')
    num_train = int(len(dataset) * 0.8)
    split = [num_train, len(dataset)-num_train]
    train_dataset, test_dataset = random_split(dataset, split)
    train_loader = DataLoader(train_dataset, batch_size=64,
                              shuffle=True, num_workers=5)
    test_loader = DataLoader(test_dataset, batch_size=64,
                             shuffle=True, num_workers=5)
    params = list(net.parameters())
    if encoder:
        params = params + list(encoder.parameters())
    optimizer = torch.optim.Adam(params)

    for epoch in range(args.num_its):
        train(epoch, train_loader, test_loader, net, optimizer, log, encoder,
              args.report_freq)
        if args.save_dir:
            generate_image(x_demo, net, os.path.join(
                args.save_dir, '{}_{}.jpg'.format(args.mode, epoch+1)))
            torch.save(net.state_dict(), os.path.join(
                args.save_dir, '{}_{}.pickle'.format(args.mode, epoch+1)))
            torch.save(encoder.state_dict(), os.path.join(
                args.save_dir, '{}_{}_enc.pickle'.format(args.mode, epoch+1)))

    log.close()

def parse_args():
    parser = argparse.ArgumentParser('Supervised experiment')
    parser.add_argument('--ae', type=int, default=0,
                        help='whether to auto-encode')
    parser.add_argument('--mode', required=True,
                        help='[action, mlp]')
    parser.add_argument('--num_its', type=int, default=10)
    parser.add_argument('--report_freq', type=int, default=1250)
    parser.add_argument('--degrees', type=int, default=3)
    parser.add_argument('--log_dir')
    parser.add_argument('--save_dir')
    return parser.parse_args()


if __name__ == '__main__':
    main()
