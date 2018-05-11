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

Throughout we use ZXZ Euler angles to represent the group element. We use:
https://en.wikipedia.org/wiki/Wigner_D-matrix

Uses https://github.com/AMLab-Amsterdam/lie_learn
"""
import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import block_diag
from functools import partial
from glob import glob
from torch.utils.data import Dataset, DataLoader
import os.path
from PIL import Image
from lie_learn.groups.SO3 import change_coordinates as SO3_coordinates
from lie_learn.representations.SO3.wigner_d import wigner_D_matrix
from tensorboardX import SummaryWriter
import argparse
from utils import MLP, random_split


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class ShapeDataset(Dataset):
    def __init__(self, directory, transformer_fn=None):
        self.directory = directory
        self.files = glob(os.path.join(directory, '**/*.jpg'))
        self.transformer = transformer_fn

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        image = Image.open(filename)
        image_tensor = torch.tensor(np.array(image), dtype=torch.float32) / 255

        # Remove extension, then retrieve _ separated floats
        quaternion = [float(x)
                      for x in filename[:-len('.jpg')].split('_')[-4:]]

        # Make gray scale
        image_tensor = image_tensor.mean(-1)

        group_el = quaternion

        if self.transformer:
            group_el = self.transformer(group_el)

        return group_el, image_tensor


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
        self.in_dims = in_dims
        self.matrix_dims = int(np.sum(np.arange(degrees+1) * 2 + 1))
        self.item_rep = nn.Parameter(torch.randn((self.matrix_dims, in_dims)))
        self.deconv = DeconvNet(self.matrix_dims * self.in_dims, 50)
        if with_mlp:
            self.mlp = MLP(self.matrix_dims * in_dims,
                           self.matrix_dims * in_dims, 50, 3)
        else:
            self.mlp = None

    def forward(self, matrices):
        """Input dim is [batch, matrix_dims, matrix_dims]."""
        item = torch.einsum('bij,jd->bid', (matrices, self.item_rep)) \
            .view(-1, self.matrix_dims * self.in_dims)

        if self.mlp:
            item = self.mlp(item)
        out = self.deconv(item[:, :, None, None])
        return out[:, 0, :, :]


class AngleMLPNet(nn.Module):
    """Uses MLP from group angles."""
    def __init__(self, degrees, in_dims=10):
        super().__init__()
        matrix_dims = int(np.sum(np.arange(degrees+1) * 2 + 1))
        self.mlp = MLP(3, matrix_dims * in_dims, 50, 3)
        self.deconv = DeconvNet(matrix_dims * in_dims, 50)

    def forward(self, angles):
        return self.deconv(self.mlp(angles)[:, :, None, None])[:, 0, :, :]


def block_wigner_d(max_degrees, angles):
    """Create block diagonal of multiple wigner d functions."""
    degree_matrices = [wigner_D_matrix(degree, *angles)
                       for degree in range(max_degrees+1)]

    block_form = block_diag(*degree_matrices)
    return torch.tensor(block_form, dtype=torch.float32)


def wigner_transformer(max_degrees, quaternion):
    """Map to ZYZ angles and then to blocked Wigned D matrices."""
    # To ZYZ euler angles
    angles = SO3_coordinates(quaternion, 'Q', 'EA323')
    return block_wigner_d(max_degrees, angles)


def angle_transformer(quaternion):
    """Map to ZYZ Euler angles."""
    angles = SO3_coordinates(quaternion, 'Q', 'EA323')
    return torch.tensor(angles, dtype=torch.float32)


def test(loader, net):
    losses = []
    for it, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        reconstruction = net(x)
        loss = nn.MSELoss()(reconstruction, y)
        losses.append(loss.item())
    return np.mean(losses)


def train(epoch, train_loader, test_loader, net, optimizer, log):
    losses = []
    for it, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        reconstruction = net(x)

        loss = nn.MSELoss()(reconstruction, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (it + 1) % 500 == 0 or it + 1 == len(train_loader):
            train_loss = np.mean(losses[-500:])
            global_it = epoch * len(train_loader) + it + 1
            log.add_scalar('train_loss', train_loss, global_it)
            test_loss = test(test_loader, net)
            log.add_scalar('test_loss', test_loss, global_it)
            print(it+1, train_loss, test_loss)


def main():
    args = parse_args()
    degrees = 3

    log = SummaryWriter(args.log_dir)

    if args.mode == 'action':
        transformer_fn = partial(wigner_transformer, degrees)
        net = ActionNet(degrees).to(device)
    elif args.mode == 'mlp':
        transformer_fn = angle_transformer
        net = AngleMLPNet(degrees).to(device)
    else:
        raise RuntimeError('Mode {} not found'.format(args.mode))

    dataset = ShapeDataset('./shapes', transformer_fn=transformer_fn)
    num_train = int(len(dataset) * 0.8)
    split = [num_train, len(dataset)-num_train]
    train_dataset, test_dataset = random_split(dataset, split)
    train_loader = DataLoader(train_dataset, batch_size=64,
                              shuffle=True, num_workers=5)
    test_loader = DataLoader(test_dataset, batch_size=64,
                             shuffle=True, num_workers=5)
    optimizer = torch.optim.Adam(net.parameters())

    for epoch in range(10):
        train(epoch, train_loader, test_loader, net, optimizer, log)

    log.close()


def parse_args():
    parser = argparse.ArgumentParser('Supervised experiment')
    parser.add_argument('--mode', required=True,
                        help='[action, mlp]')
    parser.add_argument('--log_dir', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    main()
