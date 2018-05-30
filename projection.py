import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread

import torch
from s2cnn import so3_rotation
import torch.nn.functional as F


def s2_rotation(x, a, b, c):
    x = so3_rotation(x.view(*x.size(), 1).expand(*x.size(), x.size(-1)), a, b, c)
    return x[..., 0]


def plot(x, text, normalize=False):
    assert x.size(0) == 1
    assert x.size(1) in [1, 3]
    x = x[0]
    if x.dim() == 4:
        x = x[..., 0]

    nch = x.size(0)
    is_rgb = (nch == 3)

    if normalize:
        x = x - x.view(nch, -1).mean(-1).view(nch, 1, 1)
        x = 0.4 * x / x.view(nch, -1).std(-1).view(nch, 1, 1)

    x = x.detach().cpu().numpy()
    x = x.transpose((1, 2, 0)).clip(0, 1)

    if is_rgb:
        plt.imshow(x)
    else:
        plt.imshow(x[:, :, 0], cmap='gray')
    plt.axis("off")

    plt.text(0.5, 0.5, text,
             horizontalalignment='center',
             verticalalignment='center',
             transform=plt.gca().transAxes,
             color='white', fontsize=20)


def orthographic_grid(n_x, n_y):
    xs = np.linspace(start=-1, stop=1, num=n_x, endpoint=True)
    ys = np.linspace(start=-1, stop=1, num=n_y, endpoint=True)

    y, x = np.meshgrid(ys, xs, indexing='ij')

    rho = np.sqrt(x**2 + y**2)

    # Use NaN propagation to make coords outside circle NaN
    rho = np.where(rho > 1, np.nan, rho)
    c = np.arcsin(rho)

    a0 = 0
    b0 = 0
    b = np.arcsin(np.cos(c) * np.sin(b0) + y * np.sin(c) * np.cos(b0) / rho)
    a = a0 + np.arctan2(x * np.sin(c), rho * np.cos(c) * np.cos(b0) - y * np.sin(c) * np.sin(b0))

    # Map to [-1, 1]
    b_hat = 2 * b / np.pi
    a_hat = a / np.pi

    # Create grid of (alpha,beta) coordinates.
    grid = np.stack((a_hat, b_hat), -1)

    # Map NaN coords to points outsize [-1, 1] so PyTorch makes it 0.
    grid = np.where(np.isnan(grid), -2, grid)
    return grid


def main():
    # load image
    x = imread("earth128.jpg").astype(np.float32).transpose((2, 0, 1)) / 255
    b = 64
    grid = torch.tensor(orthographic_grid(200, 200), dtype=torch.float, device="cuda")[None]

    x = torch.tensor(x, dtype=torch.float, device="cuda")
    x = x.view(1, 3, 2 * b, 2 * b)


    abc = (30 / 180 * np.pi, 50 / 180 * np.pi, 0)  # rotation angles

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plot(x, "x : signal on the sphere")

    plt.subplot(2, 2, 2)
    proj = F.grid_sample(x, grid)
    plot(proj, "projected")

    plt.subplot(2, 2, 3)
    rot = s2_rotation(x, *abc)
    plot(rot, "R(x) : rotation using fft")

    plt.subplot(2, 2, 4)
    proj = F.grid_sample(rot, grid)
    plot(proj, "rotated projected")

    plt.tight_layout()
    plt.show()
    # plt.savefig("fig.jpeg")


if __name__ == "__main__":
    main()