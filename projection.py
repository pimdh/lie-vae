import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread

import torch
from s2cnn import so3_rotation
from s2cnn.soft.gpu.s2_fft import S2_fft_real, S2_ifft_real
import torch.nn.functional as F

from lie_vae.utils import orthographic_grid
from lie_vae.lie_tools import complex_block_wigner_matrix_multiply

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
    ref_rot = s2_rotation(x, *abc)
    angles = torch.tensor(abc, dtype=torch.float, device='cuda')[None]
    spectra = S2_fft_real()(x).transpose(0, 1)  # [d**2, b, c, 2]->[b, d**2, c, 2]
    rotated_spectra = complex_block_wigner_matrix_multiply(angles, spectra, b-1)
    rot = S2_ifft_real()(rotated_spectra.transpose(0, 1).contiguous())
    print((rot - ref_rot).abs().max())
    plot(rot, "R(x) : rotation using fft")

    plt.subplot(2, 2, 4)
    proj = F.grid_sample(rot, grid)
    plot(proj, "rotated projected")

    plt.tight_layout()
    plt.show()
    # plt.savefig("fig.jpeg")


if __name__ == "__main__":
    main()