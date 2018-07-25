"""Equivariance loss for Encoder."""
from math import pi
import torch
import torch.nn as nn
import torch.nn.functional as F
from lie_vae.lie_tools import s2s1rodrigues
from lie_vae.experiments.utils import expand_dim


class EquivarianceLoss(nn.Module):
    """Equivariance Loss for SO(2) subgroup."""

    def __init__(self, model, num_samples=None, lamb=1.0, log=None, report_freq=1):
        super().__init__()
        self.model = model
        self.num_samples = num_samples
        self.lamb = lamb
        self.log = log
        self.report_freq = report_freq
        self.diffs = []

    def forward(self, img, encoding, it):
        assert encoding.shape[-2:] == (3, 3), "Rotation matrix input required"
        if self.num_samples:
            img, encoding = img[:self.num_samples], encoding[:self.num_samples]
        n = img.shape[0]
        theta = torch.rand(n, device=encoding.device) * 2 * pi
        v = torch.tensor([1, 0, 0], dtype=torch.float32, device=encoding.device)
        s1 = torch.stack((torch.cos(theta), torch.sin(theta)), 1)
        g = s2s1rodrigues(expand_dim(v, n), s1)

        enc_rot = g.bmm(encoding)
        img_rot = self.rotate(img, theta)
        img_rot_enc = self.model.encode(img_rot)[0][0]

        diffs = (enc_rot - img_rot_enc).pow(2).view(n, -1).sum(-1)
        self.diffs.append(diffs)
        mean = diffs.mean()

        lamb = self.lamb(it)

        if self.log and (it+1) % self.report_freq == 0:
            agg_diffs = torch.cat(self.diffs)
            self.log.add_scalar('equivariance', agg_diffs.mean(), it+1)
            self.log.add_scalar('equivariance_lamb', lamb, it+1)
            self.diffs = []

        return mean * lamb

    @staticmethod
    def rotate(img, theta):
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        zero = torch.zeros_like(theta)
        affine = torch.stack([cos, -sin, zero, sin, cos, zero], 1).view(-1, 2, 3)
        grid = F.affine_grid(affine, img.size())
        return F.grid_sample(img, grid)
