import torch
from math import pi
from lie_vae.lie_tools import random_group_matrices, rodrigues


class ContinuityLoss:
    """Auxiliary loss term for improving continuity of decoder."""

    def __init__(self, model, num_samples=32, lamb=1, scale=2 * pi / 200,
                 log=None, report_freq=1):
        self.model = model
        self.num_samples = num_samples
        self.lamb = lamb
        self.scale = scale

        self.log = log
        self.report_freq = report_freq

        self.diffs = []

    def __call__(self, it):
        device = next(self.model.parameters()).device
        g_starts = random_group_matrices(self.num_samples, device=device)
        v = torch.randn(self.num_samples, 3, device=device)
        v = v / v.norm(p=2, dim=1, keepdim=True) * self.scale
        g_ends = g_starts.bmm(rodrigues(v))

        dec_starts, dec_ends = self.model.decode(
            torch.stack([g_starts, g_ends], 0))

        diffs = (dec_starts - dec_ends).pow(2).view(self.num_samples, -1).sum(-1)

        self.diffs.apppend(diffs)

        mean = diffs.mean()

        if self.log and (it+1) % self.report_freq == 0:
            agg_diffs = torch.cat(diffs)
            self.log.add_scalar('discontinuity', agg_diffs.mean(), it+1)
            self.log.add_scalar('discontinuity_max', agg_diffs.max(), it+1)
            self.log.add_histogram('continuity', agg_diffs.detach().cpu().numpy(), it+1)
            self.diffs = []

        return mean * self.lamb