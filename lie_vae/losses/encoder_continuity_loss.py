"""Continuity loss for Encoder."""
import torch
import torch.nn as nn


class EncoderContinuityLoss(nn.Module):
    """Encoder continuity loss for nearby input pairs."""

    def __init__(self, model, lamb=1.0, log=None, report_freq=1):
        super().__init__()
        self.model = model
        self.lamb = lamb
        self.log = log
        self.report_freq = report_freq
        self.diffs = []

    def forward(self, encodings, it):
        assert encodings.shape[-2:] == (3, 3), "Rotation matrix input required"

        encodings = encodings.view(-1, 2, 3, 3)
        diffs = (encodings[:, 0] - encodings[:, 1]).pow(2).view(-1, 9).sum(-1)
        self.diffs.append(diffs)
        mean = diffs.mean()

        if callable(self.lamb):
            lamb = self.lamb(it)
        else:
            lamb = self.lamb

        if self.log and (it+1) % self.report_freq == 0:
            agg_diffs = torch.cat(self.diffs)
            self.log.add_scalar('encoder_continuity', agg_diffs.mean(), it+1)
            self.log.add_scalar('encoder_continuity_lamb', lamb, it+1)
            self.diffs = []

        return mean * lamb
