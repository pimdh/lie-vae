import torch
import numpy as np
import math
from torch.distributions.multivariate_normal import MultivariateNormal, \
    _batch_diag, _batch_potrf_lower, _batch_inverse


def _three_dim_tril_inverse(L):
    """Compute three dim lower triag inverse batch-wise."""
    inv = torch.zeros_like(L)
    i = torch.arange(3, dtype=torch.long, device=L.device)
    inv[..., i, i] = 1 / L[..., i, i]
    inv[..., 1, 0] = -L[..., 1, 0] / (L[..., 0, 0] * L[..., 1, 1])
    inv[..., 2, 1] = -L[..., 2, 1] / (L[..., 2, 2] * L[..., 1, 1])
    inv[..., 2, 0] = (L[..., 1, 0] * L[..., 2, 1] - L[..., 2, 0] * L[..., 1, 1]) \
                     / (L[..., 2, 2] * L[..., 1, 1] * L[..., 0, 0])
    return inv


class ThreeDimTrilInverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        inverse = _three_dim_tril_inverse(input)
        ctx.save_for_backward(inverse)
        return inverse

    @staticmethod
    def backward(ctx, grad_output):
        inverse, = ctx.saved_variables
        s = inverse.shape
        inverse = inverse.view(-1, *s[-2:])
        grad_output = grad_output.view(-1, *s[-2:])
        inverse_t = inverse.transpose(1, 2)
        grad = -torch.bmm(inverse_t, torch.bmm(grad_output, inverse_t))
        grad = grad.view(*s)
        tril = grad.new_ones(s[-2:]).tril().expand_as(grad)
        return grad * tril


three_dim_tril_inverse = ThreeDimTrilInverse.apply


def _batch_mahalanobis_three(L, x):
    r"""
    Computes the squared Mahalanobis distance :math:`\mathbf{x}^\top\mathbf{M}^{-1}\mathbf{x}`
    for a factored 3 dimensional :math:`\mathbf{M} = \mathbf{L}\mathbf{L}^\top`.

    Accepts batches for both L and x.
    """
    L_inv = three_dim_tril_inverse(L).transpose(-2, -1)
    return (x.unsqueeze(-1) * L_inv).sum(-2).pow(2.0).sum(-1)


class ThreevariateNormal(MultivariateNormal):
    def __init__(self, loc, scale_tril):
        super().__init__(loc, scale_tril=scale_tril)
        assert scale_tril.shape[-2] == 3 and scale_tril.shape[-1] == 3

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        diff = value - self.loc
        M = _batch_mahalanobis_three(self.scale_tril, diff)
        log_det = _batch_diag(self.scale_tril).abs().log().sum(-1)
        return -0.5 * (M + self.loc.size(-1) * math.log(2 * math.pi)) - log_det


def test_threevariate_normal():
    from time import time
    m = torch.randn(10, 10, 3, 3, dtype=torch.float32)
    cov = torch.einsum('abij,ablj->abil', (m, m))
    scales = _batch_potrf_lower(cov)
    # Test near identity to avoid numerical issues.
    scales = torch.eye(3, dtype=torch.float32) + 0.1 * scales
    scales.requires_grad = True

    inverses_a = _batch_inverse(scales)
    inverses_b = three_dim_tril_inverse(scales)

    np.testing.assert_allclose(inverses_a.detach(), inverses_b.detach(), atol=1E-7, rtol=1E-7)

    locs = torch.randn(10, 10, 3, dtype=torch.float32) * 0.1
    vals = torch.randn(5, 10, 10, 3, dtype=torch.float32) * 0.1

    start = time()
    log_prob_a = MultivariateNormal(locs, scale_tril=scales).log_prob(vals)
    grad_a = torch.autograd.grad(log_prob_a.mean(), scales, retain_graph=True)[0]
    time_a = time()-start

    start = time()
    log_prob_b = ThreevariateNormal(locs, scale_tril=scales).log_prob(vals)
    grad_b = torch.autograd.grad(log_prob_b.mean(), scales, retain_graph=True)[0]
    time_b = time()-start

    np.testing.assert_allclose(log_prob_b.detach(), log_prob_a.detach(), atol=1E-7, rtol=1E-7)
    np.testing.assert_allclose(grad_b.detach(), grad_a.detach(), atol=1E-3, rtol=1E-4)

    print(time_a, time_b)


def test_inverse_grad():
    m = torch.randn(10, 10, 3, 3, dtype=torch.float32)
    cov = torch.einsum('abij,ablj->abil', (m, m))
    scales = _batch_potrf_lower(cov)
    # Test near identity to avoid numerical issues.
    scales = torch.eye(3, dtype=torch.float32) + 0.1 * scales
    scales.requires_grad = True
    torch.autograd.gradcheck(three_dim_tril_inverse, [scales], atol=2E-5, eps=1E-3)


if __name__ == '__main__':
    test_threevariate_normal()
    test_inverse_grad()
