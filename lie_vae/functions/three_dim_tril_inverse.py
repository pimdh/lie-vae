import torch


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