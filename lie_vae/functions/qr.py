import torch
from .three_dim_tril_inverse import three_dim_tril_inverse


class QR(torch.autograd.Function):
    """Differentiable QR decomposition.

    Uses https://arxiv.org/abs/1009.6112
    """

    @staticmethod
    def forward(ctx, a):
        q, r = torch.qr(a)
        ctx.save_for_backward(q, r)
        return q, r

    @staticmethod
    def backward(ctx, dq, dr):
        q, r = ctx.saved_variables
        p_l = torch.tril(torch.ones_like(dq), diagonal=-1)

        x = r.matmul(dr.t()) - dr.matmul(r.t()) \
            + q.t().matmul(dq) - dq.t().matmul(q)
        return q.matmul(dr + (p_l * x).matmul(r.inverse().t()))


qr = QR.apply


class QRBatched3(torch.autograd.Function):
    """Differentiable QR decomposition for 3D matrices."""

    @staticmethod
    def forward(ctx, a):
        batch_shape = a.shape[:-2]
        matrix_shape = a.shape[-2:]
        a = a.view(-1, *matrix_shape)
        qs, rs = zip(*[torch.qr(x) for x in a])
        q = torch.stack(qs, 0)
        r = torch.stack(rs, 0)
        ctx.save_for_backward(q, r)
        q = q.view(*batch_shape, *matrix_shape)
        r = r.view(*batch_shape, *matrix_shape)
        return q, r

    @staticmethod
    def backward(ctx, dq, dr):
        batch_shape = dq.shape[:-2]
        matrix_shape = dq.shape[-2:]
        dq = dq.view(-1, *matrix_shape)
        dr = dr.view(-1, *matrix_shape)

        q, r = ctx.saved_variables
        p_l = torch.tril(dq.new_ones(dq.shape[-2:]), diagonal=-1).expand_as(dq)

        x = r.bmm(dr.transpose(-2, -1)) - dr.bmm(r.transpose(-2, -1)) \
            + q.transpose(-2, -1).bmm(dq) - dq.transpose(-2, -1).bmm(q)
        dq = q.bmm(dr + (p_l * x).bmm(three_dim_tril_inverse(r.transpose(-2, -1))))
        return dq.view(*batch_shape, *matrix_shape)


qr_batched3 = QRBatched3.apply


def test_qr():
    from time import time
    import numpy as np

    # Test correctness
    for i in range(1000):
        a = torch.rand(3, 3).double() * 10
        a.requires_grad = True

        torch.autograd.gradcheck(qr, (a,), eps=1E-3, atol=5E-3, rtol=5E-3)
        torch.autograd.gradcheck(qr_batched3, (a,), eps=1E-3, atol=5E-3, rtol=5E-3)

    # Test in batch
    xs = torch.rand(10, 3, 3).double() * 10
    xs.requires_grad = True
    torch.autograd.gradcheck(qr_batched3, (xs,), eps=1E-3, atol=5E-3, rtol=5E-3)

    # Evaluate performance of batched 3 versus general
    xs = torch.rand(10, 3, 3).double() * 10
    xs.requires_grad = True

    start = time()
    for x in xs:
        q, r = qr(x)
        torch.autograd.grad(q.sum()+r.sum(), (x,), retain_graph=True)
    print(time()-start)

    start = time()
    q, r = qr_batched3(xs)
    torch.autograd.grad(q.sum()+r.sum(), (xs,), retain_graph=True)
    print(time()-start)


if __name__ == '__main__':
    test_qr()