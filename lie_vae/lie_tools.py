import numpy as np
import torch
from lie_learn.groups.SO3 import change_coordinates as SO3_coordinates
from lie_learn.representations.SO3.pinchon_hoggan.pinchon_hoggan_dense \
    import Jd as Jd_np
from lie_learn.representations.SO3.wigner_d import \
    wigner_D_matrix as reference_wigner_D_matrix

from .utils import randomR

device_strings = ['cpu']
for i in range(torch.cuda.device_count()):
    device_strings.append('cuda:%d' % i)

# Copy to all devices
Jd = {d: [torch.tensor(J, dtype=torch.float32, device=d) for J in Jd_np]
      for d in device_strings}


def map2LieAlgebra(v):
    """Map a point in R^N to the tangent space at the identity, i.e.
    to the Lie Algebra
    Arg:
        v = vector in R^N, (..., 3) in our case
    Return:
        R = v converted to Lie Algebra element, (3,3) in our case"""

    # make sure this is a sample from R^3
    assert v.size()[-1] == 3

    R_x = torch.tensor([[ 0., 0., 0.],
                        [ 0., 0.,-1.],
                        [ 0., 1., 0.]], device=v.device)

    R_y = torch.tensor([[ 0., 0., 1.],
                        [ 0., 0., 0.],
                        [-1., 0., 0.]], device=v.device)

    R_z = torch.tensor([[ 0.,-1., 0.],
                        [ 1., 0., 0.],
                        [ 0., 0., 0.]], device=v.device)

    R = R_x * v[..., 0, None, None] + \
        R_y * v[..., 1, None, None] + \
        R_z * v[..., 2, None, None]
    return R


def map2LieVector(X):
    """Map Lie algebra in ordinary (3, 3) matrix rep to vector.

    In literature known as 'vee' map.

    inverse of map2LieAlgebra
    """
    return torch.stack((-X[..., 1, 2], X[..., 0, 2], -X[..., 0, 1]), -1)


def rodrigues(v):
    theta = v.norm(p=2, dim=-1, keepdim=True)
    # normalize K
    K = map2LieAlgebra(v/theta)

    I = torch.eye(3, device=v.device)
    R = I + torch.sin(theta)[..., None]*K \
        + (1. - torch.cos(theta))[..., None]*(K@K)
    a = torch.sin(theta)[..., None]
    return R


def log_map(R):
    """Map Matrix SO(3) element to Algebra element.

    Input is taken to be 3x3 matrices of ordinary representation.
    Output algebra element in 3x3 L_i representation.
    Uses https://en.wikipedia.org/wiki/Rotation_group_SO(3)#Logarithm_map
    """
    anti_sym = .5 * (R - R.transpose(-1, -2))
    theta = torch.acos(.5 * (torch.trace(R)-1))
    return theta / torch.sin(theta) * anti_sym


def group_matrix_to_quaternions(r):
    """Map batch of SO(3) matrices to quaternions."""
    n = r.size(0)

    diags = [r[:, 0, 0], r[:, 1, 1], r[:, 2, 2]]
    denom_pre = torch.stack([
        1 + diags[0] - diags[1] - diags[2],
        1 - diags[0] + diags[1] - diags[2],
        1 - diags[0] - diags[1] + diags[2],
        1 + diags[0] + diags[1] + diags[2]
    ], 1)
    denom = 0.5 * torch.sqrt(1E-6 + torch.abs(denom_pre))

    case0 = torch.stack([
        denom[:, 0],
        (r[:, 0, 1] + r[:, 1, 0]) / (4 * denom[:, 0]),
        (r[:, 0, 2] + r[:, 2, 0]) / (4 * denom[:, 0]),
        (r[:, 1, 2] - r[:, 2, 1]) / (4 * denom[:, 0])
    ], 1)
    case1 = torch.stack([
        (r[:, 0, 1] + r[:, 1, 0]) / (4 * denom[:, 1]),
        denom[:, 1],
        (r[:, 1, 2] + r[:, 2, 1]) / (4 * denom[:, 1]),
        (r[:, 2, 0] - r[:, 0, 2]) / (4 * denom[:, 1])
    ], 1)
    case2 = torch.stack([
        (r[:, 0, 2] + r[:, 2, 0]) / (4 * denom[:, 2]),
        (r[:, 1, 2] + r[:, 2, 1]) / (4 * denom[:, 2]),
        denom[:, 2],
        (r[:, 0, 1] - r[:, 1, 0]) / (4 * denom[:, 2])
    ], 1)
    case3 = torch.stack([
        (r[:, 1, 2] - r[:, 2, 1]) / (4 * denom[:, 3]),
        (r[:, 2, 0] - r[:, 0, 2]) / (4 * denom[:, 3]),
        (r[:, 0, 1] - r[:, 1, 0]) / (4 * denom[:, 3]),
        denom[:, 3]
    ], 1)

    cases = torch.stack([case0, case1, case2, case3], 1)

    return cases[torch.arange(n, dtype=torch.long),
                 torch.argmax(denom.detach(), 1)]


def quaternions_to_eazyz(q):
    """Map batch of quaternion to Euler angles ZYZ. Output is not mod 2pi."""
    eps = 1E-6
    return torch.stack([
        torch.atan2(q[:, 1] * q[:, 2] - q[:, 0] * q[:, 3],
                    q[:, 0] * q[:, 2] + q[:, 1] * q[:, 3]),
        torch.acos(torch.clamp(q[:, 3] ** 2 - q[:, 0] ** 2
                               - q[:, 1] ** 2 + q[:, 2] ** 2,
                               -1.0+eps, 1.0-eps)),
        torch.atan2(q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2],
                    q[:, 1] * q[:, 3] - q[:, 0] * q[:, 2])
    ], 1)


def group_matrix_to_eazyz(r):
    """Map batch of SO(3) matrices to Euler angles ZYZ."""
    return quaternions_to_eazyz(group_matrix_to_quaternions(r))


def _z_rot_mat(angle, l):
    m = torch.zeros(
        (angle.size(0), 2 * l + 1, 2 * l + 1),
        dtype=torch.float32, device=angle.device)

    inds = torch.arange(
        0, 2 * l + 1, 1, dtype=torch.long, device=angle.device)
    reversed_inds = torch.arange(
        2 * l, -1, -1, dtype=torch.long, device=angle.device)

    frequencies = torch.arange(
        l, -l - 1, -1, dtype=torch.float32, device=angle.device)[None]

    m[:, inds, reversed_inds] = torch.sin(frequencies * angle[:, None])
    m[:, inds, inds] = torch.cos(frequencies * angle[:, None])
    return m


def wigner_d_matrix(angles, degree):
    """Create wigner D matrices for batch of ZYZ Euler anglers for degree l."""
    J = Jd[str(angles.device)][degree][None]
    x_a = _z_rot_mat(angles[:, 0], degree)
    x_b = _z_rot_mat(angles[:, 1], degree)
    x_c = _z_rot_mat(angles[:, 2], degree)
    return x_a.matmul(J).matmul(x_b).matmul(J).matmul(x_c)


def block_wigner_matrix_multiply(angles, data, max_degree):
    """Transform data using wigner d matrices for all degrees.

    vector_dim is dictated by max_degree by the expression:
    vector_dim = \sum_{i=0}^max_degree (2 * max_degree + 1) = (max_degree+1)^2

    The representation is the direct sum of the irreps of the degrees up to max.
    The computation is equivalent to a block-wise matrix multiply.

    The data are the Fourier modes of a R^{data_dim} signal.

    Input:
    - angles (batch, 3)  ZYZ Euler angles
    - vector (batch, vector_dim, data_dim)

    Output: (batch, vector_dim, data_dim)
    """
    outputs = []
    start = 0
    for degree in range(max_degree+1):
        dim = 2 * degree + 1
        matrix = wigner_d_matrix(angles, degree)
        outputs.append(matrix.bmm(data[:, start:start+dim, :]))
        start += dim
    return torch.cat(outputs, 1)


# Tests
def test_algebra_maps():
    vs = torch.randn(100, 3)
    matrices = map2LieAlgebra(vs)
    vs_prime = map2LieVector(matrices)
    matrices_prime = map2LieAlgebra(vs_prime)

    np.testing.assert_allclose(vs_prime.detach().numpy(), vs.detach().numpy())
    np.testing.assert_allclose(matrices_prime.detach().numpy(), matrices.detach().numpy())


def test_log_exp(scale, error):
    for _ in range(50):
        v_start = torch.randn(3) * scale
        R = rodrigues(v_start)
        v = map2LieVector(log_map(R))
        R_prime = rodrigues(v)
        v_prime = map2LieVector(log_map(R_prime))
        np.testing.assert_allclose(R_prime.detach(), R.detach(),
                                   rtol=error, atol=error)
        np.testing.assert_allclose(v_prime.detach(), v.detach(),
                                   rtol=error, atol=error)


def test_coordinate_changes():
    r = torch.stack(
        [torch.tensor(randomR(), dtype=torch.float32) for _ in range(10000)], 0)

    q_reference = SO3_coordinates(r.numpy().astype(np.float64), 'MAT', 'Q')
    q = group_matrix_to_quaternions(r)
    np.testing.assert_allclose(q, q_reference, rtol=1E-5, atol=1E-5)

    ea_reference = SO3_coordinates(q.numpy().astype(np.float64), 'Q', 'EA323')
    ea = quaternions_to_eazyz(q).remainder(2*np.pi)
    np.testing.assert_allclose(ea, ea_reference, rtol=1E-5, atol=1E-5)


def test_wigner_d_matrices():
    for l in range(5):
        r = torch.stack(
            [torch.tensor(randomR(), dtype=torch.float32)
             for _ in range(10000)], 0)
        angles = group_matrix_to_eazyz(r)

        reference = np.stack([reference_wigner_D_matrix(l, *angle.numpy().T)
                              for angle in angles], 0)

        matrices = wigner_d_matrix(angles, l)

        np.testing.assert_allclose(matrices, reference, rtol=1E-4, atol=1E-5)


if __name__ == '__main__':
    test_algebra_maps()
    test_log_exp(0.1, 1E-4)
    test_log_exp(10, 5E-4)
    test_coordinate_changes()
    test_wigner_d_matrices()

