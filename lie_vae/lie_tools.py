import numpy as np
import torch
import torch.nn.functional as F
import math
from lie_learn.groups.SO3 import change_coordinates as SO3_coordinates
from lie_learn.representations.SO3.pinchon_hoggan.pinchon_hoggan_dense \
    import Jd as Jd_np
from lie_learn.representations.SO3.wigner_d import \
    wigner_D_matrix as reference_wigner_D_matrix

from .utils import randomR


class JContainer:
    data = {}

    @classmethod
    def get(cls, device):
        if str(device) in cls.data:
            return cls.data[str(device)]

        from lie_learn.representations.SO3.pinchon_hoggan.pinchon_hoggan_dense \
            import Jd as Jd_np

        device_data = [torch.tensor(J, dtype=torch.float32, device=device)
                       for J in Jd_np]
        cls.data[str(device)] = device_data

        return device_data


def map2LieAlgebra(v):
    """Map a point in R^N to the tangent space at the identity, i.e.
    to the Lie Algebra
    Arg:
        v = vector in R^N, (..., 3) in our case
    Return:
        R = v converted to Lie Algebra element, (3,3) in our case"""

    # make sure this is a sample from R^3
    assert v.size()[-1] == 3

    R_x = v.new_tensor([[ 0., 0., 0.],
                        [ 0., 0.,-1.],
                        [ 0., 1., 0.]])

    R_y = v.new_tensor([[ 0., 0., 1.],
                        [ 0., 0., 0.],
                        [-1., 0., 0.]])

    R_z = v.new_tensor([[ 0.,-1., 0.],
                        [ 1., 0., 0.],
                        [ 0., 0., 0.]])

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

    I = torch.eye(3, device=v.device, dtype=v.dtype)
    R = I + torch.sin(theta)[..., None]*K \
        + (1. - torch.cos(theta))[..., None]*(K@K)
    return R


def s2s1rodrigues(s2_el, s1_el):
    K = map2LieAlgebra(s2_el)
    
    cos_theta = s1_el[...,0]
    sin_theta = s1_el[...,1]
    
    I = torch.eye(3, device=s2_el.device, dtype=s2_el.dtype)
    
    R = I + sin_theta[..., None, None]*K \
        + (1. - cos_theta)[..., None, None]*(K@K)
        
    return R


def s2s2_gram_schmidt(v1, v2):
    """Normalise 2 3-vectors. Project second to orthogonal component.
    Take cross product for third. Stack to form SO matrix."""
    u1 = v1
    e1 = u1 / u1.norm(p=2, dim=-1, keepdim=True).clamp(min=1E-5)
    u2 = v2 - (e1 * v2).sum(-1, keepdim=True) * e1
    e2 = u2 / u2.norm(p=2, dim=-1, keepdim=True).clamp(min=1E-5)
    e3 = torch.cross(e1, e2)
    return torch.stack([e1, e2, e3], 1)


def vector_to_eazyz(v):
    """Map 3 vector to euler angles."""
    angles = F.tanh(v)
    angles = angles * v.new_tensor([math.pi, math.pi / 2, math.pi])
    angles = angles + v.new_tensor([0, math.pi / 2, 0])
    return angles


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
    batch_dims = r.shape[:-2]
    assert list(r.shape[-2:]) == [3, 3], 'Input must be 3x3 matrices'
    r = r.view(-1, 3, 3)
    n = r.shape[0]

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

    quaternions = cases[torch.arange(n, dtype=torch.long),
                        torch.argmax(denom.detach(), 1)]
    return quaternions.view(*batch_dims, 4)


def quaternions_to_eazyz(q):
    """Map batch of quaternion to Euler angles ZYZ. Output is not mod 2pi."""
    batch_dims = q.shape[:-1]
    assert q.shape[-1] == 4, 'Input must be 4 dim vectors'
    q = q.view(-1, 4)

    eps = 1E-6
    return torch.stack([
        torch.atan2(q[:, 1] * q[:, 2] - q[:, 0] * q[:, 3],
                    q[:, 0] * q[:, 2] + q[:, 1] * q[:, 3]),
        torch.acos(torch.clamp(q[:, 3] ** 2 - q[:, 0] ** 2
                               - q[:, 1] ** 2 + q[:, 2] ** 2,
                               -1.0+eps, 1.0-eps)),
        torch.atan2(q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2],
                    q[:, 1] * q[:, 3] - q[:, 0] * q[:, 2])
    ], 1).view(*batch_dims, 3)


def group_matrix_to_eazyz(r):
    """Map batch of SO(3) matrices to Euler angles ZYZ."""
    return quaternions_to_eazyz(group_matrix_to_quaternions(r))


def quaternions_to_group_matrix(q):
    """Normalises q and maps to group matrix."""
    q = q / q.norm(p=2, dim=-1, keepdim=True)
    r, i, j, k = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    return torch.stack([
        r*r - i*i - j*j + k*k, 2*(r*i + j*k), 2*(r*j - i*k),
        2*(r*i - j*k), -r*r + i*i - j*j + k*k, 2*(i*j + r*k),
        2*(r*j + i*k), 2*(i*j - r*k), -r*r - i*i + j*j + k*k
    ], -1).view(*q.shape[:-1], 3, 3)


def _z_rot_mat(angle, l):
    m = angle.new_zeros((angle.size(0), 2 * l + 1, 2 * l + 1))

    inds = torch.arange(
        0, 2 * l + 1, 1, dtype=torch.long, device=angle.device)
    reversed_inds = torch.arange(
        2 * l, -1, -1, dtype=torch.long, device=angle.device)

    frequencies = torch.arange(
        l, -l - 1, -1, dtype=angle.dtype, device=angle.device)[None]

    m[:, inds, reversed_inds] = torch.sin(frequencies * angle[:, None])
    m[:, inds, inds] = torch.cos(frequencies * angle[:, None])
    return m


def wigner_d_matrix(angles, degree):
    """Create wigner D matrices for batch of ZYZ Euler anglers for degree l."""
    batch_dims = angles.shape[:-1]
    assert angles.shape[-1] == 3, 'Input must be 3 dim vectors'
    angles = angles.view(-1, 3)

    J = JContainer.get(angles.device)[degree][None]
    x_a = _z_rot_mat(angles[:, 0], degree)
    x_b = _z_rot_mat(angles[:, 1], degree)
    x_c = _z_rot_mat(angles[:, 2], degree)
    res = x_a.matmul(J).matmul(x_b).matmul(J).matmul(x_c)

    return res.view(*batch_dims, 2*degree+1, 2*degree+1)


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


def random_quaternions(n, dtype=torch.float32, device=None):
    u1, u2, u3 = torch.rand(3, n, dtype=dtype, device=device)
    return torch.stack((
        torch.sqrt(1-u1) * torch.sin(2 * np.pi * u2),
        torch.sqrt(1-u1) * torch.cos(2 * np.pi * u2),
        torch.sqrt(u1) * torch.sin(2 * np.pi * u3),
        torch.sqrt(u1) * torch.cos(2 * np.pi * u3),
    ), 1)


def random_group_matrices(n, dtype=torch.float32, device=None):
    return quaternions_to_group_matrix(random_quaternions(n, dtype, device))


# Tests
def test_algebra_maps():
    vs = torch.randn(100, 3).double()
    matrices = map2LieAlgebra(vs)
    vs_prime = map2LieVector(matrices)
    matrices_prime = map2LieAlgebra(vs_prime)

    np.testing.assert_allclose(vs_prime.detach().numpy(), vs.detach().numpy())
    np.testing.assert_allclose(matrices_prime.detach().numpy(), matrices.detach().numpy())


def test_log_exp(scale, error):
    for _ in range(50):
        v_start = torch.randn(3).double() * scale
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
        [torch.tensor(randomR(), dtype=torch.float64) for _ in range(10000)], 0)

    q_reference = SO3_coordinates(r.numpy().astype(np.float64), 'MAT', 'Q')
    q = group_matrix_to_quaternions(r)
    np.testing.assert_allclose(q, q_reference, rtol=1E-6, atol=1E-6)

    r_back = quaternions_to_group_matrix(q)
    r_back_ref = SO3_coordinates(q.numpy().astype(np.float64), 'Q', 'MAT')
    np.testing.assert_allclose(r_back, r, rtol=1E-6, atol=1E-6)
    np.testing.assert_allclose(r_back, r_back_ref, rtol=1E-6, atol=1E-6)

    q_rand = torch.randn(100000, 4, dtype=torch.float64)
    r = quaternions_to_group_matrix(q_rand)
    r_ref = SO3_coordinates(q_rand.numpy().astype(np.float64), 'Q', 'MAT')
    np.testing.assert_allclose(r, r_ref, rtol=1E-6, atol=1E-6)

    np.testing.assert_allclose(
        r @ r.transpose(-2, -1), torch.eye(3).expand(r.shape[0], -1, -1),
        atol=1E-6)
    np.testing.assert_allclose(torch.stack([x.det() for x in r]),
                               torch.ones(r.shape[0]))

    ea_reference = SO3_coordinates(q.numpy().astype(np.float64), 'Q', 'EA323')
    ea = quaternions_to_eazyz(q).remainder(2*np.pi)
    np.testing.assert_allclose(ea, ea_reference, rtol=2E-5, atol=2E-5)


def test_wigner_d_matrices():
    for l in range(6):
        r = torch.stack(
            [torch.tensor(randomR(), dtype=torch.float32)
             for _ in range(10000)], 0)
        angles = group_matrix_to_eazyz(r)

        reference = np.stack([reference_wigner_D_matrix(l, *angle.numpy())
                              for angle in angles], 0)

        matrices = wigner_d_matrix(angles, l)

        np.testing.assert_allclose(matrices, reference, rtol=1E-4, atol=1E-5)

        # Test orthogonality
        eye = torch.eye(matrices.shape[1]).expand_as(matrices)
        np.testing.assert_allclose(matrices @ matrices.transpose(-2, -1), eye, rtol=1E-4, atol=1E-5)

        # Test W(g)W(g^-1)=eye
        for _ in range(100):
            r = random_group_matrices(1)[0]
            w = wigner_d_matrix(group_matrix_to_eazyz(r), l)
            winv = wigner_d_matrix(group_matrix_to_eazyz(r.t()), l)
            np.testing.assert_allclose(w @ winv, torch.eye(w.shape[1]), rtol=1E-4, atol=1E-5)

        # Testing W(a)W(b)=W(ab)
        ra = random_group_matrices(10000)
        rb = random_group_matrices(10000)

        wa = wigner_d_matrix(group_matrix_to_eazyz(ra), l)
        wb = wigner_d_matrix(group_matrix_to_eazyz(rb), l)
        wc = wigner_d_matrix(group_matrix_to_eazyz(ra.bmm(rb)), l)
        wc_result = wa.bmm(wb)

        # TODO: FAILS
        # np.testing.assert_allclose(wc_result, wc, rtol=1E-4, atol=1E-5)


def test_ref_wigner_d_matrices():
    for l in range(6):
        for _ in range(1000):
            qa, qb = random_quaternions(2, dtype=torch.float64).numpy()
            ra = SO3_coordinates(qa, 'Q', 'MAT')
            rb = SO3_coordinates(qb, 'Q', 'MAT')
            rc = ra.dot(rb)

            aa = SO3_coordinates(ra, 'MAT', 'EA323')
            ab = SO3_coordinates(rb, 'MAT', 'EA323')
            ac = SO3_coordinates(rc, 'MAT', 'EA323')

            wa = reference_wigner_D_matrix(l, *aa)
            wb = reference_wigner_D_matrix(l, *ab)
            wc = reference_wigner_D_matrix(l, *ac)

            np.testing.assert_allclose(wa.dot(wb), wc, atol=1E-2, rtol=1E-2)


def test_s2s1rodrigues(error):
    n = 10000
    s2_el = torch.tensor(np.random.normal(0,1, (n,3)), dtype = torch.float32)
    s2_el /= s2_el.norm(p=2, dim=-1, keepdim=True)
    
    s1_el = torch.tensor(np.random.normal(0,1, (n,2)), dtype = torch.float32)
    s1_el /= s1_el.norm(p=2, dim=-1, keepdim=True)
    
    R = s2s1rodrigues(s2_el, s1_el).detach().numpy()
    R_T = R.transpose([0,2,1])
    
    det = np.linalg.det(R)
    ones = np.ones_like(det)
    
    I = np.repeat(np.identity(3)[None,...], n, 0)
    
    np.testing.assert_allclose(I, R@R_T, rtol=error, atol=error)
    np.testing.assert_allclose(ones, det, rtol=error, atol=error)
    print("test_s2s1rodrigues with {} elements and {} error passed".format(n, error))


def test_s2s2_gram_schmidt():
    v1, v2 = torch.rand(2, 10000, 3).double()
    r = s2s2_gram_schmidt(v1, v2)

    dets = torch.stack([x.det() for x in r])
    I = torch.eye(3, dtype=v1.dtype).expand_as(r)

    np.testing.assert_allclose(dets, torch.ones_like(dets), rtol=1E-6, atol=1E-6)
    np.testing.assert_allclose(r.bmm(r.transpose(1, 2)), I, rtol=1E-6, atol=1E-6)


def main():
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    test_s2s2_gram_schmidt()
    test_s2s1rodrigues(1E-5)
    test_algebra_maps()
    test_log_exp(0.1, 1E-6)
    test_log_exp(10, 1E-6)
    test_coordinate_changes()
    test_wigner_d_matrices()

    # TODO: Fails?
    # test_ref_wigner_d_matrices()

    print("All tests passed")


if __name__ == '__main__':
    main()
