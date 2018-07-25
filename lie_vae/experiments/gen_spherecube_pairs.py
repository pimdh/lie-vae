import argparse
import torch
import numpy as np
from tempfile import NamedTemporaryFile
from subprocess import call, PIPE

from lie_vae.lie_tools import random_group_matrices, rodrigues, group_matrix_to_quaternions


def generate(num, step_size, dir, size=64, tmppath=None, silent=True):
    a_r = random_group_matrices(num)
    d = rodrigues(torch.randn(num, 3) * step_size)
    b_r = a_r.bmm(d)
    r = torch.stack([a_r, b_r], 1)
    q = group_matrix_to_quaternions(r)

    names = [['{:06}_{}_{:.4f}_{:.4f}_{:.4f}_{:.4f}'.format(i, j, *q[i, j])
              for j in range(2)] for i in range(num)]
    names = np.array(names)

    data = np.zeros((num, 2), dtype=[('quaternion', 'f4', (4,)), ('name', 'a50')])
    data['quaternion'] = q
    data['name'] = names
    data = data.flatten()

    datafile = NamedTemporaryFile(dir=tmppath)
    np.save(datafile, data)
    datafile.flush()

    call(['blender', '--background', '--python', 'blender_spherecube.py',
          '--', str(num), dir, '--quaternions', datafile.name, '--size', str(size)],
         stdout=(PIPE if silent else None))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('num', type=int)
    parser.add_argument('dir')
    parser.add_argument('--step_size', type=float, default=2 * np.pi / 60)
    parser.add_argument('--size', type=int, default=64)
    args = parser.parse_args()
    generate(args.num, args.step_size, args.dir, size=args.size, silent=False)


if __name__ == '__main__':
    main()
