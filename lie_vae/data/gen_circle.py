import argparse
import torch
import numpy as np
from tempfile import NamedTemporaryFile
from subprocess import call, PIPE
from lie_vae.lie_tools import rodrigues, group_matrix_to_quaternions, s2s1rodrigues


def generate(num, axis, dir, size=64, tmppath=None, silent=True, start=None,
             noise=0):
    axis = torch.tensor(axis, dtype=torch.float32)
    axis = axis / axis.norm()

    angles = torch.linspace(0, 2 * np.pi, num)
    circles = torch.stack([torch.cos(angles), torch.sin(angles)], 1)

    r = s2s1rodrigues(axis[None], circles)

    if noise > 0:
        noises = torch.randn(num, 3) * noise
        r = rodrigues(noises).matmul(r)

    if start is not None:
        r = rodrigues(torch.tensor(start, dtype=torch.float32))[None].matmul(r)

    quaternions = group_matrix_to_quaternions(r)

    names = ['{:4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}'.format(a, *q)
             for a, q in zip(angles, quaternions)]

    data = np.zeros(num, dtype=[('quaternion', 'f4', (4,)), ('name', 'a50')])
    data['quaternion'] = quaternions
    data['name'] = names

    datafile = NamedTemporaryFile(dir=tmppath)
    np.save(datafile, data)
    datafile.flush()

    call(['blender', '--background', '--python', 'blender_spherecube.py',
          '--', str(num), dir, '--quaternions', datafile.name, '--size', str(size)],
         stdout=(PIPE if silent else None))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('num', type=int)
    parser.add_argument('axis', type=str)
    parser.add_argument('dir')
    parser.add_argument('--size', type=int, default=64)
    parser.add_argument('--start', type=str)
    parser.add_argument('--noise', type=float, default=0)
    args = parser.parse_args()
    axis = [float(x) for x in args.axis.split(',')]
    if args.start is not None:
        start = [float(x) for x in args.start.split(',')]
    else:
        start = None
    generate(args.num, axis, args.dir,
             size=args.size, silent=False, start=start, noise=args.noise)


if __name__ == '__main__':
    main()
