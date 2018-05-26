"""Added helper for geodesic rendering of spherecube."""
import torch
import numpy as np
from lie_vae.lie_tools import randomR, group_matrix_to_quaternions, rodrigues
import argparse
import json
from subprocess import call


parser = argparse.ArgumentParser()
parser.add_argument('num', type=int)
parser.add_argument('dir')
parser.add_argument('--size', type=int, default=32)
parser.add_argument('--identity_start', action='store_true')
args = parser.parse_args()

t = torch.linspace(0, 2*np.pi, args.num)

direction = torch.randn(3)
direction = direction / direction.norm()
x = t[:, None] * direction[None]
g = rodrigues(x)
g[0] = torch.eye(3)

if not args.identity_start:
    mean = torch.tensor(randomR(), dtype=torch.float32)
    g = mean.expand_as(g).bmm(g)

quaternions = group_matrix_to_quaternions(g).numpy().round(decimals=4).tolist()
q_json = json.dumps(quaternions)
print(q_json)


call(['blender', '--background', '--python', 'blender_spherecube.py', '--',
      str(args.num), args.dir, '--quaternions', q_json,
      '--size', str(args.size)])
