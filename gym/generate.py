#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python generate.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
import PIL
from PIL import ImageEnhance
from lie_learn.groups.SO3 import change_coordinates
import os
import os.path


def random_quaternions(n):
    u1, u2, u3 = np.random.uniform(0., 1., size=(3, n))
    return np.stack((
        np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
        np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
        np.sqrt(u1) * np.sin(2 * np.pi * u3),
        np.sqrt(u1) * np.cos(2 * np.pi * u3),
    ), 1).round(decimals=4)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument("num_samples", type=int)
    parser.add_argument("dir", type=str)
    parser.add_argument("--greyscale", type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(args.dir):
        os.makedirs(args.dir)

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    with tf.Session():
        tf_util.initialize()
        env = gym.make(args.envname)
        skip_steps = 200
        max_steps = env.spec.timestep_limit
        recorded_steps = 0
        env.reset()

        sim = env.unwrapped.sim
        sim.render(200, 200, mode='offscreen')
        cam = sim._render_context_offscreen.cam
        cam.distance = 2.5
        while recorded_steps < args.num_samples:
            obs = env.reset()
            done = False
            steps = 0

            quaternions = random_quaternions(max_steps - skip_steps)
            angles = change_coordinates(quaternions, 'Q', 'EA321')

            while not done and recorded_steps < args.num_samples:
                if steps >= skip_steps:
                    qpos = env.unwrapped.sim.data.qpos

                    # All zeros: walks towards camera
                    # ZYX intrinsic Euler angles.
                    # We first rotate the 'heading'
                    # Then rotate the elevation
                    # Then rotate the resulting image
                    q = quaternions[steps-skip_steps]
                    azimuth, elevation, roll = angles[steps-skip_steps]

                    cam.lookat[0:3] = qpos[0:3] - [0, 0, .4]
                    cam.azimuth = (azimuth * 180 / np.pi) + 180
                    cam.elevation = elevation * 180 / np.pi
                    i = sim.render(200, 200, mode='offscreen')
                    img = PIL.Image.fromarray(i)

                    if args.greyscale:
                        img = img.convert('L')
                    img = img.rotate(roll * 180 / np.pi - 180)
                    img = ImageEnhance.Brightness(img).enhance(4)
                    img = ImageEnhance.Contrast(img).enhance(0.8)
                    img = img.resize((64, 64), PIL.Image.LANCZOS)

                    filename = '{:.4f}_{:.4f}_{:.4f}_{:.4f}.jpg'.format(*q)
                    path = os.path.join(args.dir, filename)
                    with open(path, 'w') as f:
                        img.save(f, quality=90)

                    recorded_steps += 1

                action = policy_fn(obs[None, :])
                obs, r, done, _ = env.step(action)
                steps += 1
                if steps >= max_steps:
                    break

            print(recorded_steps, ' / ', args.num_samples)


if __name__ == '__main__':
    main()
