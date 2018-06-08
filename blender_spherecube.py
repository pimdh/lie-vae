# Example:
# blender --background --python blender_spherecube.py -- 100 /tmp/spherecube --size 64
import argparse, sys, os
import numpy as np
import bpy
import json

parser = argparse.ArgumentParser()
parser.add_argument('num', type=int)
parser.add_argument('dir')
parser.add_argument('--size', type=int, default=32)
parser.add_argument('--quaternions')
argv = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args(argv)

if not os.path.exists(args.dir):
    os.makedirs(args.dir)

bpy.ops.wm.open_mainfile(filepath='cube.blend')

for scene in bpy.data.scenes:
    ratio = 1
    scene.render.resolution_x = args.size*ratio
    scene.render.resolution_y = args.size*ratio
    scene.render.resolution_percentage = 100/ratio

for obj in bpy.data.objects:
    if obj.type == 'LAMP':
        obj.cycles_visibility.camera = not obj.cycles_visibility.camera

for m in bpy.data.materials:
    m.specular_intensity = 0.0

origin = (0, 0, 0)
b_empty = bpy.data.objects.new("Empty", None)
b_empty.rotation_mode = 'QUATERNION'
b_empty.location = origin
cam = scene.objects['Camera']
cam.location = (0, 5, 0)
cam_constraint = cam.constraints.new(type='TRACK_TO')
cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
cam_constraint.up_axis = 'UP_Y'
cam_constraint.use_target_z = True
b_empty = b_empty
cam_constraint.target = b_empty
cam.parent = b_empty  # setup parenting

scn = bpy.context.scene
scn.objects.link(b_empty)
scn.objects.active = b_empty

scene.render.image_settings.quality = 90  # set output format to .png

names = None
if args.quaternions:
    quaternions = np.load(args.quaternions)
    if quaternions.dtype == [('quaternion', '<f4', (4,)), ('name', 'S50')]:
        quaternions, names = quaternions['quaternion'], quaternions['name']
    assert len(quaternions.shape) == 2 and quaternions.shape[1] == 4, \
        "Parsed quaternions of incorrect shape {}".format(quaternions.shape)
else:
    # Uniform quaternion sampling
    # From http://planning.cs.uiuc.edu/node198.html
    u1, u2, u3 = np.random.uniform(0., 1., size=(3, args.num))
    quaternions = np.stack((
        np.sqrt(1-u1) * np.sin(2 * np.pi * u2),
        np.sqrt(1-u1) * np.cos(2 * np.pi * u2),
        np.sqrt(u1) * np.sin(2 * np.pi * u3),
        np.sqrt(u1) * np.cos(2 * np.pi * u3),
    ), 1).round(decimals=4)


# bpy.ops.render.render(write_still=True)
for i, quaternion in enumerate(quaternions):
    quaternion = quaternions[i]
    print("Quaternion {}: {}".format(i + 1, quaternion))

    b_empty.rotation_quaternion = tuple(float(x) for x in quaternion)

    if names is not None:
        filename = names[i].decode('utf-8')
    elif args.quaternions:
        filename = '{:06}'.format(i)
    else:
        filename = '{:.4f}_{:.4f}_{:.4f}_{:.4f}'.format(*quaternion)
    scene.render.filepath = os.path.join(args.dir, filename)
    bpy.ops.render.render(write_still=True)  # render still
