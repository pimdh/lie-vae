# Example:
# blender --background --python blender_spherecube.py -- 100 /tmp/spherecube --size 64
import argparse, sys, os
import numpy as np
import bpy
import json

parser = argparse.ArgumentParser()
parser.add_argument('dir')
parser.add_argument('quaternions')
parser.add_argument('--size', type=int, default=500)
argv = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args(argv)

if not os.path.exists(args.dir):
    os.makedirs(args.dir)

bpy.ops.wm.open_mainfile(filepath='axes.blend')

scene = bpy.data.scenes[0]
origin = (0, 0, 0)
b_empty = bpy.data.objects.new("Empty", None)
b_empty.rotation_mode = 'QUATERNION'
b_empty.location = origin
cam = scene.objects['Camera']
cam.location = (0, 2.5, 0)
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

quaternions = json.loads(args.quaternions)
quaternions = np.array(quaternions, dtype=np.float32).round(decimals=4)
assert len(quaternions.shape) == 2 and quaternions.shape[1] == 4, \
    "Parsed quaternions of incorrect shape {}".format(quaternions.shape)

# bpy.ops.render.render(write_still=True)
for i, quaternion in enumerate(quaternions):
    quaternion = quaternions[i]
    print("Quaternion {}: {}".format(i + 1, quaternion))

    b_empty.rotation_quaternion = tuple(float(x) for x in quaternion)

    path = os.path.join(args.dir, './{:03}'.format(i, *quaternion))
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)  # render still
