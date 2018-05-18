# Based on https://raw.githubusercontent.com/panmari/stanford-shapenet-renderer/master/render_blender.py
# A simple script that uses blender to render views of a single object by rotation the camera around it.
# Also produces depth map at the same time.
#
# Example:
# Single x 100k:
# blender --background --python blender_script.py -- --views 100000 assets/chair.obj --output_folder ./data/chairs/single --output_size 64x64
#
# All x 1000:
# find assets_chairs -name '*.obj' -print0 | xargs -0 -n1 -P6 -I {} blender --background --python blender_script.py -- --output_size 64x64 --output_folder data/chairs/all --views 1000 {}
#
# 10 types x 100k:
# cat data/chairs/selected_chairs.txt | xargs -n1 -P5 -I {} blender --background --python blender_script.py -- --output_size 64x64 --output_folder data/chairs/ten --views 100000 {}
#
# ShapenetCore: http://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetCore.v2.zip
#
# Gives Quaternion rotation angles. Convert to other stuff with module pyquaternion.

import argparse, sys, os
import numpy as np
import bpy

parser = argparse.ArgumentParser()
parser.add_argument('num', type=int)
parser.add_argument('dir')
parser.add_argument('--size', type=int, default=32)
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

# Uniform quaternion sampling
# From http://planning.cs.uiuc.edu/node198.html
u1, u2, u3 = np.random.uniform(0., 1., size=(3, args.num))
quaternions = np.stack((
    np.sqrt(1-u1) * np.sin(2 * np.pi * u2),
    np.sqrt(1-u1) * np.cos(2 * np.pi * u2),
    np.sqrt(u1) * np.sin(2 * np.pi * u3),
    np.sqrt(u1) * np.cos(2 * np.pi * u3),
), 1).round(decimals=4)

bpy.ops.render.render(write_still=True)

for i in range(args.num):
    quaternion = quaternions[i]
    print("Quaternion {}: {}".format(i + 1, quaternion))

    b_empty.rotation_quaternion = tuple(float(x) for x in quaternion)

    path = os.path.join(args.dir, './{:.4f}_{:.4f}_{:.4f}_{:.4f}'.format(*quaternion))
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)  # render still
