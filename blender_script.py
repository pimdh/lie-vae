# Based on https://raw.githubusercontent.com/panmari/stanford-shapenet-renderer/master/render_blender.py
# A simple script that uses blender to render views of a single object by rotation the camera around it.
# Also produces depth map at the same time.
#
# Example:
# blender --background --python blender_script.py -- --views 100000 assets/chair.obj --output_folder ./shapes --output_size 64x64
# find assets_chairs -name '*.obj' -print0 | xargs -0 -n1 -P6 -I {} blender --background --python blender_script.py -- --output_size 64x64 --output_folder shapes_chairs --views 1000 {}
#
# ShapenetCore: http://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetCore.v2.zip
#
# Gives Quaternion rotation angles. Convert to other stuff with module pyquaternion.

import argparse, sys, os
import numpy as np

parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
parser.add_argument('--views', type=int, default=30,
                    help='number of views to be rendered')
parser.add_argument('obj', type=str,
                    help='Path to the obj file to be rendered.')
parser.add_argument('--output_folder', type=str, default='/tmp',
                    help='The path the output will be dumped to.')
parser.add_argument('--scale', type=float, default=1,
                    help='Scaling factor applied to model. Depends on size of mesh.')
parser.add_argument('--remove_doubles', type=bool, default=True,
                    help='Remove double vertices to improve mesh quality.')
parser.add_argument('--edge_split', type=bool, default=True,
                    help='Adds edge split filter.')
parser.add_argument('--depth_scale', type=float, default=1.4,
                    help='Scaling that is applied to depth. Depends on size of mesh. Try out various values until you get a good result. Ignored if format is OPEN_EXR.')
parser.add_argument('--color_depth', type=str, default='8',
                    help='Number of bit per channel used for output. Either 8 or 16.')
parser.add_argument('--format', type=str, default='PNG',
                    help='Format of files generated. Either PNG or OPEN_EXR')
parser.add_argument('--output_size', type=str, default='600x600',
                    help='Output size, e.g.: 600x600')
parser.add_argument('--quaternion', type=str,
                    help='Compute one fixed quaternion. 4 floats separated by commas')

argv = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args(argv)

import bpy

# Set up rendering of depth map.
bpy.context.scene.use_nodes = True
tree = bpy.context.scene.node_tree
links = tree.links

# Add passes for additionally dumping albedo and normals.
bpy.context.scene.render.layers["RenderLayer"].use_pass_normal = True
bpy.context.scene.render.layers["RenderLayer"].use_pass_color = True
bpy.context.scene.render.image_settings.file_format = args.format
bpy.context.scene.render.image_settings.color_depth = args.color_depth

# Clear default nodes
for n in tree.nodes:
    tree.nodes.remove(n)

# Create input render layer node.
render_layers = tree.nodes.new('CompositorNodeRLayers')

# Delete default cube
bpy.data.objects['Cube'].select = True
bpy.ops.object.delete()

bpy.ops.import_scene.obj(filepath=args.obj)

for object in bpy.context.scene.objects:
    if object.name in ['Camera', 'Lamp']:
        continue
    bpy.context.scene.objects.active = object

    if args.remove_doubles:
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.remove_doubles()
        bpy.ops.object.mode_set(mode='OBJECT')
    if args.edge_split:
        bpy.ops.object.modifier_add(type='EDGE_SPLIT')
        bpy.context.object.modifiers["EdgeSplit"].split_angle = 1.32645
        bpy.ops.object.modifier_apply(apply_as='DATA', modifier="EdgeSplit")

boxes = []
dims = []
for object in bpy.context.scene.objects:
    if object.name in ['Camera', 'Lamp']:
        continue
    bpy.context.scene.objects.active = object

    box = np.array(object.bound_box)
    boxes.append(box)
    dims.append(np.array(object.dimensions))


dims = np.stack(dims, 0)
boxes = np.concatenate(boxes, 0)
# print(boxes)
sides = (boxes.max(axis=0) - boxes.min(axis=0))
# print(dims)
# print("Max dims ", dims.max())
print("Sides ", sides)

# d = np.linalg.norm(box, axis=1).max()
# print("Size ", d)

scale = 0.757301986217 / sides.max() * .7
print("Scale ", scale)
# scale = 0.8


for object in bpy.context.scene.objects:
    if object.name in ['Camera', 'Lamp']:
        continue
    bpy.context.scene.objects.active = object

    bpy.context.scene.objects.active.scale = (scale, scale, scale)
    #
    # bpy.ops.transform.resize(value=(scale,scale,scale))
    # bpy.ops.object.transform_apply(scale=True)

# Make light just directional, disable shadows.
lamp = bpy.data.lamps['Lamp']
lamp.type = 'SUN'
lamp.shadow_method = 'NOSHADOW'
# Possibly disable specular shading:
lamp.use_specular = False

# Add another light source so stuff facing away from light is not completely dark
bpy.ops.object.lamp_add(type='SUN')
lamp2 = bpy.data.lamps['Sun']
lamp2.shadow_method = 'NOSHADOW'
lamp2.use_specular = False
lamp2.energy = 0.5
bpy.data.objects['Sun'].rotation_euler = bpy.data.objects['Lamp'].rotation_euler
bpy.data.objects['Sun'].rotation_euler[0] += 180


def parent_obj_to_camera(b_camera):
    """Create empty element at origin and make child of camera."""
    origin = (0, 0, 0)
    b_empty = bpy.data.objects.new("Empty", None)
    b_empty.rotation_mode = 'QUATERNION'
    b_empty.location = origin
    b_camera.parent = b_empty  # setup parenting

    scn = bpy.context.scene
    scn.objects.link(b_empty)
    scn.objects.active = b_empty
    return b_empty


scene = bpy.context.scene
dims = [int(x) for x in args.output_size.split('x')]
scene.render.resolution_x, scene.render.resolution_y = dims
scene.render.resolution_percentage = 100
# scene.render.alpha_mode = 'TRANSPARENT'
cam = scene.objects['Camera']
# cam.location = (0, 1, 0.6)
cam.location = (0, 1, 0)
cam_constraint = cam.constraints.new(type='TRACK_TO')
cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
cam_constraint.up_axis = 'UP_Y'
cam_constraint.use_target_z = True
b_empty = parent_obj_to_camera(cam)
cam_constraint.target = b_empty

fp = os.path.join(args.output_folder, args.obj)
scene.render.image_settings.file_format = 'JPEG'  # set output format to .png
scene.render.image_settings.quality = 98  # set output format to .png

# Uniform quaternion sampling
# From http://planning.cs.uiuc.edu/node198.html
u1, u2, u3 = np.random.uniform(0., 1., size=(3, args.views))
quaternions = np.stack((
    np.sqrt(1-u1) * np.sin(2 * np.pi * u2),
    np.sqrt(1-u1) * np.cos(2 * np.pi * u2),
    np.sqrt(u1) * np.sin(2 * np.pi * u3),
    np.sqrt(u1) * np.cos(2 * np.pi * u3),
), 1).round(decimals=4)


num_views = 1 if args.quaternion else args.views
for i in range(num_views):
    if args.quaternion:
        quaternion = [float(x) for x in args.quaternion.split(',')]
    else:
        quaternion = quaternions[i]
    print("Quaternion {}: {}".format(i + 1, quaternion))

    b_empty.rotation_quaternion = tuple(float(x) for x in quaternion)

    scene.render.filepath = fp + \
        '/{:.4f}_{:.4f}_{:.4f}_{:.4f}'.format(*quaternion)
    bpy.ops.render.render(write_still=True)  # render still
