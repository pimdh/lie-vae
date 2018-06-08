import numpy as np
from tempfile import TemporaryDirectory, NamedTemporaryFile
from subprocess import call, PIPE
from glob import glob
from PIL import Image


def render_axes(quaternions, size=500, tmppath=None, silent=True):
    quaternions = np.asarray(quaternions)
    assert quaternions.shape[-1] == 4, 'Quaternions should be batches of 4 vectors'
    batch_shape = quaternions.shape[:-1]
    quaternions = quaternions.reshape((-1, 4)).tolist()

    datafile = NamedTemporaryFile(dir=tmppath)
    np.save(datafile, quaternions)
    datafile.flush()

    with TemporaryDirectory(dir=tmppath) as t:
        call(['blender', '--background', '--python', 'blender_axes.py',
              '--', t, datafile.name, '--size', str(size)],
             stdout=(PIPE if silent else None))

        paths = sorted(glob(t+'/**.png'))

        images = [np.array(Image.open(path).convert('RGB')) for path in paths]
    image_shape = images[0].shape
    return np.stack(images).reshape(*batch_shape, *image_shape)


