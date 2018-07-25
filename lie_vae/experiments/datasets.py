import re

import numpy as np
import os.path
from glob import glob

import torch
from PIL import Image
from torch.utils.data import Dataset, TensorDataset

from lie_learn.groups.SO3 import change_coordinates as SO3_coordinates
from lie_vae.lie_tools import block_wigner_matrix_multiply, random_quaternions, quaternions_to_eazyz


class ShapeDataset(Dataset):
    num_workers = 5
    rgb = False
    single_id = False

    def __init__(self, directory, subsample=1.):
        self.directory = directory
        index_path = os.path.join(directory, 'files.txt')
        if os.path.exists(index_path):
            with open(index_path, 'r') as f:
                self.files = f.read().splitlines()
            self.root = directory
        else:
            self.files = glob(os.path.join(directory, '**/*.jpg'), recursive=True)
            self.files += glob(os.path.join(directory, '**/*.png'), recursive=True)
            self.root = None
        self.files = sorted(self.files)

        if subsample < 1:
            seed = np.random.get_state()
            np.random.seed(0)
            self.files = np.random.choice(self.files, int(len(self.files) * subsample), replace=False)
            np.random.set_state(seed)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        return self.load_file(filename, self.root)

    @classmethod
    def load_file(cls, filename, root):
        path = os.path.join(root, filename) if root else filename
        image = Image.open(path)
        image_tensor = torch.tensor(np.array(image), dtype=torch.float32) / 255
        quaternion = cls.filename_to_quaternion(filename)

        if not cls.rgb:
            if image_tensor.dim() == 3:  # Mean if RGB
                image_tensor = image_tensor.mean(-1)

            image_tensor = image_tensor.unsqueeze(0)  # Add color channel
        else:
            image_tensor = image_tensor[:, :, :3].permute(2, 0, 1)

        group_el = torch.tensor(SO3_coordinates(quaternion, 'Q', 'MAT'),
                                dtype=torch.float32)
        name = 0 if cls.single_id else cls.filename_to_name(filename)

        return name, group_el, image_tensor

    @classmethod
    def filename_to_quaternion(cls, filename):
        """Remove extension, then retrieve _ separated floats"""
        matches = re.findall(r'-?[01]\.[0-9]{4}', filename)
        assert len(matches) == 4, 'No quaternion found in '+filename
        return [float(x) for x in matches]

    @classmethod
    def filename_to_name(cls, filename):
        match = re.search(r'([A-z0-9]+)\.obj', filename)

        assert match is not None, 'Could not find object id from filename'

        return match.group(1)

    @staticmethod
    def prep_batch(batch):
        return batch


class SphereCubeDataset(ShapeDataset):
    rgb = True
    single_id = True

    def __init__(self, subsample=1.):
        super().__init__('data/spherecube', subsample=subsample)


class ScPairsDataset(ShapeDataset):
    rgb = True
    single_id = True

    def __init__(self, subsample=1.):
        super().__init__('data/sc-pairs')

        n = len(self.files) // 2
        if subsample < 1:
            seed = np.random.get_state()
            np.random.seed(0)
            self.indices = np.random.permutation(n)[:int(n*subsample)]
            np.random.set_state(seed)
        else:
            self.indices = np.arange(n)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idx = self.indices[idx]
        filenames = self.files[2*idx:2*idx+2]
        assert len(filenames) == 2, "File not found"
        names, gs, imgs = zip(*[self.load_file(f, self.root) for f in filenames])
        names = torch.tensor(names)
        gs = torch.stack(gs, 0)
        imgs = torch.stack(imgs, 0)
        return names, gs, imgs


    @staticmethod
    def prep_batch(batch):
        return [t.view(-1, *t.shape[2:]) for t in batch]  # Flatten pairs


class ToyDataset(TensorDataset):
    num_workers = 0
    single_id = True
    rgb = False

    def __init__(self, tensors=None, device=None, path='data/toy.pickle'):
        if tensors is None:
            tensors = torch.load(path)
        if device is not None:
            tensors = [t.to(device) for t in tensors]
        super().__init__(*tensors)

    @classmethod
    def generate(cls, n=1000, degrees=6, rep_copies=10, device=None, batch_size=64):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        harmonics = torch.randn((degrees+1)**2, rep_copies, device=device)
        harmonics = harmonics / harmonics.norm() * 10
        xs, qs = [], []
        for i in range(0, n, batch_size):
            batch_n = min(i + batch_size, n)-i
            q = random_quaternions(batch_n, device=device)
            x = block_wigner_matrix_multiply(
                quaternions_to_eazyz(q), harmonics.expand(batch_n, -1, -1), degrees)
            xs.append(x), qs.append(q)
        return cls(tensors=(torch.cat(qs, 0),
                            harmonics.expand(n, -1, -1),
                            torch.cat(xs, 0)),
                   device=device)

    def save(self, path='data/toy.pickle'):
        torch.save(self.tensors, path)

    @staticmethod
    def prep_batch(batch):
        return batch
