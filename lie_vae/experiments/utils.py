from functools import partial
from itertools import accumulate

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


class Subset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

    def __getattr__(self, item):
        return getattr(self.dataset, item)


def random_split(dataset, lengths):
    assert sum(lengths) == len(dataset)
    seed = np.random.get_state()
    np.random.seed(0)
    indices = torch.tensor(np.random.permutation(sum(lengths)),
                           dtype=torch.long)
    np.random.set_state(seed)

    return [Subset(dataset, indices[offset - length:offset])
            for offset, length in zip(accumulate(lengths), lengths)]


class View(nn.Module):
    def __init__(self, *v):
        super().__init__()
        self.v = v

    def forward(self, x):
        return x.view(*self.v)


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class ConstantSchedule:
    def __init__(self, value):
        self.value = value

    def __call__(self, x):
        return self.value


class LinearSchedule:
    def __init__(self, start_y, end_y, start_x, end_x):
        self.min_y = min(start_y, end_y)
        self.max_y = max(start_y, end_y)
        self.start_x = start_x
        self.start_y = start_y
        self.coef = (end_y - start_y) / (end_x - start_x)

    def __call__(self, x):
        return np.clip((x - self.start_x) * self.coef + self.start_y,
                       self.min_y, self.max_y).item(0)


def cycle(iterable):
    """Cycle iterable non-caching."""
    while True:
        for x in iterable:
            yield x


def expand_dim(x, n, dim=0):
    if dim < 0:
        dim = x.dim()+dim+1
    return x.unsqueeze(dim).expand(*[-1]*dim, n, *[-1]*(x.dim()-dim))


def test_linear_schedule():
    s = LinearSchedule(4, 10, 1, 4)

    np.testing.assert_allclose(s(0), 4)
    np.testing.assert_allclose(s(1), 4)
    np.testing.assert_allclose(s(2), 6)
    np.testing.assert_allclose(s(3), 8)
    np.testing.assert_allclose(s(4), 10)
    np.testing.assert_allclose(s(5), 10)

    s = LinearSchedule(10, 4, 1, 4)

    np.testing.assert_allclose(s(0), 10)
    np.testing.assert_allclose(s(1), 10)
    np.testing.assert_allclose(s(2), 8)
    np.testing.assert_allclose(s(3), 6)
    np.testing.assert_allclose(s(4), 4)
    np.testing.assert_allclose(s(5), 4)


if __name__ == '__main__':
    test_linear_schedule()