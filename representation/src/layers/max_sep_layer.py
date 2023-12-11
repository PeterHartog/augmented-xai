import sys

import numpy as np
import torch
import torch.nn as nn
from numpy.linalg import norm
from scipy.spatial.distance import cdist


# V(order) = [1, sqrt(1 - 1/order**2), sqrt(1 - 1/order**2 - 1/(order-1)**2), ...]


def V(order: int):
    if order == 1:
        return np.array([[1, -1]])
    else:
        col1 = np.zeros((order, 1))
        col1[0] = 1
        row1 = -1 / order * np.ones((1, order))
        return np.concatenate(
            (col1, np.concatenate((row1, np.sqrt(1 - 1 / (order**2)) * V(order - 1)), axis=0)), axis=1
        )


def create_prototypes(nr_prototypes: int):
    assert nr_prototypes > 0
    prototypes = V(nr_prototypes - 1).T

    if nr_prototypes >= 1000:
        sys.setrecursionlimit(10000)

    assert prototypes.shape == (nr_prototypes, nr_prototypes - 1)
    assert np.all(np.abs(np.sum(np.power(prototypes, 2), axis=1) - 1) <= 1e-6)
    distances = cdist(prototypes, prototypes)

    assert distances[~np.eye(*distances.shape, dtype=bool)].std() <= 1e-3
    return prototypes.astype(np.float32)


def create_prototypes_random(nr_prototypes):
    prototypes = norm(np.random.uniform(size=(nr_prototypes, nr_prototypes - 1)))
    assert prototypes.shape == (nr_prototypes, nr_prototypes - 1)
    assert np.all(np.abs(np.sum(np.power(prototypes, 2), axis=1) - 1) <= 1e-6)
    return prototypes.astype(np.float32)


def create_noisy_prototypes(nr_prototypes, noise_scale=0):
    prototypes = create_prototypes(nr_prototypes)
    if noise_scale != 0:
        noise = np.random.normal(loc=0.0, scale=noise_scale, size=prototypes.shape)
        prototypes = norm(prototypes + noise)
    distances = cdist(prototypes, prototypes)
    avg_dist = distances[~np.eye(*distances.shape, dtype=bool)].mean()
    return prototypes.astype(np.float32), avg_dist


class MaxSeparatedLayer(nn.Module):
    def __init__(self, num_classes: int, prototype: str = "normal") -> None:
        super().__init__()
        if prototype == "normal":
            prototypes = create_prototypes(num_classes)
        elif prototype == "random":
            prototypes = create_prototypes_random(num_classes)
        elif prototype == "noisy":
            prototypes, _ = create_noisy_prototypes(num_classes)
        else:
            raise ValueError("Invalid prototype type")

        self.prototypes = nn.Parameter(torch.from_numpy(prototypes))

    def forward(self, x):
        return torch.matmul(x, self.prototypes.t())
