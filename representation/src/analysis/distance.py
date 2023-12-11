from typing import Optional

import torch
from scipy.spatial.distance import cdist
from torch import Tensor


def scale_and_shift(tensor: Tensor) -> Tensor:
    tensor = tensor - tensor.min()
    tensor = tensor / tensor.max()
    return tensor


def tensor_distance(a: Tensor, b: Tensor, distance_method: str) -> Tensor:
    return torch.tensor(cdist(a.numpy(), b.numpy(), metric=distance_method))  # type: ignore


def calculate_distances(batch: Tensor, distance_method: str, rank: bool = False, top_k: Optional[int] = None) -> Tensor:
    if rank:
        _, rank = torch.sort(batch, dim=-1)
        batch = rank  # type: ignore

    # if top_k is not None:
    #     val, ind = torch.topk(batch, k=top_k, dim=-1)
    #     torch.gather()

    heatmap = tensor_distance(batch, batch, distance_method)
    return heatmap


def _calculate_score(heatmap: Tensor, method: str = "mean") -> Tensor:
    assert method in ["mean", "std", "max"]
    if method == "mean":
        return torch.mean(heatmap, dim=0) if heatmap.size(dim=0) > 1 else heatmap.squeeze(0)
    elif method == "std":
        return torch.std(heatmap, dim=0) if heatmap.size(dim=0) > 1 else torch.tensor(0)
    else:
        return torch.max(heatmap, dim=0).values if heatmap.size(dim=0) > 1 else heatmap.squeeze(0)


def calculate_score(heatmap: Tensor, section: str = "full", method: str = "mean") -> Optional[Tensor]:
    assert section in ["full", "canon", "random", "no_canon"]
    if not heatmap.shape[0] > 1:
        return None

    distance_per_mol = _calculate_score(heatmap, method)
    if section == "full":
        return _calculate_score(distance_per_mol, method)
    elif section == "canon" and distance_per_mol.shape[0] > 1:
        return distance_per_mol[0]
    elif section == "random" and distance_per_mol.shape[0] > 2:
        return distance_per_mol[1]
    elif section == "no_canon" and distance_per_mol.shape[0] > 2:
        return _calculate_score(distance_per_mol[1:], method)
    else:
        return None
