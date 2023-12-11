import numpy as np
import pandas as pd
import torch
from torch import Tensor

# from representation.src.analysis.reorder import reorder_attributions


def gather_sorted_attributions(df: pd.DataFrame, attr: Tensor, smile_idx: int) -> Tensor:
    smile_len = df["smile_lenths"].values[smile_idx] + 1
    sorted_ids = df["sorted_indices"].values[smile_idx]

    attr = attr if len(attr.shape) == 2 else attr.sum(-1).squeeze()
    attr = attr[smile_idx, 1:smile_len].squeeze()

    sorted_imp = torch.index_select(attr, 0, torch.tensor(sorted_ids))
    return sorted_imp


def gather_sorted_alerts(df: pd.DataFrame, attr: Tensor, smile_idx: int) -> Tensor:
    smile_len = df["smile_lenths"].values[smile_idx] + 1
    sorted_ids = df["sorted_indices"].values[smile_idx]
    hit_ids = df["alert_hits"].values[smile_idx]

    sorted_ids = [i for i in sorted_ids if i in hit_ids]

    if not sorted_ids:
        return torch.tensor([0])

    attr = attr if len(attr.shape) == 2 else attr.sum(-1).squeeze()
    attr = attr[smile_idx, 1:smile_len].squeeze()

    sorted_imp = torch.index_select(attr, 0, torch.tensor(sorted_ids))
    return sorted_imp


def gather_batches(
    df: pd.DataFrame, attr: Tensor, id_col: str, attr_type: str = "atom", scale: bool = False
) -> dict[int, Tensor]:
    assert attr_type in ["full", "smile", "atom", "alert"]
    id_dict = {}
    for id in df.loc[:, id_col].unique():
        identifier: list[bool] = df.loc[:, id_col].eq(id).tolist()
        if attr_type == "full":
            id_dict[id] = [attr[smile_idx, :] for smile_idx in np.where(identifier)[0]]
        elif attr_type == "smile":
            id_dict[id] = [
                attr[smile_idx, 1:].squeeze()[: df["smile_lenths"].values[smile_idx]]
                for smile_idx in np.where(identifier)[0]
            ]
        elif attr_type == "atom":
            id_dict[id] = [gather_sorted_attributions(df, attr, smile_idx) for smile_idx in np.where(identifier)[0]]
        elif attr_type == "alert":
            id_dict[id] = [gather_sorted_alerts(df, attr, smile_idx) for smile_idx in np.where(identifier)[0]]
        else:
            raise ValueError(f"{attr_type} not found.")

    id_dict = (
        {id: torch.stack(imp, dim=0).view(len(imp), -1) for id, imp in id_dict.items()}
        if not attr_type == "smile"
        else id_dict
    )

    return id_dict
