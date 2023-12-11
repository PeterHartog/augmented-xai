from typing import Optional

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch import Tensor


def get_canon_order(smi: str) -> list[int]:
    return list(Chem.CanonicalRankAtoms(Chem.MolFromSmiles(smi)))  # type: ignore


def get_chars(smi: str) -> list[str]:
    return [at.GetSymbol() for at in Chem.MolFromSmiles(smi).GetAtoms()]  # type: ignore


def get_char_indices(smi: str) -> list[int]:
    return [i for i, char in enumerate(smi) if char.isalpha()]


def corrected_char_indices(smi: str) -> list[list[int]]:
    char_idx = get_char_indices(smi)

    char_corrected_idx = []
    for symbol in get_chars(smi):
        sym_len = len(symbol)
        char_corrected_idx.append(char_idx[:sym_len])
        char_idx = char_idx[sym_len:]
    return char_corrected_idx


def sorted_indices(smi):
    sorted_idx = [list(a) for a in zip(corrected_char_indices(smi), get_canon_order(smi))]
    sorted_idx = sorted(sorted_idx, key=lambda x: x[1])
    sorted_idx = sum([i for i, _ in sorted_idx], [])  # unlist
    return sorted_idx


def set_positive_patterns(alerts: pd.DataFrame) -> list:
    patterns = [Chem.MolFromSmarts(p) for p in alerts.smarts.values]  # type: ignore
    return patterns


def find_struct_alerts(smiles: str, included_smarts: list, excluded_smarts: Optional[list] = None):
    m = Chem.MolFromSmiles(smiles)  # type: ignore
    atom_matches: list[tuple] = []
    for pattern in included_smarts:
        atom_matches.append(m.GetSubstructMatches(pattern))
    return list(set(sum(list(sum(atom_matches, ())), ())))


def atom_to_smile_indices(smi: str, atom_idx: list[int]) -> list[int]:
    char_idx = corrected_char_indices(smi)
    return sum([char_idx[i] for i in atom_idx], [])


def reorder_smi(smi: str, order: list[int]) -> str:
    sorted_smi = "".join([smi[i] for i in order])
    return sorted_smi


def reorder_attributions(attributions: Tensor, order: np.ndarray, full: bool = True) -> Tensor:
    if full:
        ind_tensor = torch.tensor([i for i in order]).repeat(attributions.shape[1], 1).transpose(0, 1)
        sorted_attr = torch.gather(attributions, 0, ind_tensor)
    else:
        sorted_attr = torch.zeros_like(attributions).scatter_(0, torch.tensor([i for i in order]), attributions)
    return sorted_attr


def parse_data(df: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
    df["smile_lenths"] = df.loc[:, smiles_col].apply(len)
    df["sorted_indices"] = df.loc[:, smiles_col].apply(sorted_indices)
    df["atom_indices"] = df.loc[:, smiles_col].apply(get_canon_order)
    df["character_idices"] = df.loc[:, smiles_col].apply(get_char_indices)

    return df


def parse_hits(df: pd.DataFrame, smiles_col: str, alerts: pd.DataFrame) -> pd.DataFrame:
    positive_patterns = set_positive_patterns(alerts)

    df["alert_hits"] = df.loc[:, smiles_col].apply(
        lambda x: atom_to_smile_indices(x, find_struct_alerts(x, positive_patterns))
    )

    return df
