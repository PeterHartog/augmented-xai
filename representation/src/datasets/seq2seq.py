"""Seq2seq dataset."""
from typing import Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from representation.src.tokenizer.abstract import AbstractTokenizer


class Seq2seqDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        smiles_col: str,
        mapping: dict[str, Union[str, list[str]]],
        tokenizer: AbstractTokenizer,
    ) -> None:
        self.tokenizer = tokenizer

        limit_seq_len = self.tokenizer.max_seq_len - 2
        self.smiles: pd.Series[str] = df[smiles_col][df[smiles_col].apply(len) <= limit_seq_len].dropna()
        self.len = len(self.smiles)

        self.mappings = {k: torch.from_numpy(df[v].to_numpy().astype(np.float32)) for k, v in mapping.items()}

    def __getitem__(self, index):
        src = self.smiles.iloc[index]
        src = self.tokenizer.tensorize(self.tokenizer.tokenize(src))  # type: ignore
        output = {
            "src": src.to(torch.float32),
            "tgt": src.to(torch.float32),
        }
        for k, v in self.mappings.items():
            output[k] = v[index]

        return output

    def __len__(self):
        return self.len
