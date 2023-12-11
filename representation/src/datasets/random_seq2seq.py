"""Randomized Seq2seq dataset."""
from typing import Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from representation.src.smiles_utils.randomization import randomize_smiles
from representation.src.tokenizer.abstract import AbstractTokenizer


class RandomSeq2seqDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        smiles_col: str,
        mapping: dict[str, Union[str, list[str]]],
        tokenizer: AbstractTokenizer,
        random_type: str = "unrestricted",
    ) -> None:
        self.tokenizer = tokenizer

        self.random_type = random_type

        self.limit_seq_len = self.tokenizer.max_seq_len - 2
        self.smiles: pd.Series[str] = df[smiles_col][df[smiles_col].apply(len) <= self.limit_seq_len].dropna()
        self.len = len(self.smiles)

        self.mappings = {k: torch.from_numpy(df[v].to_numpy().astype(np.float32)) for k, v in mapping.items()}

    def __getitem__(self, index):
        smi = self.smiles.iloc[index]
        src_smi = randomize_smiles(smi, random_type=self.random_type, max_seq_len=self.limit_seq_len)
        src_token = self.tokenizer.tensorize(self.tokenizer.tokenize(src_smi[0]))
        src = src_token.to(torch.float32)
        tgt_token = self.tokenizer.tensorize(self.tokenizer.tokenize(smi))
        tgt = tgt_token.to(torch.float32)

        output = {"src": src, "tgt": tgt}
        for k, v in self.mappings.items():
            output[k] = v[index]

        return output

    def __len__(self):
        return self.len
