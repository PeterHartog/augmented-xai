import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class SequencePositionalEncoding(nn.Module):
    """Helper Module that adds positional encoding to sequence embedding."""

    def __init__(self, emb_size: int, dropout: float, maxlen: Optional[int] = None, batch_first: bool = True) -> None:
        super().__init__()
        if not maxlen:
            maxlen = 5000

        self.batch_first = batch_first

        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding: Tensor) -> Tensor:
        batch_idx, token_idx = (0, 1) if self.batch_first else (1, 0)
        if token_embedding.dim == 2:
            token_embedding = token_embedding.unsqueeze(batch_idx)
        seq_len = token_embedding.size(token_idx)

        pos_emb = self.pos_embedding.squeeze()[:seq_len, :]  # type: ignore
        pos_emb = pos_emb.unsqueeze(batch_idx)  # (1, seq_len, emb_size)

        return self.dropout(token_embedding + pos_emb)  # Pos embedding behind buffer


class LearnablePositionalEncoding(nn.Module):
    """Helper Module that adds positional encoding to sequence embedding."""

    def __init__(self, emb_size: int, dropout: float, maxlen: Optional[int] = None, batch_first: bool = True) -> None:
        super().__init__()
        if not maxlen:
            maxlen = 5000

        self.batch_first = batch_first

        self.dropout = nn.Dropout(dropout)
        self.pos_embedding = nn.Embedding(maxlen, emb_size)
        self.pos_embedding = torch.nn.Parameter(torch.randn(maxlen, emb_size))

    def forward(self, token_embedding: Tensor) -> Tensor:
        batch_idx, token_idx = (0, 1) if self.batch_first else (1, 0)
        if token_embedding.dim == 2:
            token_embedding = token_embedding.unsqueeze(batch_idx)

        seq_len = token_embedding.size(token_idx)
        pos_emb = self.pos_embedding.squeeze()[:seq_len, :]  # type: ignore
        pos_emb = pos_emb.unsqueeze(batch_idx)  # (1, seq_len, emb_size)

        return self.dropout(token_embedding + pos_emb)  # Pos embedding behind buffer
