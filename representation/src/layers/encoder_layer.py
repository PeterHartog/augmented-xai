"""Encoder layer for transformer model."""

from typing import Optional

import torch.nn as nn
from aidd_codebase.registries import AIDD
from torch import Tensor

from representation.src.modules.cust_attention import CustomMultiheadAttention
from representation.src.modules.mlp import MLP


@AIDD.ModuleRegistry.register(key="encoder_layer")
class EncoderLayer(nn.Module):
    def __init__(
        self,
        emb_size: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        norm: str = "layer",
        activation: str = "relu",
        batch_first: bool = False,
        skip_connection: bool = False,
        num_feedforward: int = 2,
        max_sep_feedforward: bool = False,
        need_attention_weights: bool = True,
        average_attention_weights: bool = True,
    ):
        super().__init__()

        self.batch_first = batch_first
        self.need_attention_weights = need_attention_weights
        self.average_attention_weights = average_attention_weights

        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.self_attn = (
            nn.MultiheadAttention(emb_size, num_heads, dropout=dropout, batch_first=batch_first)
            if not need_attention_weights
            else CustomMultiheadAttention(emb_size, num_heads, dropout=dropout, batch_first=batch_first)
        )

        self.feed_forward = MLP(
            emb_size,
            emb_size,
            dim_feedforward,
            dropout,
            norm,
            activation,
            max_sep_feedforward,
            num_feedforward,
            skip_connection,
        )

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_padding_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        x = src
        self_attn, self_attn_weights = self.self_attn(
            x,
            x,
            x,
            key_padding_mask=src_padding_mask,
            need_weights=self.need_attention_weights,
            attn_mask=src_mask,
            average_attn_weights=self.average_attention_weights,
        )
        x = x + self.dropout1(self_attn)
        x = self.norm1(x)

        x = x + self.dropout2(self.feed_forward(x))
        x = self.norm2(x)
        return x, self_attn_weights
