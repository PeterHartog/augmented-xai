"""Decoder layer for transformer model."""

from typing import Optional

import torch.nn as nn
from torch import Tensor

from representation.src.modules.mlp import MLP


class DecoderLayer(nn.Module):
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
        self.norm3 = nn.LayerNorm(emb_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.self_attn = nn.MultiheadAttention(emb_size, num_heads, dropout=dropout, batch_first=batch_first)
        self.cross_attn = nn.MultiheadAttention(emb_size, num_heads, dropout=dropout, batch_first=batch_first)

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
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_padding_mask: Optional[Tensor] = None,
        memory_padding_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        x = tgt
        self_attn, self_attn_weights = self.self_attn(
            x,
            x,
            x,
            key_padding_mask=tgt_padding_mask,
            attn_mask=tgt_mask,
            need_weights=self.need_attention_weights,
            average_attn_weights=self.average_attention_weights,
        )
        x = x + self.dropout1(self_attn)
        x = self.norm1(x)

        cross_attn, cross_attn_weights = self.cross_attn(
            x,
            memory,
            memory,
            key_padding_mask=memory_padding_mask,
            attn_mask=memory_mask,
            need_weights=self.need_attention_weights,
            average_attn_weights=self.average_attention_weights,
        )
        x = x + self.dropout2(cross_attn)
        x = self.norm2(x)

        x = x + self.dropout3(self.feed_forward(x))
        x = self.norm3(x)
        return x, self_attn_weights, cross_attn_weights
