"""Decoder model for transformer architecture."""
from typing import Dict, List, Optional, Union

import torch.nn as nn
from aidd_codebase.utils.typescripts import Tensor

from representation.src.utils.clones import _get_clones, _get_duplicates


class TransformerDecoder(nn.Module):
    attn_weights: Optional[Dict[str, List[Tensor]]]

    def __init__(
        self,
        decoder_layer: Union[nn.ModuleList, nn.Module],
        num_decoder_layers: int = 3,
        batch_first: bool = False,
        share_weights: bool = False,
        need_attention_weights: bool = True,
        average_attention_weights: bool = True,
    ):
        super().__init__()

        self.batch_first = batch_first
        self.attn_weights = None
        self.need_attention_weights = need_attention_weights
        self.average_attention_weights = average_attention_weights

        if isinstance(decoder_layer, nn.ModuleList):
            self.layers = decoder_layer
        else:
            self.layers = (
                _get_clones(decoder_layer, num_decoder_layers)
                if not share_weights
                else _get_duplicates(decoder_layer, num_decoder_layers)
            )

    def forward(
        self,
        emb_tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor,
        memory_mask: Tensor,
        tgt_padding_mask: Tensor,
        memory_padding_mask: Tensor,
    ) -> Tensor:
        x = emb_tgt
        attn_weights = []
        cross_weights = []
        for layer in self.layers:
            x, attn_weight, cross_weight = layer(
                x,
                memory,
                tgt_mask,
                memory_mask,
                tgt_padding_mask,
                memory_padding_mask,
            )
            attn_weights.append(attn_weight)
            cross_weights.append(cross_weight)
        if self.need_attention_weights:
            self.attn_weights = {"dec_attn_weights": attn_weights, "cross_attn_weights": cross_weights}
        return x

    def get_attn_weights(self) -> Dict[str, List[Tensor]]:
        if self.attn_weights is None:
            raise ValueError("No attention weights were saved.")
        return self.attn_weights
