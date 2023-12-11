"""Encoder model for transformer architecture."""
from typing import Dict, List, Optional, Union

import torch.nn as nn
from aidd_codebase.utils.typescripts import Tensor

from representation.src.utils.clones import _get_clones, _get_duplicates


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer: Union[nn.ModuleList, nn.Module],
        num_encoder_layers: int = 3,
        batch_first: bool = False,
        share_weights: bool = False,
        need_attention_weights: bool = True,
        average_attention_weights: bool = True,
    ):
        super().__init__()

        self.batch_first = batch_first
        self.attn_weights: Optional[Dict[str, List[Tensor]]] = None
        self.encodings: Optional[Dict[str, List[Tensor]]] = None

        self.save_attention = need_attention_weights
        self.save_encodings = need_attention_weights
        self.average_attention_weights = average_attention_weights

        if isinstance(encoder_layer, nn.ModuleList):
            self.layers = encoder_layer
        else:
            self.layers = (
                _get_clones(encoder_layer, num_encoder_layers)
                if not share_weights
                else _get_duplicates(encoder_layer, num_encoder_layers)
            )

    def forward(
        self,
        embed_src: Tensor,
        src_mask: Optional[Tensor] = None,
        pad_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = embed_src
        self.attn_weights, self.encodings = None, None
        attn_weights, encodings = [], []
        for layer in self.layers:
            x, attn_weight = layer(x, src_mask, pad_mask)
            attn_weights.append(attn_weight)
            encodings.append(x)
        if self.save_attention:
            self.attn_weights = {"enc_attn_weights": attn_weights}
        if self.save_encodings:
            self.encodings = {"enc_attn_encodings": encodings}

        return x

    def get_attn_weights(self) -> Dict[str, List[Tensor]]:
        if self.attn_weights is None:
            raise ValueError("No attention weights were saved.")
        return self.attn_weights

    def get_attn_encodings(self) -> Dict[str, List[Tensor]]:
        if self.encodings is None:
            raise ValueError("No attention weights were saved.")
        return self.encodings

    # def forward(
    #     self,
    #     embed_src: Tensor,
    #     src_mask: Optional[Tensor] = None,
    #     pad_mask: Optional[Tensor] = None,
    # ) -> Tensor:
    #     x = embed_src
    #     attn_weights = []
    #     for layer in self.layers:
    #         x, attn_weight = layer(x, src_mask, pad_mask)
    #         attn_weights.append(attn_weight)  # .detach())
    #     if self.need_attention_weights:
    #         self.attn_weights = {"enc_attn_weights": attn_weights}
    #     return x
    #     #         attn_weights.append(attn_weight)
    #     #     encodings.append(x)
    #     # if self.save_attn:
    #     #     self.attn_weights = {"enc_attn_weights": attn_weights}
    #     # if self.save_encoding:
    #     #     self.encodings = {"enc_attn_encodings": encodings}

    # def get_attn_weights(self) -> Dict[str, List[Tensor]]:
    #     if self.attn_weights is None:
    #         raise ValueError("No attention weights were saved.")
    #     return self.attn_weights
