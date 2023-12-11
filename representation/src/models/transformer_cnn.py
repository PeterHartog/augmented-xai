"""Encoder model for transformer architecture."""
from dataclasses import dataclass, field

import torch.nn as nn
from aidd_codebase.registries import AIDD

from representation.src.models.backbone import (
    EncoderPredictor,
    EncoderPredictorArguments,
)
from representation.src.modules.highway_units import _HighwayUnit
from representation.src.modules.textcnn import TextCNN
from representation.src.utils.inspect_kwargs import set_kwargs


@AIDD.ModelRegistry.register_arguments(key="transformer_cnn")
@dataclass(unsafe_hash=True)
class TransformerCNNArguments(EncoderPredictorArguments):
    cnn_dropout: float = 0.1
    kernel_sizes: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20])
    filters: list[int] = field(
        default_factory=lambda: [
            100,
            200,
            200,
            200,
            200,
            100,
            100,
            100,
            100,
            100,
            160,
            160,
        ]
    )
    highway_kernel: int = 1
    stride: int = 1


@AIDD.ModelRegistry.register(key="transformer_cnn", author="Peter Hartog", credit_type="None")
class TransformerCNN(EncoderPredictor):
    def set_model_kwargs(self, kwargs) -> TransformerCNNArguments:
        return set_kwargs(TransformerCNNArguments, **kwargs)

    def init_predictor(self, args: TransformerCNNArguments) -> tuple[nn.Module, nn.Module]:
        if not args.max_seq_len:
            raise ValueError("TransformerCNN requires padding to max and max sequence length.")

        text_cnn = TextCNN(
            kernel_sizes=args.kernel_sizes,
            filters=args.filters,
            stride=args.stride,
            in_size=args.max_seq_len,
            out_size=args.emb_size,
            dropout=args.cnn_dropout,
            batch_first=args.batch_first,
        )

        highway = _HighwayUnit(in_features=args.emb_size, out_features=args.emb_size)
        return nn.Sequential(text_cnn, highway), nn.Linear(args.emb_size, args.output_dim)
