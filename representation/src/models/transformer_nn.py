"""Encoder model for transformer architecture."""
from dataclasses import dataclass

import torch.nn as nn
from aidd_codebase.registries import AIDD

from representation.src.models.backbone import (
    EncoderPredictor,
    EncoderPredictorArguments,
)
from representation.src.modules.embedding_adaptor import EmbeddingAdaptor
from representation.src.modules.mlp import MLP
from representation.src.utils.inspect_kwargs import set_kwargs


@AIDD.ModelRegistry.register_arguments(key="transformer_nn")
@dataclass(unsafe_hash=True)
class TransformerNNArguments(EncoderPredictorArguments):
    adaption: str = "sum"
    adapt_embedding: bool = False

    dim_pred_hidden: int = 512
    dim_pred_final: int = 512
    nn_dropout: float = 0.1


@AIDD.ModelRegistry.register(key="transformer_nn", author="Peter Hartog", credit_type="None")
class TransformerNN(EncoderPredictor):
    def set_model_kwargs(self, kwargs) -> TransformerNNArguments:
        return set_kwargs(TransformerNNArguments, **kwargs)

    def init_predictor(self, args: TransformerNNArguments) -> tuple[nn.Module, nn.Module]:
        assert args.max_seq_len is not None

        if args.adaption == "flatten":
            adapter = nn.Flatten()
            input_dim = args.emb_size * args.max_seq_len
        else:
            adapter = EmbeddingAdaptor(
                args.adaption,
                args.adapt_embedding,
                args.batch_first,
                switch_dims=False if args.adapt_embedding else True,
                in_size=args.emb_size if args.adapt_embedding else args.max_seq_len,
            )
            input_dim = args.max_seq_len if args.adapt_embedding else args.emb_size

        mlp = MLP(
            input_dim=input_dim,
            hidden_dim=args.dim_pred_hidden,
            output_dim=args.dim_pred_final,
            dropout=args.nn_dropout,
        )
        return nn.Sequential(adapter, mlp), nn.Linear(args.dim_pred_final, args.output_dim)
