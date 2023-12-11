"""Encoder Decoder Language model."""
import collections
from dataclasses import dataclass, field
from typing import List, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from aidd_codebase.registries import AIDD
from aidd_codebase.utils.typescripts import Tensor

from representation.src.layers.decoder_layer import DecoderLayer
from representation.src.layers.encoder_layer import EncoderLayer
from representation.src.modules.beamsearch import BeamSearch, GreedySearch
from representation.src.modules.decoder import TransformerDecoder
from representation.src.modules.embedding.embedding import TokenEmbedding
from representation.src.modules.embedding.positional import (
    LearnablePositionalEncoding,
    SequencePositionalEncoding,
)
from representation.src.modules.encoder import TransformerEncoder
from representation.src.tokenizer.abstract import AbstractTokenizer
from representation.src.utils.inspect_kwargs import set_kwargs


def select_embeddings(
    learnable_positional_encoding: bool, dropout: float, vocab_size: int, emb_size: int, max_seq_len: Optional[int]
) -> nn.Module:
    tok_emb = TokenEmbedding(vocab_size=vocab_size, emb_size=emb_size)
    pos_emb = (
        LearnablePositionalEncoding(emb_size=emb_size, dropout=dropout, maxlen=max_seq_len)
        if learnable_positional_encoding
        else SequencePositionalEncoding(emb_size=emb_size, dropout=dropout, maxlen=max_seq_len)
    )
    return nn.Sequential(tok_emb, pos_emb)


@AIDD.ModelRegistry.register_arguments(key="encoder_decoder")
@dataclass(unsafe_hash=True)
class EncoderDecoderArguments:
    src_vocab_size: int = 68  # 112
    tgt_vocab_size: int = 68  # 112
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    emb_size: int = 512
    num_heads: int = 8
    num_feedforward: int = 2
    dim_feedforward: int = 512
    dropout: float = 0.1
    norm: str = "layer"
    activation: str = "relu"

    learnable_positional_encoding: bool = False
    skip_connection: bool = False
    max_sep_feedforward: bool = False
    vector_embed: bool = False

    batch_first: bool = True
    tasks: List = field(default_factory=lambda: ["lm"])
    init_method: str = "xavier_init"
    lr: float = 1e-4
    weight_decay: float = 0.01
    weight_sharing: bool = False
    need_attention_weights: bool = False
    average_attention_weights: bool = False

    max_seq_len: Optional[int] = None


@AIDD.ModelRegistry.register(
    key="encoder_decoder",
    author="Peter Hartog",
    credit_type="None",
)
class EncoderDecoder(pl.LightningModule):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.save_hyperparameters()
        args = set_kwargs(EncoderDecoderArguments, **kwargs)

        # Outputs
        self.training_step_outputs = collections.defaultdict(list)
        self.validation_step_outputs = collections.defaultdict(list)
        self.test_step_outputs = collections.defaultdict(list)
        self.prediction_step_outputs = collections.defaultdict(list)

        # Settings
        self.tasks = args.tasks
        self.batch_first = args.batch_first
        self.vector_embed = args.vector_embed
        self.num_heads = args.num_heads
        self.need_attention_weights = args.need_attention_weights
        self.src_vocab_size = args.src_vocab_size
        self.tgt_vocab_size = args.tgt_vocab_size
        self.max_seq_len = args.max_seq_len

        # Embedding
        self.src_embed = select_embeddings(
            learnable_positional_encoding=args.learnable_positional_encoding,
            dropout=args.dropout,
            vocab_size=args.src_vocab_size,
            emb_size=args.emb_size,
            max_seq_len=args.max_seq_len,
        )
        self.tgt_embed = (
            self.src_embed
            if args.src_vocab_size == args.tgt_vocab_size
            else select_embeddings(
                learnable_positional_encoding=args.learnable_positional_encoding,
                dropout=args.dropout,
                vocab_size=args.tgt_vocab_size,
                emb_size=args.emb_size,
                max_seq_len=args.max_seq_len,
            )
        )

        # Model
        encoder_layer = EncoderLayer(
            emb_size=args.emb_size,
            num_heads=args.num_heads,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            norm=args.norm,
            activation=args.activation,
            batch_first=args.batch_first,
            skip_connection=args.skip_connection,
            num_feedforward=args.num_feedforward,
            max_sep_feedforward=args.max_sep_feedforward,
            need_attention_weights=args.need_attention_weights,
            average_attention_weights=args.average_attention_weights,
        )

        decoder_layer = DecoderLayer(
            emb_size=args.emb_size,
            num_heads=args.num_heads,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            norm=args.norm,
            activation=args.activation,
            batch_first=args.batch_first,
            skip_connection=args.skip_connection,
            num_feedforward=args.num_feedforward,
            max_sep_feedforward=args.max_sep_feedforward,
            need_attention_weights=args.need_attention_weights,
            average_attention_weights=args.average_attention_weights,
        )

        self.encoder = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_encoder_layers=args.num_encoder_layers,
            batch_first=args.batch_first,
            share_weights=args.weight_sharing,
            need_attention_weights=args.need_attention_weights,
            average_attention_weights=args.average_attention_weights,
        )

        self.decoder = TransformerDecoder(
            decoder_layer=decoder_layer,
            num_decoder_layers=args.num_decoder_layers,
            batch_first=args.batch_first,
            share_weights=args.weight_sharing,
            need_attention_weights=args.need_attention_weights,
            average_attention_weights=args.average_attention_weights,
        )

        self.fc_out = nn.Linear(args.emb_size, args.tgt_vocab_size)

        self.model = nn.ModuleList([self.src_embed, self.tgt_embed, self.encoder, self.decoder, self.fc_out])

        self.pad_idx = 0
        self.criterium = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")

        self.lr = args.lr
        self.weight_decay = args.weight_decay

        if args.init_method:
            self.reset_model(args.init_method)

    def reset_model(self, init_method: str) -> None:
        param_init = AIDD.ModuleRegistry.get(key=init_method)
        self.model = param_init(self.model)

    def set_tokenizer(self, tokenizer: AbstractTokenizer) -> None:
        self.tokenizer = tokenizer

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            src: [batch_size, src_seq_len]
            src_mask: [batch_size, src_seq_len, src_seq_len]
            src_padding_mask: [batch_size, src_seq_len]

        Returns:
            hidden_state: [batch_size, src_seq_len, emb_size] | [batch_size, 1, emb_size]
        """
        src_embed = self.src_embed(src)
        hidden_state = self.encoder(src_embed, src_mask, src_padding_mask)
        if self.vector_embed:
            assert src_padding_mask is not None, "padding mask must be there when using vector embedding"
            hidden_state = hidden_state.masked_fill(src_padding_mask.unsqueeze(-1), 0)  # (B, S, E)
            hidden_state = hidden_state.sum(dim=1 if self.batch_first else 0, keepdim=True)  # (B, 1, E)
        return hidden_state

    def lm(
        self,
        memory: Tensor,
        tgt: Tensor,
        memory_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
        memory_padding_mask: Optional[Tensor] = None,
        tgt_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        tgt_emb = self.tgt_embed(tgt)
        memory_mask = memory_mask if not self.vector_embed else None
        memory_padding_mask = memory_padding_mask if not self.vector_embed else None
        decoder_out = self.decoder(tgt_emb, memory, tgt_mask, memory_mask, tgt_padding_mask, memory_padding_mask)
        logits = self.fc_out(decoder_out)
        return logits

    def _step(self, batch, batch_idx):
        """
        Args:
            batch: dict
                src: (batch_size, seq_len)
                src_padding_mask: (batch_size, seq_len)
                tgt: (batch_size, seq_len)
                tgt_padding_mask: (batch_size, seq_len)
        Returns:
            output: dict
        """
        output = {}

        output["batch_size"] = batch["src"].shape[0 if self.batch_first else 1]
        src, src_pad_mask = batch["src"], batch["src_padding_mask"]

        hidden_state = self(src, None, src_pad_mask)

        losses = []
        if "lm" in self.tasks:
            tgt, tgt_mask, tgt_pad_mask = batch["tgt"], batch["tgt_mask"], batch["tgt_padding_mask"]
            output["lm_logits"] = self.lm(
                hidden_state, tgt[:, :-1], None, tgt_mask[:-1, :-1], src_pad_mask, tgt_pad_mask[:, :-1]
            )
            loss = self.criterium(output["lm_logits"].transpose(1, 2), tgt[:, 1:].long()).sum(dim=1).mean()
            losses.append(loss)

        output["loss"] = torch.mean(torch.stack(losses))
        return output

    def training_step(self, batch, batch_idx):
        output = self._step(batch, batch_idx)
        self.training_step_outputs["loss"].append(output["loss"].detach().cpu())
        self.log("Batch/Train/Loss", output["loss"], batch_size=output["batch_size"], prog_bar=True)
        return output

    def on_train_epoch_end(self):
        stacked_loss = [loss.cpu() for loss in self.training_step_outputs["loss"]]
        self.log("Epoch/Train/Loss", torch.stack(stacked_loss).mean(), prog_bar=True)
        self.training_step_outputs.clear()  # free memory

    def validation_step(self, batch, batch_idx):
        output = self._step(batch, batch_idx)
        self.validation_step_outputs["loss"].append(output["loss"].detach().cpu())
        self.log("Batch/Validation/Loss", output["loss"], batch_size=output["batch_size"], prog_bar=True)
        return output

    def on_validation_epoch_end(self):
        stacked_loss = [loss.cpu() for loss in self.validation_step_outputs["loss"]]
        self.log("Epoch/Validation/Loss", torch.stack(stacked_loss).mean(), prog_bar=True)
        self.validation_step_outputs.clear()  # free memory

    def test_step(self, batch, batch_idx):
        output = self._step(batch, batch_idx)
        self.test_step_outputs["loss"].append(output["loss"].detach().cpu())
        self.log("Batch/Test/Loss", output["loss"], batch_size=output["batch_size"], prog_bar=True)
        return output

    def on_test_epoch_end(self):
        stacked_loss = [loss.cpu() for loss in self.test_step_outputs["loss"]]
        self.log("Epoch/Test/Loss", torch.stack(stacked_loss).mean(), prog_bar=True)
        self.test_step_outputs.clear()  # free memory

    def predict(
        self,
        memory: Tensor,
        memory_mask: Optional[Tensor] = None,
        memory_padding_mask: Optional[Tensor] = None,
        alg: str = "greedy",
        k: Optional[int] = None,
    ) -> Tensor:
        assert alg in ["greedy", "beam"], "algorithm choice must be greedy or beam"
        if alg == "beam":
            assert k is not None, "k must be defined if algorithm is beam"

        memory_mask = memory_mask if not self.vector_embed else None
        memory_padding_mask = memory_padding_mask if not self.vector_embed else None

        searcher = GreedySearch if alg == "greedy" else BeamSearch
        searcher = searcher(
            embed=self.tgt_embed,
            decoder=self.decoder,
            fc_out=self.fc_out,
            max_seq_len=self.tokenizer.max_seq_len,
            batch_first=self.batch_first,
            device=self.device,  # type: ignore
            pad_idx=self.tokenizer.pad_idx,
            bos_idx=self.tokenizer.bos_idx,
            eos_idx=self.tokenizer.eos_idx,
            k=k,
        )
        logits = searcher.search(memory=memory, memory_mask=memory_mask, memory_padding_mask=memory_padding_mask)
        return logits

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return {"optimizer": self.optimizer}
