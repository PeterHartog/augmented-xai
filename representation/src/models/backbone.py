"""Encoder model for transformer architecture."""
import collections
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from aidd_codebase.registries import AIDD
from aidd_codebase.utils.typescripts import Tensor
from torchmetrics.functional import (
    accuracy,
    auroc,
    f1_score,
    matthews_corrcoef,
    precision,
    recall,
)

from representation.src.layers.encoder_layer import EncoderLayer
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


@dataclass(unsafe_hash=True)
class EncoderPredictorArguments:
    src_vocab_size: int = 68  # 112
    num_encoder_layers: int = 3
    emb_size: int = 512
    num_heads: int = 8
    num_feedforward: int = 2
    dim_feedforward: int = 512
    norm: str = "layer"
    activation: str = "relu"
    skip_connection: bool = False
    weight_sharing: bool = False
    max_sep_feedforward: bool = False

    emb_dropout: float = 0.1
    enc_dropout: float = 0.1

    output_dim: int = 2

    batch_first: bool = True
    mask_hidden_state: bool = True
    vector_embed: bool = False
    learnable_positional_encoding: bool = False

    lr: float = 1e-4
    weight_decay: float = 0.01
    freeze_encoder: bool = False
    need_attention_weights: bool = True
    average_attention_weights: bool = True

    max_seq_len: Optional[int] = None
    init_method: Optional[str] = "xavier_init"


class EncoderPredictor(pl.LightningModule):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.save_hyperparameters()

        args = self.set_model_kwargs(kwargs)

        # Outputs
        self.training_step_outputs = collections.defaultdict(list)
        self.validation_step_outputs = collections.defaultdict(list)
        self.test_step_outputs = collections.defaultdict(list)
        self.prediction_step_outputs = collections.defaultdict(list)
        self.prediction_outputs = collections.defaultdict(list)

        # Settings
        self.batch_first = args.batch_first
        self.vector_embed = args.vector_embed
        self.output_dim = args.output_dim
        self.num_heads = args.num_heads
        self.mask_hidden_state = args.mask_hidden_state
        self._freeze_encoder = args.freeze_encoder
        self.need_attention_weights = args.need_attention_weights
        self.src_vocab_size = args.src_vocab_size

        # Embedding
        self.src_embed = select_embeddings(
            learnable_positional_encoding=args.learnable_positional_encoding,
            dropout=args.emb_dropout,
            vocab_size=args.src_vocab_size,
            emb_size=args.emb_size,
            max_seq_len=None,  # args.max_seq_len,
        )

        # Model
        encoder_layer = EncoderLayer(
            emb_size=args.emb_size,
            num_heads=args.num_heads,
            dim_feedforward=args.dim_feedforward,
            dropout=args.enc_dropout,
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

        self.sigmoid = nn.Sigmoid()

        self.predictor, self.fc_out = self.init_predictor(args)
        self.model = nn.ModuleList([self.src_embed, self.encoder, self.predictor, self.fc_out])

        self.criterion = nn.BCEWithLogitsLoss()

        self.lr = args.lr
        self.weight_decay = args.weight_decay

        if self._freeze_encoder:
            self.freeze_encoder()

        if args.init_method:
            self.reset_model(args.init_method)

    @abstractmethod
    def set_model_kwargs(self, kwargs) -> EncoderPredictorArguments:
        return set_kwargs(EncoderPredictorArguments, **kwargs)

    @abstractmethod
    def init_predictor(self, args) -> tuple[nn.Module, nn.Module]:
        raise NotImplementedError()

    def set_encoder(self, model: nn.Module) -> None:
        self.src_embed: nn.Module = model.src_embed  # type: ignore
        self.encoder: nn.Module = model.encoder  # type: ignore
        self.model = nn.ModuleList([self.src_embed, self.encoder, self.predictor, self.fc_out])  # type: ignore
        if self._freeze_encoder:
            self.freeze_encoder()

    def freeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.model = nn.ModuleList([self.predictor, self.fc_out])

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
            hidden_state: [batch_size, src_seq_len, emb_size]
        """
        src_embed = self.src_embed(src)
        hidden_state = self.encoder(src_embed, src_mask, src_padding_mask)

        if src_padding_mask is not None and self.mask_hidden_state:
            hidden_state = hidden_state.masked_fill(src_padding_mask.unsqueeze(-1), 0)  # (B, S, E)
        return hidden_state

    def classify(
        self,
        memory: Tensor,
    ) -> Tensor:
        x = self.predictor(memory)
        x = self.fc_out(x)
        return x

    def _step(self, batch, batch_idx):
        """
        Args:
            batch: dict
                src: (batch_size, seq_len)
                src_padding_mask: (batch_size, seq_len)
                y: (batch_size,)

        Returns:
            output: dict
        """
        output = {}
        output["batch_size"] = batch["src"].shape[0 if self.batch_first else 1]
        src, src_pad_mask = batch["src"], batch["src_padding_mask"]
        y = batch["label"]
        y = F.one_hot(y.long(), num_classes=self.output_dim).float() if self.output_dim > 1 else y

        hidden_state = self(src, None, src_pad_mask)

        losses = []
        logits = self.classify(hidden_state).squeeze()
        metric_dict = self._calc_metrics(logits, y)
        loss = self.criterion(logits.squeeze(), y.squeeze())
        losses.append(loss)

        output["loss"] = torch.mean(torch.stack(losses))
        return {**output, **metric_dict}  # type: ignore

    def predict_proba_embed(
        self,
        src_embed: Tensor,
        src_mask: Optional[Tensor] = None,
        src_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        hidden_state = self.encoder(src_embed, src_mask, src_padding_mask)
        logits = self.classify(hidden_state)
        return logits

    def predict_proba(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        src_embed = self.src_embed(src)
        hidden_state = self.encoder(src_embed, src_mask, src_padding_mask)
        logits = self.classify(hidden_state)
        return logits

    def predict(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        logits = self.predict_proba(src, src_mask, src_padding_mask)
        return (
            torch.argmax(self.sigmoid(logits), dim=1).long()
            if self.output_dim != 1
            else torch.round(self.sigmoid(logits)).long()
        )

    def training_step(self, batch, batch_idx):
        output = self._step(batch, batch_idx)
        for k, v in output.items():
            self.training_step_outputs[k].append(v)
        self.log("Batch/Train/Loss", output["loss"], batch_size=output["batch_size"])
        return output

    def on_train_epoch_end(self):
        stacked_loss = [loss.cpu() for loss in self.training_step_outputs["loss"]]
        self.log("Epoch/Train/Loss", torch.stack(stacked_loss).mean())
        self._log_metrics_epoch(self.training_step_outputs, "train")
        self.training_step_outputs.clear()  # free memory

    def validation_step(self, batch, batch_idx):
        output = self._step(batch, batch_idx)
        for k, v in output.items():
            self.validation_step_outputs[k].append(v)
        self.log("Batch/Validation/Loss", output["loss"], batch_size=output["batch_size"])
        return output

    def on_validation_epoch_end(self):
        stacked_loss = [loss.cpu() for loss in self.validation_step_outputs["loss"]]
        self.log("Epoch/Validation/Loss", torch.stack(stacked_loss).mean(), prog_bar=True)
        self._log_metrics_epoch(self.validation_step_outputs, "validation")
        self.validation_step_outputs.clear()  # free memory

    def test_step(self, batch, batch_idx):
        output = self._step(batch, batch_idx)
        for k, v in output.items():
            self.test_step_outputs[k].append(v)
        self.log("Batch/Test/Loss", output["loss"], batch_size=output["batch_size"])
        return output

    def on_test_epoch_end(self):
        stacked_loss = [loss.cpu() for loss in self.test_step_outputs["loss"]]
        self.log("Epoch/Test/Loss", torch.stack(stacked_loss).mean(), prog_bar=True)
        self._log_metrics_epoch(self.test_step_outputs, "test")
        self.test_step_outputs.clear()  # free memory

    # @torch.inference_mode(False)  # important for gradient-based interpretation
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        output = {}
        output["batch_size"] = batch["src"].shape[0 if self.batch_first else 1]
        src, src_pad_mask = batch["src"], batch["src_padding_mask"]
        y = batch["label"]
        y = F.one_hot(y.long(), num_classes=self.output_dim).float()
        logits = self.predict_proba(src, None, src_pad_mask)

        self.prediction_step_outputs["neg_pred"].append(logits[:, 0] if self.output_dim != 1 else 1 - logits)
        self.prediction_step_outputs["pos_pred"].append(logits[:, 1] if self.output_dim != 1 else logits)
        self.prediction_step_outputs["final_pred"].append(
            torch.argmax(self.sigmoid(logits), dim=1).long()
            if self.output_dim != 1
            else torch.round(self.sigmoid(logits)).long()
        )
        # _ = self._step(batch, batch_idx)

    def on_predict_epoch_end(self):
        self.prediction_step_outputs.clear()  # free memory

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=30)
        # CosineAnnealingWarmRestarts(optimizer, 25)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "Batch/Validation/Loss"}
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return {"optimizer": self.optimizer}

    def _calc_metrics(self, logits, y):
        pred = self.sigmoid(logits)
        pred, y = pred.squeeze(), y.squeeze()
        acc = accuracy(pred, y.long(), task="binary")
        au_roc = auroc(pred, y.long(), task="binary")
        f1 = f1_score(pred, y.long(), task="binary")
        mcc = matthews_corrcoef(pred, y.long(), task="binary")
        rec = recall(pred, y.long(), task="binary")
        prec = precision(pred, y.long(), task="binary")

        return {"auroc": au_roc, "acc": acc, "f1": f1, "mcc": mcc, "recall": rec, "precision": prec}

    def _log_metrics_epoch(self, outputs: dict, stage: str) -> None:
        assert set(["auroc", "acc", "f1", "mcc"]).issubset(set(outputs.keys())), "accuracy scores not in outputs"
        self.log(
            f"Epoch/{stage.capitalize()}/AUROC",
            torch.stack([x.cpu().float() for x in outputs["auroc"]]).mean(),
            prog_bar=True,
        )
        self.log(f"Epoch/{stage.capitalize()}/Accuracy", torch.stack([x.cpu().float() for x in outputs["acc"]]).mean())
        self.log(f"Epoch/{stage.capitalize()}/F1", torch.stack([x.cpu().float() for x in outputs["f1"]]).mean())
        self.log(f"Epoch/{stage.capitalize()}/MCC", torch.stack([x.cpu().float() for x in outputs["mcc"]]).mean())
        self.log(f"Epoch/{stage.capitalize()}/Recall", torch.stack([x.cpu().float() for x in outputs["recall"]]).mean())
        self.log(
            f"Epoch/{stage.capitalize()}/Precision", torch.stack([x.cpu().float() for x in outputs["precision"]]).mean()
        )
