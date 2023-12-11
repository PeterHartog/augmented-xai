# from typing import Dict
from dataclasses import dataclass

import pytorch_lightning as pl
import torch
import torch.nn as nn
from aidd_codebase.registries import AIDD

# Metrics
from torchmetrics.classification import BinaryAUROC

# Layers
from representation.src.modules.mlp import MLP
from representation.src.utils.inspect_kwargs import set_kwargs


@AIDD.ModelRegistry.register_arguments(key="mlp")
@dataclass(unsafe_hash=True)
class MLPArguments:
    input_dim: int = 1024
    output_dim: int = 1

    name: str = "mlp"

    hidden_dim: int = 128
    dropout: float = 0.2
    norm: str = "layer"
    act_fn: str = "relu"
    max_sep: bool = False
    num_layers: int = 3
    skip: bool = True
    lr: float = 1e-3
    weight_decay: float = 0.0
    init_method: str = "xavier_init"


@AIDD.ModelRegistry.register(key="mlp")
class LightningMLP(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

        self.save_hyperparameters()

        args = set_kwargs(MLPArguments, **kwargs)

        self.lr = args.lr
        self.weight_decay = args.weight_decay

        self.model = MLP(
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            norm=args.norm,
            act_fn=args.act_fn,
            max_sep=args.max_sep,
            num_layers=args.num_layers,
            skip=args.skip,
        )

        # self.sigmoid = nn.Sigmoid()

        self.criterion = nn.BCEWithLogitsLoss()

        if args.init_method:
            self.reset_model(args.init_method)

    def reset_model(self, init_method: str) -> None:
        param_init = AIDD.ModuleRegistry.get(key=init_method)
        self.model = param_init(self.model)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["X"], batch["y"]
        batch_size = len(batch)

        logits = self(x)

        loss = self.criterion(logits.squeeze(), y)
        outputs = {"loss": loss, "pred": logits, "y": y}
        self.log("batch/train_loss", loss, batch_size=batch_size)
        return outputs

    def validation_step(self, batch, batch_idx):
        x, y = batch["X"], batch["y"]
        batch_size = len(batch)

        logits = self(x)

        loss = self.criterion(logits.squeeze(), y)
        outputs = {"valid_loss": loss, "pred": logits, "y": y, "batch_size": batch_size}
        self.log("batch/valid_loss", loss, batch_size=batch_size)
        return outputs

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        # if outputs is not a list, it means we are using 1 GPU
        if not isinstance(outputs, list):
            outputs = [outputs]
        auroc = BinaryAUROC(thresholds=None)
        auroc_score = auroc(torch.cat([x["pred"] for x in outputs]), torch.cat([x["y"] for x in outputs]))
        self.log("batch/valid_auroc", auroc_score, prog_bar=True)

    def validation_epoch_end(self, outputs):
        auroc = BinaryAUROC(thresholds=None)
        auroc_score = auroc(torch.cat([x["pred"] for x in outputs]), torch.cat([x["y"] for x in outputs]))
        self.log("epoch/valid_auroc", auroc_score, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch["X"], batch["y"]
        batch_size = len(batch)

        logits = self(x)

        loss = self.criterion(logits.squeeze(), y)
        outputs = {"test_loss": loss, "pred": logits, "y": y}
        self.log("batch/test_loss", loss, batch_size=batch_size)
        return outputs

    def test_epoch_end(self, outputs):
        auroc = BinaryAUROC(thresholds=None)
        auroc_score = auroc(torch.cat([x["pred"] for x in outputs]), torch.cat([x["y"] for x in outputs]))
        self.log("epoch/test_auroc", auroc_score)
        self.log("epoch/test_loss", torch.stack([x["test_loss"] for x in outputs]).mean())

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=5)
        self.monitor = "batch/valid_loss"
        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler, "monitor": self.monitor}

    # def training_step(self, batch, batch_idx):
    #     x, y = batch["X"], batch["y"]
    #     probs = self(x)
    #     loss = self.criterion(probs.squeeze(), y)
    #     output = {"loss": loss, "pred": torch.round(probs), "y": y}
    #     self.log("batch/train_loss", loss, batch_size=len(batch))
    #     self.log("batch/train_auroc", AUROC(task="binary", thresholds=None)(torch.round(probs), y))
    #     return output

    # def validation_step(self, batch, batch_idx):
    #     x, y = batch["X"], batch["y"]
    #     probs = self(x)
    #     loss = self.criterion(probs.squeeze(), y)
    #     output = {"valid_loss": loss, "pred": torch.round(probs), "y": y}
    #     self.log("batch/valid_loss", loss, batch_size=len(batch))
    #     self.log("batch/valid_auroc", AUROC(task="binary", thresholds=None)(torch.round(probs), y))
    #     return output

    # def test_step(self, batch, batch_idx):
    #     x, y = batch["X"], batch["y"]
    #     probs = self(x)
    #     loss = self.criterion(probs.squeeze(), y)
    #     output = {"test_loss": loss, "pred": torch.round(probs), "y": y}
    #     self.log("batch/test_loss", loss, batch_size=len(batch))
    #     self.log("batch/test_auroc", AUROC(task="binary", thresholds=None)(torch.round(probs), y))
    #     return output

    # def test_epoch_end(self, outputs):
    #     auroc = AUROC(task="binary", thresholds=None)
    #     auroc_score = auroc(torch.cat([x["pred"] for x in outputs]), torch.cat([x["y"] for x in outputs]))
    #     self.log("epoch/test_auroc", auroc_score)
    #     self.log("epoch/test_loss", torch.stack([x["test_loss"] for x in outputs]).mean())

    # def configure_optimizers(self):
    #     self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    #     self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
    # mode="min", factor=0.5, patience=5)
    #     self.monitor = "batch/valid_loss"
    #     return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler, "monitor": self.monitor}
