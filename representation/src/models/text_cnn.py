# TODO
# from typing import Dict
from dataclasses import dataclass

import pytorch_lightning as pl
import torch
import torch.nn as nn
from aidd_codebase.registries import AIDD

# Metrics
from torchmetrics.classification import AUROC

# Layers
from representation.src.modules.mlp import MLP


@AIDD.ModelRegistry.register_arguments(key="mlp")
@dataclass(unsafe_hash=True)
class TextCNNArguments:
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
class LightningTextCNN(pl.LightningModule):
    def __init__(self):  # , model_args: Dict):
        super().__init__()

        self.save_hyperparameters()

        # args = MLPArguments(**model_args)
        args = TextCNNArguments()

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

        self.criterion = nn.BCEWithLogitsLoss()

        if args.init_method:
            self.reset_model(args.init_method)

    def reset_model(self, init_method: str) -> None:
        param_init = AIDD.ModuleRegistry.get(key=init_method)
        self.model = param_init(self.model)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.criterion(self(x).squeeze(), y)
        self.log("batch/train_loss", loss, batch_size=len(batch))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.criterion(self(x).squeeze(), y)
        self.log("batch/validation_loss", loss, batch_size=len(batch))
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        loss = self.criterion(self(x).squeeze(), y)
        output = {"test_loss": loss, "pred": torch.round(self(x)), "y": y}
        self.log("batch/test_loss", loss, batch_size=len(batch))
        return output

    def test_epoch_end(self, outputs):
        auroc = AUROC(task="binary", thresholds=None)
        auroc_score = auroc(torch.cat([x["pred"] for x in outputs]), torch.cat([x["y"] for x in outputs]))
        self.log("epoch/test_auroc", auroc_score)
        self.log("epoch/test_loss", torch.stack([x["test_loss"] for x in outputs]).mean())

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=5)
        self.monitor = "batch/validation_loss"
        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler, "monitor": self.monitor}
