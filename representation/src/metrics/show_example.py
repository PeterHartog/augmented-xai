from typing import Any

import pandas as pd
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT


class ShowExamplePredictions(Callback):
    """
    Callback to show example predictions from each validation epoch.
    """

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.all_predictions = []

    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        if batch_idx == 0:
            src, tgt = batch["src"][0], batch["tgt"][0]
            logits = outputs["lm_logits"][0]  # type: ignore
            src = pl_module.tokenizer.smile_return(src.detach().cpu())  # type: ignore
            predict = pl_module.tokenizer.smile_return(torch.argmax(logits.detach(), dim=-1).cpu())  # type: ignore
            tgt = pl_module.tokenizer.smile_return(tgt.detach().cpu())  # type: ignore
            self.all_predictions.append([trainer.current_epoch, src, predict, tgt])

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        table = wandb.Table(columns=["Epoch", "Input", "Prediction", "Ground Truth"], data=self.all_predictions)
        pl_module.logger.experiment.log({"Examples": table})  # type: ignore


class SaveTestPredictions(Callback):
    """
    Callback to show example predictions from each validation epoch.
    """

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.all_predictions = []

    def on_test_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        src, src_pad_mask = batch["src"], batch["src_padding_mask"]
        tgt = batch["tgt"][:, 1:]

        hidden_state = pl_module(src, None, src_pad_mask)
        predictions = pl_module.predict(hidden_state, None, src_pad_mask, alg="greedy", k=None)  # type: ignore

        for i in range(predictions.shape[0 if pl_module.batch_first else 1]):
            src_ = pl_module.tokenizer.smile_return(src[i].detach().cpu())  # type: ignore
            predict_ = pl_module.tokenizer.smile_return(predictions[i].detach().cpu())  # type: ignore
            tgt_ = pl_module.tokenizer.smile_return(tgt[i].detach().cpu())  # type: ignore
            self.all_predictions.append([src_, predict_, tgt_])

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        df = pd.DataFrame(self.all_predictions, columns=["Input", "Prediction", "Ground Truth"])
        df.to_csv("/projects/mai/users/kvvb168_peter/molecular-interpretation/quick_test.csv")
