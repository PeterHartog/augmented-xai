from typing import Any

import pandas as pd
import pytorch_lightning as pl
import torch
from aidd_codebase.utils.directories import validate_or_create_full_path
from pytorch_lightning.callbacks import Callback


class SaveData(Callback):
    def __init__(
        self,
        save_dir: str = "saved_data",
        ignore_cols: list[str] = ["tgt", "src_mask", "tgt_mask", "src_padding_mask", "tgt_padding_mask"],
    ) -> None:
        super().__init__()
        self.save_dir = save_dir
        self.ignore_mask = ignore_cols
        self.init = True

    def init_file(self, columns: list[str]) -> None:
        pd.DataFrame(columns=columns).to_csv(
            validate_or_create_full_path(f"{self.save_dir}/prediction_data.csv"), header=True, index=False
        )
        self.init = False

    def on_predict_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        batch = {
            k: [x.numpy().tolist() for x in v.cpu().unbind(dim=0)]
            for k, v in batch.items()
            if k not in self.ignore_mask
        }
        step_outputs = {
            k: x.numpy().tolist()
            for k, v in pl_module.prediction_step_outputs.items()  # type: ignore
            for x in torch.stack(v).cpu().unbind(dim=0)
            if k not in self.ignore_mask
        }
        batch["src"] = [pl_module.tokenizer.smile_return(torch.tensor(x)) for x in batch["src"]]  # type: ignore
        batch.update(step_outputs)

        pl_module.prediction_step_outputs.clear()  # type: ignore

        if self.init:
            self.init_file(columns=[k for k in batch.keys()])

        pd.DataFrame(batch).to_csv(
            validate_or_create_full_path(f"{self.save_dir}/prediction_data.csv"), mode="a", header=False, index=False
        )
