from typing import Any

import pandas as pd
import pytorch_lightning as pl
import torch
from aidd_codebase.utils.directories import validate_or_create_full_path
from pytorch_lightning.callbacks import Callback
from torch import Tensor


class AttentionMap(Callback):
    def __init__(self, save_dir: str = "attention") -> None:
        super().__init__()
        self.layer = -1
        self.save_dir = save_dir

    def attention_map(self, all_layer_attentions: list[Tensor], layer: int = -1):
        cam = all_layer_attentions[layer]
        cam = cam.mean(1)  # average over attention head
        cam = cam.mean(dim=1).squeeze()
        return cam

    def on_predict_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        pd.DataFrame().to_csv(
            validate_or_create_full_path(f"{self.save_dir}/attention_maps.csv"), header=True, index=False
        )
        return super().on_predict_start(trainer, pl_module)

    def on_predict_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        torch.set_grad_enabled(False)
        enc_attn = pl_module.encoder.get_attn_weights()  # type: ignore
        attention_maps = self.attention_map(enc_attn["enc_attn_weights"])

        if len(attention_maps.shape) == 1:
            attention_maps = attention_maps.unsqueeze(0)

        attention_maps = [
            x.numpy().tolist() for x in attention_maps.cpu().unbind(dim=0 if pl_module.batch_first else 1)
        ]

        pd.DataFrame(attention_maps).to_csv(f"{self.save_dir}/attention_maps.csv", mode="a", header=False, index=False)


class Summed(Callback):
    def __init__(self, include_baseline: bool = True, save_dir: str = "attention") -> None:
        super().__init__()
        self.layer = -1
        self.save_dir = save_dir

    def summed_attention_map(self, all_layer_attentions: list[Tensor], start_layer: int = 0) -> Tensor:
        cams = all_layer_attentions
        cams = [cam.mean(1) for cam in cams]  # average over attention heads
        cams = [cam / cam.sum(dim=-1, keepdim=True) for cam in cams]
        cam = torch.stack(cams[start_layer:], dim=0).sum(dim=0)
        cam = cam.mean(1).squeeze().detach()
        return cam

    def on_predict_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        pd.DataFrame().to_csv(validate_or_create_full_path(f"{self.save_dir}/summed.csv"), header=True, index=False)
        return super().on_predict_start(trainer, pl_module)

    def on_predict_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        torch.set_grad_enabled(False)
        enc_attn = pl_module.encoder.get_attn_weights()  # type: ignore
        rollout = self.summed_attention_map(enc_attn["enc_attn_weights"])

        if len(rollout.shape) == 1:
            rollout = rollout.unsqueeze(0)

        rollout = [x.numpy().tolist() for x in rollout.cpu().unbind(dim=0)]

        pd.DataFrame(rollout).to_csv(
            validate_or_create_full_path(f"{self.save_dir}/summed.csv"), mode="a", header=False, index=False
        )


class Rollout(Callback):
    def __init__(self, include_baseline: bool = True, save_dir: str = "attention") -> None:
        super().__init__()
        self.layer = -1
        self.save_dir = save_dir

    def rollout_attention_map(self, all_layer_attentions: list[Tensor], start_layer: int = 0) -> Tensor:
        # adding residual consideration- code adapted from https://github.com/samiraabnar/attention_flow
        cams = all_layer_attentions
        cams = [cam.mean(1) for cam in cams]  # average over attention heads
        num_tokens = cams[0].shape[1]
        batch_size = cams[0].shape[0]

        eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(cams[0].device)
        cams = [cam + eye for cam in cams]
        cams = [cam / cam.sum(dim=-1, keepdim=True) for cam in cams]
        rollout = cams[start_layer]
        for cam in cams[start_layer:]:
            rollout = cam.bmm(rollout)
        rollout = rollout.mean(dim=1).squeeze()
        return rollout

    def on_predict_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        pd.DataFrame().to_csv(validate_or_create_full_path(f"{self.save_dir}/rollout.csv"), header=True, index=False)
        return super().on_predict_start(trainer, pl_module)

    def on_predict_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        torch.set_grad_enabled(False)
        enc_attn = pl_module.encoder.get_attn_weights()  # type: ignore
        rollout = self.rollout_attention_map(enc_attn["enc_attn_weights"])

        if len(rollout.shape) == 1:
            rollout = rollout.unsqueeze(0)

        rollout = [x.numpy().tolist() for x in rollout.cpu().unbind(dim=0)]

        pd.DataFrame(rollout).to_csv(
            validate_or_create_full_path(f"{self.save_dir}/rollout.csv"), mode="a", header=False, index=False
        )
