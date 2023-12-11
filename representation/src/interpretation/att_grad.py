from typing import Any, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from aidd_codebase.utils.directories import validate_or_create_full_path
from pytorch_lightning.callbacks import Callback
from torch import Tensor


def select_index(model_output: Tensor, index: Optional[int] = None) -> Tensor:
    if index is None:
        index = np.argmax(model_output.cpu().data.numpy(), axis=-1)
    one_hot = torch.zeros((1, model_output.size()[-1])).to(model_output.device)
    one_hot[0, index] = 1
    return one_hot


class Grads(Callback):
    def __init__(self, save_dir: str = "att_grad") -> None:
        super().__init__()
        self.layer = -1
        self.save_dir = save_dir

    def get_grad(self, all_layer_attentions: list[Tensor], loss: Tensor, layer: int = -1):
        cam = all_layer_attentions[layer]
        cam = torch.autograd.grad(loss, cam, retain_graph=False, create_graph=False)[0].detach()
        cam = cam.mean(1)
        cam = cam.mean(dim=1).squeeze().detach()
        return cam

    def on_predict_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        pd.DataFrame().to_csv(validate_or_create_full_path(f"{self.save_dir}/grad.csv"), header=True, index=False)
        return super().on_predict_start(trainer, pl_module)

    @torch.inference_mode(False)  # important for gradient-based interpretation
    def on_predict_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        src, src_pad_mask = batch["src"], batch["src_padding_mask"]
        y = batch["label"].long().tolist()

        emb_src = pl_module.src_embed(src)  # type: ignore
        logits = pl_module.predict_proba_embed(emb_src, None, src_pad_mask)  # type: ignore
        enc_attn = pl_module.encoder.get_attn_weights()  # type: ignore

        pl_module.zero_grad()
        one_hot_backprop = select_index(logits.detach(), y)
        loss = torch.sum(logits * one_hot_backprop, dtype=logits.dtype)
        grad = self.get_grad(enc_attn["enc_attn_weights"], loss, layer=-1)

        if len(grad.shape) == 1:
            grad = grad.unsqueeze(0)

        grad = [x.numpy().tolist() for x in grad.cpu().unbind(dim=0 if pl_module.batch_first else 1)]
        pd.DataFrame(grad).to_csv(f"{self.save_dir}/grad.csv", mode="a", header=False, index=False)


class AttGrad(Callback):
    def __init__(self, save_dir: str = "att_grad") -> None:
        super().__init__()
        self.layer = -1
        self.save_dir = save_dir

    def attention_gradients(self, all_layer_attentions: list[Tensor], loss: Tensor, start_layer: int = 0):
        cams = all_layer_attentions
        grads = [torch.autograd.grad(loss, cam, retain_graph=True)[0] for cam in cams]

        cams = [grad * cam for grad, cam in zip(grads, cams)]
        cams = [cam.mean(1) for cam in cams]  # average over attention heads
        # cams = [cam / cam.sum(dim=-1, keepdim=True) for cam in cams]
        # grad = grads[0].mean(dim=[1, 2], keepdim=True)
        # cam = (cam * grad).mean(0)

        # Summed Attributions
        cam = torch.stack(cams[start_layer:], dim=0).sum(dim=0)

        cam = cam.mean(1).squeeze().detach()
        return cam

    def on_predict_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        pd.DataFrame().to_csv(validate_or_create_full_path(f"{self.save_dir}/att_grad.csv"), header=True, index=False)
        return super().on_predict_start(trainer, pl_module)

    @torch.inference_mode(False)  # important for gradient-based interpretation
    def on_predict_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        src, src_pad_mask = batch["src"], batch["src_padding_mask"]
        y = batch["label"].long().tolist()

        emb_src = pl_module.src_embed(src)  # type: ignore
        logits = pl_module.predict_proba_embed(emb_src, None, src_pad_mask)  # type: ignore
        enc_attn = pl_module.encoder.get_attn_weights()  # type: ignore

        pl_module.zero_grad()
        one_hot_backprop = select_index(logits.detach(), y)
        loss = torch.sum(logits * one_hot_backprop, dtype=logits.dtype)

        attention_gradients = self.attention_gradients(enc_attn["enc_attn_weights"], loss)

        if len(attention_gradients.shape) == 1:
            attention_gradients = attention_gradients.unsqueeze(0)

        attention_gradients = [
            x.numpy().tolist() for x in attention_gradients.cpu().unbind(dim=0 if pl_module.batch_first else 1)
        ]

        pd.DataFrame(attention_gradients).to_csv(f"{self.save_dir}/att_grad.csv", mode="a", header=False, index=False)
