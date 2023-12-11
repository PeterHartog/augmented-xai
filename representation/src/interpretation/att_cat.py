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


class Cat(Callback):
    def __init__(self, save_dir: str = "att_grad") -> None:
        super().__init__()
        self.layer = -1
        self.save_dir = save_dir

    def encoding_gradients(self, encodings: list[Tensor], loss: Tensor, start_layer: int = 0):
        cams = encodings
        grads = [torch.autograd.grad(loss, cam, retain_graph=True)[0] for cam in cams]

        cams = [torch.mul(grad, cam) for grad, cam in zip(grads, cams)]
        # cams = [cam / cam.sum(dim=-1, keepdim=True) for cam in cams]
        # grad = grad.mean(dim=[1, 2], keepdim=True)
        # cam = (cam * grad).mean(0)

        # Summed Attributions
        cam = torch.stack(cams[start_layer:], dim=0).sum(dim=0)

        cam = cam.mean(2).squeeze().detach()
        return cam

    def on_predict_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        pd.DataFrame().to_csv(validate_or_create_full_path(f"{self.save_dir}/cat.csv"), header=True, index=False)
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
        enc_encodings = pl_module.encoder.get_attn_encodings()  # type: ignore

        pl_module.zero_grad()
        one_hot_backprop = select_index(logits.detach(), y)
        loss = torch.sum(logits * one_hot_backprop, dtype=logits.dtype)

        encoding_gradients = self.encoding_gradients(enc_encodings["enc_attn_encodings"], loss)

        if len(encoding_gradients.shape) == 1:
            encoding_gradients = encoding_gradients.unsqueeze(0)

        encoding_gradients = [
            x.numpy().tolist() for x in encoding_gradients.cpu().unbind(dim=0 if pl_module.batch_first else 1)
        ]

        pd.DataFrame(encoding_gradients).to_csv(f"{self.save_dir}/cat.csv", mode="a", header=False, index=False)


class AttCat(Callback):
    def __init__(self, save_dir: str = "att_grad") -> None:
        super().__init__()
        self.layer = -1
        self.save_dir = save_dir

    def attentive_class_activation_tokens(
        self, encodings: list[Tensor], attentions: list[Tensor], loss: Tensor, start_layer: int = 0
    ):
        cams = encodings
        grads = [torch.autograd.grad(loss, cam, retain_graph=True)[0] for cam in cams]

        cams = [att.mean(1).bmm(grad * cam) for grad, att, cam in zip(grads, attentions, cams)]
        # cams = [cam / cam.sum(dim=-1, keepdim=True) for cam in cams]

        # Summed Attributions
        cam = torch.stack(cams[start_layer:], dim=0).sum(dim=0)

        # Multiplied Attributions
        # cam = cams[start_layer]
        # for i in range(start_layer + 1, len(cams)):
        #     cam = cams[i].bmm(cam)

        cam = cam.mean(2).squeeze().detach()
        return cam

    def on_predict_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        pd.DataFrame().to_csv(validate_or_create_full_path(f"{self.save_dir}/att_cat.csv"), header=True, index=False)
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
        enc_encodings = pl_module.encoder.get_attn_encodings()  # type: ignore

        pl_module.zero_grad()
        one_hot_backprop = select_index(logits.detach(), y)
        loss = torch.sum(logits * one_hot_backprop, dtype=logits.dtype)

        attentive_class_activation_tokens = self.attentive_class_activation_tokens(
            enc_encodings["enc_attn_encodings"], enc_attn["enc_attn_weights"], loss
        )

        if len(attentive_class_activation_tokens.shape) == 1:
            attentive_class_activation_tokens = attentive_class_activation_tokens.unsqueeze(0)

        attentive_class_activation_tokens = [
            x.numpy().tolist()
            for x in attentive_class_activation_tokens.cpu().unbind(dim=0 if pl_module.batch_first else 1)
        ]

        pd.DataFrame(attentive_class_activation_tokens).to_csv(
            f"{self.save_dir}/att_cat.csv", mode="a", header=False, index=False
        )
