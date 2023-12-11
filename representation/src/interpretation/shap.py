from typing import Any

import pandas as pd
import pytorch_lightning as pl
import torch
from aidd_codebase.utils.directories import validate_or_create_full_path
from captum.attr import GradientShap
from pytorch_lightning.callbacks import Callback


class ExplainabilitySHAP(Callback):
    def __init__(self, include_baseline: bool = True, save_dir: str = "shap") -> None:
        super().__init__()
        self.include_baseline = include_baseline
        self.save_dir = save_dir

    def on_predict_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.shap = GradientShap(pl_module.predict_proba_embed)  # type: ignore
        pd.DataFrame().to_csv(validate_or_create_full_path(f"{self.save_dir}/shap.csv"), header=True, index=False)
        pd.DataFrame(columns=["shap_delta"]).to_csv(
            validate_or_create_full_path(f"{self.save_dir}/shap_delta.csv"), header=True, index=False
        )
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
        # torch.set_grad_enabled(True)
        src, src_pad_mask = batch["src"], batch["src_padding_mask"]
        # logits = self.predict(src, None, src_pad_mask)
        # y_pred = torch.argmax(self.sigmoid(logits), dim=1).long().tolist()
        y = batch["label"].long().tolist()

        attributions, delta = self.shap.attribute(
            pl_module.src_embed(src),  # type: ignore
            additional_forward_args=(None, src_pad_mask),
            baselines=pl_module.src_embed(src * 0) if self.include_baseline else None,  # type: ignore
            target=y,
            return_convergence_delta=True,
        )
        if len(attributions.shape) == 2:
            attributions = attributions.unsqueeze(0)
        attributions = attributions.mean(-1).cpu().squeeze()  # type: ignore
        delta = delta.cpu()  # type: ignore

        if len(attributions.shape) == 1:
            attributions = attributions.unsqueeze(0)
            delta = delta.unsqueeze(0).cpu()  # type: ignore

        attributions = [x.detach().numpy().tolist() for x in attributions.unbind(dim=0 if pl_module.batch_first else 1)]
        delta = [x.numpy().tolist() for x in delta.unbind(dim=0 if pl_module.batch_first else 1)]
        pd.DataFrame(attributions).to_csv(f"{self.save_dir}/shap.csv", mode="a", header=False, index=False)
        pd.DataFrame(pd.Series(delta)).to_csv(f"{self.save_dir}/shap_delta.csv", mode="a", header=False, index=False)
