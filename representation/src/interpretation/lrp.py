from typing import Any

import pandas as pd
import pytorch_lightning as pl
import torch
from aidd_codebase.utils.directories import validate_or_create_full_path
from captum.attr import (  # remove_interpretable_embedding_layer,
    LRP,
    configure_interpretable_embedding_layer,
)
from pytorch_lightning.callbacks import Callback


class ExplainabilityLRP(Callback):
    def __init__(self, include_baseline: bool = True, save_dir: str = "lrp") -> None:
        super().__init__()
        self.include_baseline = include_baseline
        self.save_dir = save_dir

    def on_predict_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.lrp = LRP(pl_module)  # type: ignore
        self.interpretable_emb = configure_interpretable_embedding_layer(pl_module, "src_embed")

        pd.DataFrame().to_csv(validate_or_create_full_path(f"{self.save_dir}/lrp.csv"), header=True, index=False)
        pd.DataFrame(columns=["lrp_delta"]).to_csv(
            validate_or_create_full_path(f"{self.save_dir}/lrp_delta.csv"), header=True, index=False
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

        attributions, delta = self.lrp.attribute(
            self.interpretable_emb.indices_to_embeddings(src),
            # pl_module.src_embed(src),  # type: ignore
            additional_forward_args=(None, src_pad_mask),
            target=y,
            return_convergence_delta=True,
        )
        attributions = attributions.mean(-1).cpu().squeeze()  # type: ignore
        delta = delta.cpu()  # type: ignore

        attributions = [x.detach().numpy().tolist() for x in attributions.unbind(dim=0 if pl_module.batch_first else 1)]
        delta = [x.numpy().tolist() for x in delta.unbind(dim=0 if pl_module.batch_first else 1)]
        pd.DataFrame(attributions).to_csv(f"{self.save_dir}/lrp.csv", mode="a", header=False)
        pd.DataFrame(pd.Series(delta)).to_csv(f"{self.save_dir}/lrp_delta.csv", mode="a", header=False)
