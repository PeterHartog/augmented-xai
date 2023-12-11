"""Masked collate function."""
from typing import Optional

import torch
from torch import Tensor

from representation.src.datasets.collate_functions.abstract import AbstractCollate
from representation.src.utils.utils import (
    create_padding_mask,
    create_tgt_mask,
    mask_sequences,
    pad_sequences,
)


class MaskedCollate(AbstractCollate):
    def __init__(
        self,
        batch_first: bool,
        pad_idx: int,
        max_len: Optional[int],
        msk_idx: int,
        vocab_size: int,
        mask_p: float = 0.15,
        rand_p: float = 0.1,
        unchanged_p: float = 0.1,
    ) -> None:
        self.batch_first = batch_first
        self.pad_idx = pad_idx
        self.msk_idx = msk_idx
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.mask_p = mask_p
        self.rand_p = rand_p
        self.unchanged_p = unchanged_p

    def collate_fn(self, batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:  # type: ignore
        process: dict[str, list[Tensor]] = {key: [i[key] for i in batch] for key in batch[0].keys()}
        batch: dict[str, Tensor] = {
            key: torch.stack(value) for key, value in process.items() if key not in ["src", "tgt"]
        }
        batch["src"] = pad_sequences(process["src"], batch_first=True, padding_value=self.pad_idx, max_len=self.max_len)
        batch["src_padding_mask"] = create_padding_mask(batch["src"], self.pad_idx)

        batch["src"] = torch.stack(
            [
                mask_sequences(
                    s,
                    self.pad_idx,
                    self.msk_idx,
                    self.vocab_size,
                    self.mask_p,
                    self.rand_p,
                    self.unchanged_p,
                )
                for s in batch["src"].unbind(dim=0 if self.batch_first else 1)
            ]
        )

        batch["tgt"] = pad_sequences(process["tgt"], batch_first=True, padding_value=self.pad_idx, max_len=self.max_len)
        batch["tgt_mask"] = create_tgt_mask(batch["tgt"], self.batch_first)
        batch["tgt_padding_mask"] = create_padding_mask(batch["tgt"], self.pad_idx)
        return batch
