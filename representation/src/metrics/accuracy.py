from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT

from representation.src.metrics.metrics.accuracy import (
    BeamSequenceAccuracy,
    CharacterAccuracy,
    SequenceAccuracy,
)


class TrainingAccuracy(Callback):
    def __init__(
        self, pad_idx: int = 0, on_batch: bool = True, on_epoch: bool = True, remove_bos: bool = False
    ) -> None:
        super().__init__()
        self.char_acc = CharacterAccuracy(pad_idx=pad_idx, batch_first=True, average_per_sample=False).cpu()
        self.seq_acc = SequenceAccuracy(pad_idx=pad_idx, batch_first=True).cpu()
        self.on_batch = on_batch
        self.on_epoch = on_epoch
        self.remove_bos = remove_bos

    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        tgt = batch["tgt"][:, 1:] if not self.remove_bos else batch["tgt"]
        logits = outputs["lm_logits"]  # type: ignore
        self.char_acc.to(pl_module.device)
        self.seq_acc.to(pl_module.device)

        char_acc_batch = self.char_acc(logits, tgt).cpu()
        seq_acc_batch = self.seq_acc(logits, tgt).cpu()

        if self.on_batch:
            pl_module.log("Batch/Train/char_acc", char_acc_batch)
            pl_module.log("Batch/Train/seq_acc", seq_acc_batch)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        pl_module.log("Epoch/Train/char_acc", self.char_acc.compute())
        pl_module.log("Epoch/Train/seq_acc", self.seq_acc.compute())
        self.char_acc.reset()
        self.seq_acc.reset()


class ValidationAccuracy(Callback):
    def __init__(
        self, pad_idx: int = 0, on_batch: bool = True, on_epoch: bool = True, remove_bos: bool = False
    ) -> None:
        super().__init__()
        self.char_acc = CharacterAccuracy(pad_idx=pad_idx, batch_first=True, average_per_sample=False).cpu()
        self.seq_acc = SequenceAccuracy(pad_idx=pad_idx, batch_first=True).cpu()
        self.on_batch = on_batch
        self.on_epoch = on_epoch
        self.remove_bos = remove_bos

    def on_validation_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        tgt = batch["tgt"][:, 1:] if not self.remove_bos else batch["tgt"]
        logits = outputs["lm_logits"]  # type: ignore
        self.char_acc.to(pl_module.device)
        self.seq_acc.to(pl_module.device)

        char_acc_batch = self.char_acc(logits, tgt).cpu()
        seq_acc_batch = self.seq_acc(logits, tgt).cpu()

        if self.on_batch:
            pl_module.log("Batch/Validation/char_acc", char_acc_batch)
            pl_module.log("Batch/Validation/seq_acc", seq_acc_batch)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        pl_module.log("Epoch/Validation/char_acc", self.char_acc.compute())
        pl_module.log("Epoch/Validation/seq_acc", self.seq_acc.compute())
        self.char_acc.reset()
        self.seq_acc.reset()


class TestAccuracy(Callback):
    def __init__(
        self, pad_idx: int = 0, on_batch: bool = True, on_epoch: bool = True, remove_bos: bool = False
    ) -> None:
        super().__init__()
        self.char_acc = CharacterAccuracy(pad_idx=pad_idx, batch_first=True, average_per_sample=False).cpu()
        self.seq_acc = SequenceAccuracy(pad_idx=pad_idx, batch_first=True).cpu()
        self.on_batch = on_batch
        self.on_epoch = on_epoch
        self.remove_bos = remove_bos

    def on_test_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        tgt = batch["tgt"][:, 1:] if not self.remove_bos else batch["tgt"]
        logits = outputs["lm_logits"]  # type: ignore
        self.char_acc.to(pl_module.device)
        self.seq_acc.to(pl_module.device)

        char_acc_batch = self.char_acc(logits, tgt).cpu()
        seq_acc_batch = self.seq_acc(logits, tgt).cpu()

        if self.on_batch:
            pl_module.log("Batch/Test/char_acc", char_acc_batch)
            pl_module.log("Batch/Test/seq_acc", seq_acc_batch)

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        pl_module.log("Epoch/Test/char_acc", self.char_acc.compute())
        pl_module.log("Epoch/Test/seq_acc", self.seq_acc.compute().mean())
        self.char_acc.reset()
        self.seq_acc.reset()


class TestGreedySearchAccuracy(Callback):
    def __init__(self, pad_idx: int = 0, on_batch: bool = True, on_epoch: bool = True) -> None:
        super().__init__()
        self.char_acc = CharacterAccuracy(pad_idx=pad_idx, batch_first=True, average_per_sample=False).cpu()
        self.seq_acc = SequenceAccuracy(pad_idx=pad_idx, batch_first=True).cpu()
        self.on_batch = on_batch
        self.on_epoch = on_epoch

    def on_test_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        src, src_pad_mask = batch["src"], batch["src_padding_mask"]
        tgt = batch["tgt"][:, 1:]

        hidden_state = pl_module(src, None, src_pad_mask)
        predictions = pl_module.predict(hidden_state, None, src_pad_mask, alg="greedy", k=None)  # type: ignore
        self.char_acc.to(pl_module.device)
        self.seq_acc.to(pl_module.device)

        char_acc_batch = self.char_acc(predictions, tgt).cpu()
        seq_acc_batch = self.seq_acc(predictions, tgt).cpu()

        if self.on_batch:
            pl_module.log("Batch/Test/greedy_char_acc", char_acc_batch)
            pl_module.log("Batch/Test/greedy_seq_acc", seq_acc_batch)

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        pl_module.log("Epoch/Test/greedy_char_acc", self.char_acc.compute())
        pl_module.log("Epoch/Test/greedy_seq_acc", self.seq_acc.compute().mean())
        self.char_acc.reset()
        self.seq_acc.reset()


class TestBeamSearchAccuracy(Callback):
    def __init__(self, pad_idx: int = 0, on_batch: bool = True, on_epoch: bool = True, beam_width: int = 5) -> None:
        super().__init__()
        self.beam_seq_acc = BeamSequenceAccuracy(pad_idx=pad_idx, batch_first=True).cpu()
        self.on_batch = on_batch
        self.on_epoch = on_epoch
        self.beam_width = beam_width

    def on_test_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        src, src_pad_mask = batch["src"], batch["src_padding_mask"]
        tgt = batch["tgt"][:, 1:]

        hidden_state = pl_module(src, None, src_pad_mask)
        predictions = pl_module.predict(hidden_state, None, src_pad_mask, alg="beam", k=self.beam_width)  # type: ignore
        self.beam_seq_acc.to(pl_module.device)

        seq_acc_batch = self.beam_seq_acc(predictions, tgt).cpu()

        if self.on_batch:
            pl_module.log(f"Batch/Test/top{self.beam_width}_seq_acc", seq_acc_batch)

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        pl_module.log(f"Epoch/Test/top{self.beam_width}_seq_acc", self.beam_seq_acc.compute().mean())
        self.beam_seq_acc.reset()
