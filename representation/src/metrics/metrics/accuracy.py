import torch
from torch import Tensor
from torchmetrics import Metric


class CharacterAccuracy(Metric):
    def __init__(self, pad_idx: int = 0, batch_first: bool = True, average_per_sample: bool = False) -> None:
        super().__init__()
        self.add_state("averages", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("char_correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.pad_idx = pad_idx
        self.batch_first = batch_first
        self.average_per_sample = average_per_sample

    def forward(self, logits: Tensor, tgt: Tensor) -> Tensor:
        self.update(logits, tgt)
        return self.compute()

    def update(self, logits: Tensor, tgt: Tensor) -> None:
        batch_size = tgt.shape[0 if self.batch_first else 1]
        if tgt.dim() == 1:
            logits = logits.unsqueeze(0 if self.batch_first else 1)
            tgt = tgt.unsqueeze(0 if self.batch_first else 1)

        if logits.shape > tgt.shape:
            logits = torch.argmax(logits, dim=-1)

        if self.average_per_sample:
            for i in range(batch_size):
                mask = ~(logits[i].eq(self.pad_idx) & tgt[i].eq(self.pad_idx))
                correct = logits[i][mask].eq(tgt[i][mask]).sum()
                total = mask.sum()
                self.averages += correct.cpu() / total.cpu()
            self.total += torch.tensor(batch_size).cpu()
        else:
            mask = ~(logits.eq(self.pad_idx) & tgt.eq(self.pad_idx))
            correct = logits[mask].eq(tgt[mask]).sum()
            total = mask.sum()
            self.char_correct += correct.cpu()
            self.total += total.cpu()

    def compute(self) -> Tensor:
        if self.average_per_sample:
            return self.averages / self.total  # type: ignore
        else:
            return self.char_correct / self.total  # type: ignore


class SequenceAccuracy(Metric):
    def __init__(self, pad_idx: int = 0, batch_first: bool = True) -> None:
        super().__init__()
        self.add_state("seq_correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.pad_idx = pad_idx
        self.batch_first = batch_first

    def forward(self, logits: Tensor, tgt: Tensor) -> Tensor:
        self.update(logits, tgt)
        return self.compute()

    def update(self, logits: Tensor, tgt: Tensor) -> None:
        batch_size = tgt.shape[0 if self.batch_first else 1]
        seq_len = tgt.shape[1 if self.batch_first else 0]

        if tgt.dim() == 1:
            logits = logits.unsqueeze(0 if self.batch_first else 1)
            tgt = tgt.unsqueeze(0 if self.batch_first else 1)

        if logits.shape > tgt.shape:
            logits = torch.argmax(logits, dim=-1)

        seq_correct = torch.sum(torch.eq(tgt, logits), dim=1)
        seq_correct = (seq_correct == seq_len).sum()
        self.seq_correct += seq_correct.cpu()
        self.total += torch.tensor(batch_size).cpu()

    def compute(self) -> Tensor:
        return self.seq_correct / self.total  # type: ignore


class BeamSequenceAccuracy(Metric):
    def __init__(self, pad_idx: int = 0, batch_first: bool = True) -> None:
        super().__init__()
        self.add_state("seq_correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.pad_idx = pad_idx
        self.batch_first = batch_first

    def forward(self, logits: Tensor, tgt: Tensor) -> Tensor:
        self.update(logits, tgt)
        return self.compute()

    def update(self, logits: Tensor, tgt: Tensor) -> None:
        batch_size = tgt.shape[0 if self.batch_first else 1]
        seq_len = tgt.shape[1 if self.batch_first else 0]

        if tgt.dim() == 1:
            logits = logits.unsqueeze(0 if self.batch_first else 1)
            tgt = tgt.unsqueeze(0 if self.batch_first else 1)

        if logits.shape > tgt.shape:
            seqs_correct = torch.zeros((batch_size), device=self.device)
            for i, logit in enumerate(logits.unbind(-1)):
                seq_correct = torch.sum(torch.eq(tgt, logit), dim=1)
                seq_correct = seq_correct == seq_len
                seqs_correct += seq_correct
            self.seq_correct += seqs_correct.sum().long().cpu()
            self.total += torch.tensor(batch_size).cpu()

    def compute(self) -> Tensor:
        return self.seq_correct / self.total  # type: ignore
