from typing import Callable, Optional

import torch
import torch.nn as nn


class EmbeddingAdaptor(nn.Module):
    def __init__(
        self,
        adaption: str = "sum",
        adapt_embedding: bool = False,
        batch_first: bool = True,
        switch_dims: bool = False,
        in_size: Optional[int] = None,
    ):
        super().__init__()
        self.adaption = adaption
        self.adapt_embedding = adapt_embedding
        self.batch_first = batch_first
        self.switch_dims = switch_dims

        self.extra_steps: Optional[Callable] = None
        self.add_kwargs: dict = {"dim": 2, "keepdim": False}
        if adaption == "sum":
            self.adaptation = torch.sum
        elif adaption == "mean":
            self.adaptation = torch.mean
        elif adaption == "max":
            self.adaptation = torch.max
            self.extra_steps = lambda x: x.values
        elif adaption == "neural_net":
            assert in_size is not None
            self.adaptation = nn.Sequential(nn.Linear(in_size, 1), nn.ReLU())
            self.add_kwargs = {}
        else:
            raise ValueError("Adaptation not in (mean, sum, max, neural_net)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.switch_dims:
            x = x.transpose(1 if self.batch_first else 0, 2)
        x = self.adaptation(x, **self.add_kwargs)
        if self.extra_steps is not None:
            x = self.extra_steps(x)
        return x.squeeze() if len(x.shape) > 2 else x
