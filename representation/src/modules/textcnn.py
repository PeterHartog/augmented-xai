from typing import List

import torch
import torch.nn as nn
from torch import Tensor


class Max(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return torch.max(x, dim=self.dim).values


class KernelUnit(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
    ) -> None:
        super().__init__()

        self.kernel = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
            ),
            nn.ReLU(),
            Max(dim=2),
        )

    def forward(self, x: Tensor) -> Tensor:
        self.kernel  # .to(x.device)
        return self.kernel(x)


class TextCNN(nn.Module):
    def __init__(
        self,
        kernel_sizes: List[int],
        filters: List[int],
        stride: int,
        in_size: int,
        out_size: int,
        dropout: float = 0.1,
        batch_first: bool = False,
    ) -> None:
        super().__init__()
        self.in_size = in_size
        self.stride = stride
        self.out_size = out_size
        self.dropout = dropout
        self.batch_first = batch_first

        # Create Convolution and Pooling Layers
        self.kernel_sizes = kernel_sizes
        self.filters = filters
        self.kernel_layers = nn.ModuleList(
            [
                KernelUnit(
                    in_channels=self.in_size,
                    out_channels=filter,
                    kernel_size=kernel_size,
                    stride=self.stride,
                )
                for kernel_size, filter in zip(self.kernel_sizes, self.filters)
            ]
        )

        self.output_net = nn.Sequential(
            nn.Linear(sum(self.filters), self.out_size),
            nn.Dropout(self.dropout),
            nn.ReLU(),  # nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        if not self.batch_first:
            x = x.permute(1, 0, 2)  # Batch second

        concatenated_kernels = torch.cat(
            [kernel_layer(x) for kernel_layer in self.kernel_layers],
            dim=-1,
        )
        return self.output_net(concatenated_kernels)
