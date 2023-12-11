import torch.nn as nn


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size: int, dropout: float, norm: str, layer: nn.Module) -> None:
        super().__init__()
        self.norm = (
            nn.LayerNorm(size) if norm == "layer" else nn.BatchNorm1d(size) if norm == "batch" else nn.Identity()
        )
        self.dropout = nn.Dropout(dropout)
        self.layer = layer

    def forward(self, x):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(self.layer(self.norm(x)))


class Sublayer(nn.Module):
    """
    A layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size: int, dropout: float, norm: str, layer: nn.Module) -> None:
        super().__init__()
        self.norm = (
            nn.LayerNorm(size) if norm == "layer" else nn.BatchNorm1d(size) if norm == "batch" else nn.Identity()
        )
        self.dropout = nn.Dropout(dropout)
        self.layer = layer

    def forward(self, x):
        "Apply residual connection to any sublayer with the same size."
        return self.dropout(self.layer(self.norm(x)))
