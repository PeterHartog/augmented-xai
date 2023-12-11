"""MLP model."""
import torch
import torch.nn as nn

# Layers
from representation.src.layers.max_sep_layer import MaxSeparatedLayer
from representation.src.utils.sublayer import Sublayer, SublayerConnection


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        norm: str = "layer",
        act_fn: str = "relu",
        max_sep: bool = False,
        num_layers: int = 1,
        skip: bool = False,
    ):
        """
        @article{kasarla2022maximum,
          title={Maximum Separation as Inductive Bias in One Matrix},
          author={Kasarla, Tejaswi and  and Burghouts, Gertjan J and van Spengler, Max and van der Pol,
            Elise and Cucchiara, Rita and Mettes, Pascal},
          journal={arXiv preprint arXiv:2206.08704},
          year={2022}
        }

        Args:
            input_dim (int): _description_
            output_dim (int): _description_
            hidden_dim (int, optional): _description_. Defaults to 128.
            dropout (float, optional): _description_. Defaults to 0.2.
        """
        super().__init__()

        self.skip = skip
        self.fc1 = nn.Sequential(
            Sublayer(input_dim, dropout, norm, nn.Linear(input_dim, hidden_dim)),
            nn.ReLU()
            if act_fn == "relu"
            else nn.Sigmoid()
            if act_fn == "sigmoid"
            else nn.GELU()
            if act_fn == "gelu"
            else nn.Identity(),
        )

        self.hidden_layers = nn.ModuleList(
            [
                nn.Sequential(
                    SublayerConnection(hidden_dim, dropout, norm, nn.Linear(hidden_dim, hidden_dim))
                    if skip
                    else Sublayer(hidden_dim, dropout, norm, nn.Linear(hidden_dim, hidden_dim)),
                    nn.ReLU()
                    if act_fn == "relu"
                    else nn.Sigmoid()
                    if act_fn == "sigmoid"
                    else nn.GELU()
                    if act_fn == "gelu"
                    else nn.Identity(),
                )
                for _ in range(num_layers)
            ]
        )

        if max_sep:
            self.fc_out = nn.Sequential(
                nn.Linear(hidden_dim, output_dim - 1),
                MaxSeparatedLayer(output_dim, prototype="normal"),
            )
        else:
            self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.fc_out(x)
        return x


class Network(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout, norm, act_fn, max_sep, num_layers, skip):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.sigmoid(x)
        x = self.linear2(x)
        return x
