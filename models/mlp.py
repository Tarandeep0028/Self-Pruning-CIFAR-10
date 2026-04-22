from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from config import DEFAULT_GATE_INIT_MEAN, DEFAULT_GATE_INIT_STD, DEFAULT_THRESHOLD
from models.prunable_layers import PrunableLinear


class PrunableMLP(nn.Module):
    """A compact MLP for CIFAR-10 classification with self-pruning layers."""

    def __init__(
        self,
        input_dim: int = 3 * 32 * 32,
        hidden_dims: Iterable[int] = (1024, 512),
        output_dim: int = 10,
        dropout: float = 0.1,
        threshold: float = DEFAULT_THRESHOLD,
        gate_init_mean: float = DEFAULT_GATE_INIT_MEAN,
        gate_init_std: float = DEFAULT_GATE_INIT_STD,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dims = tuple(int(dim) for dim in hidden_dims)
        self.output_dim = int(output_dim)
        self.dropout_rate = float(dropout)
        self.threshold = float(threshold)
        self.gate_init_mean = float(gate_init_mean)
        self.gate_init_std = float(gate_init_std)

        layer_dims = (self.input_dim, *self.hidden_dims, self.output_dim)
        self.layers = nn.ModuleList(
            [
                PrunableLinear(
                    layer_dims[idx],
                    layer_dims[idx + 1],
                    gate_init_mean=self.gate_init_mean,
                    gate_init_std=self.gate_init_std,
                )
                for idx in range(len(layer_dims) - 1)
            ]
        )
        self.dropout = nn.Dropout(p=self.dropout_rate) if self.dropout_rate > 0 else None

    def forward(self, inputs: Tensor) -> Tensor:
        x = torch.flatten(inputs, start_dim=1)
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
            if self.dropout is not None:
                x = self.dropout(x)
        return self.layers[-1](x)

    def to_config(self) -> dict[str, object]:
        """Return a checkpoint-safe configuration for model reconstruction."""
        return {
            "input_dim": self.input_dim,
            "hidden_dims": list(self.hidden_dims),
            "output_dim": self.output_dim,
            "dropout": self.dropout_rate,
            "threshold": self.threshold,
            "gate_init_mean": self.gate_init_mean,
            "gate_init_std": self.gate_init_std,
        }

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "PrunableMLP":
        return cls(
            input_dim=int(config["input_dim"]),
            hidden_dims=tuple(int(dim) for dim in config["hidden_dims"]),
            output_dim=int(config["output_dim"]),
            dropout=float(config["dropout"]),
            threshold=float(config.get("threshold", DEFAULT_THRESHOLD)),
            gate_init_mean=float(config.get("gate_init_mean", DEFAULT_GATE_INIT_MEAN)),
            gate_init_std=float(config.get("gate_init_std", DEFAULT_GATE_INIT_STD)),
        )
