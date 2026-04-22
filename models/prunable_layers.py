from __future__ import annotations

import math
from typing import Iterator

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class PrunableLinear(nn.Module):
    """Linear layer whose weights are modulated by learnable sigmoid gates."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        gate_init_mean: float = 2.0,
        gate_init_std: float = 0.01,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gate_init_mean = float(gate_init_mean)
        self.gate_init_std = float(gate_init_std)

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1.0 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        nn.init.normal_(self.gate_scores, mean=self.gate_init_mean, std=self.gate_init_std)

    def gate_values(self) -> Tensor:
        return torch.sigmoid(self.gate_scores)

    def forward(self, inputs: Tensor) -> Tensor:
        gates = self.gate_values()
        pruned_weight = self.weight * gates
        return F.linear(inputs, pruned_weight, self.bias)


def iter_prunable_layers(model: nn.Module) -> Iterator[PrunableLinear]:
    """Yield all prunable layers inside a model."""
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            yield module


def collect_gate_tensors(model: nn.Module) -> list[Tensor]:
    """Collect sigmoid gate tensors from every prunable layer."""
    return [layer.gate_values() for layer in iter_prunable_layers(model)]


def compute_total_sparsity_loss(model: nn.Module) -> Tensor:
    """Compute the summed gate penalty used for sparsity regularization."""
    gate_tensors = collect_gate_tensors(model)
    if not gate_tensors:
        parameter = next(model.parameters(), None)
        device = parameter.device if parameter is not None else None
        return torch.tensor(0.0, device=device)
    return torch.stack([gate.sum() for gate in gate_tensors]).sum()


def compute_sparsity_percentage(model: nn.Module, threshold: float = 1e-2) -> float:
    """Return the percentage of gates whose value falls below the threshold."""
    gate_tensors = collect_gate_tensors(model)
    if not gate_tensors:
        return 0.0

    total_weights = 0
    pruned_weights = 0
    for gate in gate_tensors:
        detached_gate = gate.detach()
        total_weights += detached_gate.numel()
        pruned_weights += int((detached_gate < threshold).sum().item())
    return 100.0 * pruned_weights / total_weights
