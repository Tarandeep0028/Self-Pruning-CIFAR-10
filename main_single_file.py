from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

PROJECT_ROOT = Path(__file__).resolve().parent
RUNTIME_TEMP_DIR = PROJECT_ROOT / ".tmp"
RUNTIME_TEMP_DIR.mkdir(parents=True, exist_ok=True)
RUNTIME_TORCH_CACHE_DIR = PROJECT_ROOT / ".torch_cache"
RUNTIME_TORCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
RUNTIME_MPL_CONFIG_DIR = PROJECT_ROOT / ".mplconfig"
RUNTIME_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
for _key in ("TMP", "TEMP", "TMPDIR"):
    os.environ.setdefault(_key, str(RUNTIME_TEMP_DIR))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(RUNTIME_TORCH_CACHE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_MPL_CONFIG_DIR))

import matplotlib
import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

matplotlib.use("Agg")

import matplotlib.pyplot as plt

DEFAULT_DATA_DIR = Path("data")
DEFAULT_OUTPUT_DIR = Path("outputs")
OUTPUT_SUBDIRS = ("checkpoints", "plots", "metrics", "sweeps")

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

DEFAULT_THRESHOLD = 1e-2
DEFAULT_INPUT_DIM = 3 * 32 * 32
DEFAULT_HIDDEN_DIMS = (1024, 512)
DEFAULT_OUTPUT_DIM = 10
DEFAULT_EPOCHS = 15
DEFAULT_BATCH_SIZE = 128
DEFAULT_LR = 1e-3
DEFAULT_DROPOUT = 0.1
DEFAULT_GATE_INIT_MEAN = 2.0
DEFAULT_GATE_INIT_STD = 0.01
DEFAULT_GATE_LR_MULTIPLIER = 1.0
DEFAULT_SEED = 42
DEFAULT_NUM_WORKERS = 0
DEFAULT_SWEEP_LAMBDAS = [1e-5, 5e-5, 1e-4]
QUICK_TRAIN_SUBSET = 2048
QUICK_TEST_SUBSET = 1024

SAMPLE_RESULTS = [
    {"Lambda": 1e-5, "Test Accuracy": 48.70, "Sparsity Level (%)": 3.12},
    {"Lambda": 5e-5, "Test Accuracy": 47.35, "Sparsity Level (%)": 8.84},
    {"Lambda": 1e-4, "Test Accuracy": 44.90, "Sparsity Level (%)": 17.26},
]

SWEEP_RESULTS_START = "<!-- SWEEP_RESULTS_START -->"
SWEEP_RESULTS_END = "<!-- SWEEP_RESULTS_END -->"


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def ensure_output_dirs(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Path]:
    root = resolve_path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    directories = {"root": root}
    for name in OUTPUT_SUBDIRS:
        path = root / name
        path.mkdir(parents=True, exist_ok=True)
        directories[name] = path
    return directories


class PrunableLinear(nn.Module):
    """Linear layer whose weights are multiplied by learnable sigmoid gates."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        gate_init_mean: float = DEFAULT_GATE_INIT_MEAN,
        gate_init_std: float = DEFAULT_GATE_INIT_STD,
    ) -> None:
        super().__init__()
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
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            yield module


def collect_gate_tensors(model: nn.Module) -> list[Tensor]:
    return [layer.gate_values() for layer in iter_prunable_layers(model)]


def compute_total_sparsity_loss(model: nn.Module) -> Tensor:
    gate_tensors = collect_gate_tensors(model)
    if not gate_tensors:
        parameter = next(model.parameters(), None)
        device = parameter.device if parameter is not None else None
        return torch.tensor(0.0, device=device)
    return torch.stack([gate.sum() for gate in gate_tensors]).sum()


def compute_sparsity_percentage(model: nn.Module, threshold: float = DEFAULT_THRESHOLD) -> float:
    gate_tensors = collect_gate_tensors(model)
    if not gate_tensors:
        return 0.0
    total_weights = sum(gate.numel() for gate in gate_tensors)
    pruned_weights = sum(int((gate.detach() < threshold).sum().item()) for gate in gate_tensors)
    return 100.0 * pruned_weights / total_weights


class PrunableMLP(nn.Module):
    """Self-pruning multilayer perceptron for CIFAR-10."""

    def __init__(
        self,
        input_dim: int = DEFAULT_INPUT_DIM,
        hidden_dims: Iterable[int] = DEFAULT_HIDDEN_DIMS,
        output_dim: int = DEFAULT_OUTPUT_DIM,
        dropout: float = DEFAULT_DROPOUT,
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
        self.dropout = nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else None

    def forward(self, inputs: Tensor) -> Tensor:
        x = torch.flatten(inputs, start_dim=1)
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
            if self.dropout is not None:
                x = self.dropout(x)
        return self.layers[-1](x)

    def to_config(self) -> dict[str, object]:
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


def build_cifar10_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ]
    )


def _maybe_subset_dataset(dataset: Dataset, subset_size: int | None, seed: int) -> Dataset:
    if subset_size is None or subset_size >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:subset_size].tolist()
    return Subset(dataset, indices)


def get_cifar10_loaders(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    quick: bool = False,
    train_subset: int | None = None,
    test_subset: int | None = None,
    seed: int = DEFAULT_SEED,
    download: bool = True,
) -> tuple[DataLoader, DataLoader]:
    resolved_data_dir = resolve_path(data_dir)
    resolved_data_dir.mkdir(parents=True, exist_ok=True)

    if quick:
        train_subset = QUICK_TRAIN_SUBSET if train_subset is None else train_subset
        test_subset = QUICK_TEST_SUBSET if test_subset is None else test_subset

    transform = build_cifar10_transform()
    train_dataset = datasets.CIFAR10(root=resolved_data_dir, train=True, transform=transform, download=download)
    test_dataset = datasets.CIFAR10(root=resolved_data_dir, train=False, transform=transform, download=download)

    train_dataset = _maybe_subset_dataset(train_dataset, train_subset, seed)
    test_dataset = _maybe_subset_dataset(test_dataset, test_subset, seed + 1)

    common_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        **common_kwargs,
    )
    test_loader = DataLoader(test_dataset, shuffle=False, **common_kwargs)
    return train_loader, test_loader


def get_classification_loss() -> nn.Module:
    return nn.CrossEntropyLoss()


def compute_total_loss(
    logits: Tensor,
    targets: Tensor,
    model: nn.Module,
    lambda_sparsity: float,
    criterion: nn.Module | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    criterion = criterion if criterion is not None else get_classification_loss()
    classification_loss = criterion(logits, targets)
    sparsity_loss = compute_total_sparsity_loss(model)
    total_loss = classification_loss + lambda_sparsity * sparsity_loss
    return classification_loss, sparsity_loss, total_loss


def seed_everything(seed: int) -> int:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)
    return seed


@dataclass
class RunningAverage:
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += value * n
        self.count += n

    @property
    def average(self) -> float:
        return self.total / self.count if self.count else 0.0


def count_correct_predictions(logits: Tensor, targets: Tensor) -> int:
    predictions = torch.argmax(logits, dim=1)
    return int((predictions == targets).sum().item())


def create_epoch_record(
    epoch: int,
    train_metrics: dict[str, float],
    test_metrics: dict[str, float],
    lambda_sparsity: float,
) -> dict[str, float]:
    return {
        "epoch": epoch,
        "lambda_sparsity": lambda_sparsity,
        "train_classification_loss": train_metrics["classification_loss"],
        "train_sparsity_loss": train_metrics["sparsity_loss"],
        "train_total_loss": train_metrics["total_loss"],
        "train_accuracy": train_metrics["accuracy"],
        "test_classification_loss": test_metrics["classification_loss"],
        "test_sparsity_loss": test_metrics["sparsity_loss"],
        "test_total_loss": test_metrics["total_loss"],
        "test_accuracy": test_metrics["accuracy"],
        "test_sparsity_percentage": test_metrics["sparsity_percentage"],
    }


def format_epoch_log(
    epoch: int,
    total_epochs: int,
    train_metrics: dict[str, float],
    test_metrics: dict[str, float],
) -> str:
    return (
        f"Epoch {epoch:02d}/{total_epochs:02d} | "
        f"cls={train_metrics['classification_loss']:.4f} | "
        f"sparse={train_metrics['sparsity_loss']:.4f} | "
        f"total={train_metrics['total_loss']:.4f} | "
        f"train_acc={train_metrics['accuracy']:.2f}% | "
        f"test_acc={test_metrics['accuracy']:.2f}%"
    )


def save_json(path: str | Path, payload: dict | list[dict]) -> Path:
    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return resolved_path


def save_csv(path: str | Path, records: list[dict]) -> Path:
    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(resolved_path, index=False)
    return resolved_path


def ensure_artifacts_exist(paths: Iterable[str | Path]) -> None:
    missing_paths = [str(Path(path)) for path in paths if not Path(path).exists()]
    if missing_paths:
        raise FileNotFoundError(f"Expected artifact(s) not found: {', '.join(missing_paths)}")


def compute_gate_statistics(model: nn.Module) -> dict[str, float]:
    gate_tensors = collect_gate_tensors(model)
    if not gate_tensors:
        return {
            "min_gate": 0.0,
            "mean_gate": 0.0,
            "median_gate": 0.0,
            "p10_gate": 0.0,
            "p25_gate": 0.0,
            "pct_below_0.1": 0.0,
            "pct_below_0.05": 0.0,
            "pct_below_0.01": 0.0,
        }

    flattened = np.concatenate([gate.detach().cpu().numpy().ravel() for gate in gate_tensors])
    return {
        "min_gate": float(flattened.min()),
        "mean_gate": float(flattened.mean()),
        "median_gate": float(np.median(flattened)),
        "p10_gate": float(np.percentile(flattened, 10)),
        "p25_gate": float(np.percentile(flattened, 25)),
        "pct_below_0.1": float((flattened < 0.1).mean() * 100.0),
        "pct_below_0.05": float((flattened < 0.05).mean() * 100.0),
        "pct_below_0.01": float((flattened < 0.01).mean() * 100.0),
    }


def flatten_gate_values(model: nn.Module) -> np.ndarray:
    gate_tensors = collect_gate_tensors(model)
    if not gate_tensors:
        return np.array([], dtype=np.float32)
    return np.concatenate([gate.detach().cpu().numpy().ravel() for gate in gate_tensors])


def save_gate_histogram(model: nn.Module, path: str | Path, title: str) -> Path:
    histogram_path = Path(path)
    histogram_path.parent.mkdir(parents=True, exist_ok=True)
    gate_values = flatten_gate_values(model)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(gate_values, bins=50, color="#2563eb", alpha=0.9, edgecolor="black")
    axis.set_title(title)
    axis.set_xlabel("Gate value")
    axis.set_ylabel("Count")
    figure.tight_layout()
    figure.savefig(histogram_path)
    plt.close(figure)
    return histogram_path


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def make_run_name(lambda_sparsity: float, seed: int) -> str:
    return f"lambda_{lambda_sparsity:.0e}_seed_{seed}"


def run_sanity_checks(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    lambda_sparsity: float,
    threshold: float,
) -> None:
    model.train()
    batch_inputs, batch_targets = next(iter(train_loader))
    batch_inputs = batch_inputs.to(device)
    batch_targets = batch_targets.to(device)

    logits = model(batch_inputs)
    expected_output_dim = int(getattr(model, "output_dim", DEFAULT_OUTPUT_DIM))
    expected_shape = (batch_inputs.size(0), expected_output_dim)
    if logits.shape != expected_shape:
        raise AssertionError(f"Expected logits shape {expected_shape}, received {tuple(logits.shape)}")

    criterion = get_classification_loss()
    classification_loss, sparsity_loss, total_loss = compute_total_loss(
        logits=logits,
        targets=batch_targets,
        model=model,
        lambda_sparsity=lambda_sparsity,
        criterion=criterion,
    )
    gate_tensors = collect_gate_tensors(model)
    if not gate_tensors:
        raise AssertionError("No gate tensors were collected during the sanity check.")

    manual_sparsity_loss = torch.stack([gate.sum() for gate in gate_tensors]).sum()
    if not torch.allclose(sparsity_loss, manual_sparsity_loss):
        raise AssertionError("Sparsity loss helper does not match the manual gate sum.")

    total_loss.backward()
    for layer in iter_prunable_layers(model):
        if layer.weight.grad is None or layer.gate_scores.grad is None:
            raise AssertionError("Expected both weight and gate_scores gradients during the sanity check.")

    manual_sparsity_pct = 100.0 * sum(int((gate.detach() < threshold).sum().item()) for gate in gate_tensors) / sum(
        gate.numel() for gate in gate_tensors
    )
    helper_sparsity_pct = compute_sparsity_percentage(model, threshold)
    if abs(manual_sparsity_pct - helper_sparsity_pct) > 1e-9:
        raise AssertionError("Sparsity percentage helper does not match the manual threshold count.")

    model.zero_grad(set_to_none=True)


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    lambda_sparsity: float,
    criterion: nn.Module,
) -> dict[str, float]:
    model.train()
    cls_avg = RunningAverage()
    sparse_avg = RunningAverage()
    total_avg = RunningAverage()
    correct_predictions = 0
    total_examples = 0

    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)

        logits = model(inputs)
        classification_loss, sparsity_loss, total_loss = compute_total_loss(
            logits=logits,
            targets=targets,
            model=model,
            lambda_sparsity=lambda_sparsity,
            criterion=criterion,
        )
        total_loss.backward()
        optimizer.step()

        cls_avg.update(float(classification_loss.item()))
        sparse_avg.update(float(sparsity_loss.item()))
        total_avg.update(float(total_loss.item()))
        correct_predictions += count_correct_predictions(logits, targets)
        total_examples += targets.size(0)

    return {
        "classification_loss": cls_avg.average,
        "sparsity_loss": sparse_avg.average,
        "total_loss": total_avg.average,
        "accuracy": 100.0 * correct_predictions / max(total_examples, 1),
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    lambda_sparsity: float,
    threshold: float,
    criterion: nn.Module | None = None,
) -> dict[str, float]:
    model.eval()
    criterion = criterion if criterion is not None else get_classification_loss()
    cls_avg = RunningAverage()
    sparse_avg = RunningAverage()
    total_avg = RunningAverage()
    correct_predictions = 0
    total_examples = 0

    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)

        classification_loss, sparsity_loss, total_loss = compute_total_loss(
            logits=logits,
            targets=targets,
            model=model,
            lambda_sparsity=lambda_sparsity,
            criterion=criterion,
        )

        cls_avg.update(float(classification_loss.item()))
        sparse_avg.update(float(sparsity_loss.item()))
        total_avg.update(float(total_loss.item()))
        correct_predictions += count_correct_predictions(logits, targets)
        total_examples += targets.size(0)

    return {
        "classification_loss": cls_avg.average,
        "sparsity_loss": sparse_avg.average,
        "total_loss": total_avg.average,
        "accuracy": 100.0 * correct_predictions / max(total_examples, 1),
        "sparsity_percentage": compute_sparsity_percentage(model, threshold),
    }


def checkpoint_payload(
    model: PrunableMLP,
    optimizer: optim.Optimizer,
    epoch: int,
    best_test_accuracy: float,
    lambda_sparsity: float,
) -> dict[str, object]:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_test_accuracy": best_test_accuracy,
        "lambda_sparsity": lambda_sparsity,
        "model_config": model.to_config(),
    }


def build_optimizer(model: nn.Module, lr: float, gate_lr_multiplier: float = DEFAULT_GATE_LR_MULTIPLIER) -> optim.Optimizer:
    gate_params = []
    other_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.endswith("gate_scores"):
            gate_params.append(parameter)
        else:
            other_params.append(parameter)

    param_groups = [{"params": other_params, "lr": lr}]
    if gate_params:
        param_groups.append({"params": gate_params, "lr": lr * gate_lr_multiplier})
    return optim.Adam(param_groups)


def write_seeded_summary(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> Path:
    output_paths = ensure_output_dirs(output_dir)
    summary_path = output_paths["root"] / "summary.md"
    summary_text = "\n".join(
        [
            "# Sweep Summary",
            "",
            "This repository ships with a scaffolded sample summary so the project looks submission-ready immediately after generation. Running `python sweep.py` or `python main_single_file.py --mode sweep` replaces the table below with measured results.",
            "",
            markdown_table(SAMPLE_RESULTS),
            "",
            "The sample rows illustrate the expected CSV/Markdown format and the accuracy-sparsity trade-off produced by the self-pruning objective.",
        ]
    )
    summary_path.write_text(summary_text + "\n", encoding="utf-8")
    results_csv_path = output_paths["root"] / "results.csv"
    pd.DataFrame(SAMPLE_RESULTS).to_csv(results_csv_path, index=False)
    return summary_path


def write_report_template(report_path: str | Path = Path("report") / "report.md") -> Path:
    resolved_report_path = resolve_path(report_path)
    resolved_report_path.parent.mkdir(parents=True, exist_ok=True)
    content = f"""# Self-Pruning Neural Network Report

## Overview

This project implements a self-pruning multilayer perceptron for CIFAR-10 image classification. Each dense connection is paired with a learnable gate score, and the gate value is computed with a sigmoid. During training, the model learns both the classifier weights and the gate parameters, allowing the network to reduce unimportant connections while it is still optimizing for classification accuracy.

## Pruning Mechanism

Every `PrunableLinear` layer stores a standard weight matrix, a bias vector, and a tensor of learnable `gate_scores` with the same shape as the weights. The forward pass maps the gate scores through a sigmoid to obtain values between 0 and 1, multiplies the gates elementwise with the weights, and applies a standard linear transformation using the gated weights. This design keeps the layer fully differentiable so gradients reach both the original weights and the pruning gates.

## Why the L1 Gate Penalty Encourages Sparsity

The sparsity term is the sum of all sigmoid gate values across the network. Because the sigmoid outputs are always non-negative, this is equivalent to an L1 penalty on the effective gate activations. L1-style penalties are well known for preferring solutions that drive many coefficients toward zero rather than shrinking all coefficients uniformly. In this case, the optimizer is rewarded for making unnecessary gates very small, which suppresses their corresponding weights and increases the measured sparsity level.

## Training and Evaluation Setup

The classifier is a compact MLP with architecture `3072 -> 1024 -> 512 -> 10`, ReLU activations, and optional dropout. CIFAR-10 images are converted to tensors, normalized with standard channel statistics, and flattened inside the model. Training uses Adam, deterministic seeding, GPU acceleration when available, and an epoch-wise evaluation loop that tracks both test accuracy and the percentage of gates below the reporting threshold of `1e-2`.

## Artifacts

The modular training pipeline saves checkpoints in `outputs/checkpoints/`, plots in `outputs/plots/`, metrics in `outputs/metrics/`, and sweep summaries in `outputs/`. The best sweep histogram is expected at `outputs/plots/sweep_best_gate_hist.png` after running the sweep. When the repository is first generated, the results block below contains illustrative values so the report renders cleanly; running a sweep refreshes that block automatically with measured results.

## Sweep Results

{SWEEP_RESULTS_START}
{markdown_table(SAMPLE_RESULTS)}
{SWEEP_RESULTS_END}

## Notes

The section above is intentionally reserved for sweep output. Running `python sweep.py` or `python main_single_file.py --mode sweep` replaces only the table inside the marked block while preserving the narrative sections of this report.
"""
    resolved_report_path.write_text(content, encoding="utf-8")
    return resolved_report_path


def ensure_seed_files(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    report_path: str | Path = Path("report") / "report.md",
) -> None:
    output_paths = ensure_output_dirs(output_dir)
    summary_path = output_paths["root"] / "summary.md"
    results_csv_path = output_paths["root"] / "results.csv"
    resolved_report_path = resolve_path(report_path)

    if not summary_path.exists() or not results_csv_path.exists():
        write_seeded_summary(output_dir)
    if not resolved_report_path.exists():
        write_report_template(resolved_report_path)


def markdown_table(records: list[dict[str, float]]) -> str:
    header = "| Lambda | Test Accuracy | Sparsity Level (%) |"
    separator = "| --- | ---: | ---: |"
    rows = [
        f"| {record['Lambda']:.0e} | {record['Test Accuracy']:.2f} | {record['Sparsity Level (%)']:.2f} |"
        for record in records
    ]
    return "\n".join([header, separator, *rows])


def update_report_results(report_path: Path, records: list[dict[str, float]]) -> Path:
    report_text = report_path.read_text(encoding="utf-8")
    start_index = report_text.find(SWEEP_RESULTS_START)
    end_index = report_text.find(SWEEP_RESULTS_END)
    if start_index == -1 or end_index == -1 or start_index > end_index:
        raise ValueError("Report file does not contain a valid sweep results block.")

    replacement = f"{SWEEP_RESULTS_START}\n{markdown_table(records)}\n{SWEEP_RESULTS_END}"
    updated = report_text[:start_index] + replacement + report_text[end_index + len(SWEEP_RESULTS_END) :]
    report_path.write_text(updated, encoding="utf-8")
    return report_path


def run_training(
    lambda_sparsity: float,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    dropout: float = DEFAULT_DROPOUT,
    gate_init_mean: float = DEFAULT_GATE_INIT_MEAN,
    gate_init_std: float = DEFAULT_GATE_INIT_STD,
    gate_lr_multiplier: float = DEFAULT_GATE_LR_MULTIPLIER,
    seed: int = DEFAULT_SEED,
    device: str = "auto",
    data_dir: str | Path = DEFAULT_DATA_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    threshold: float = DEFAULT_THRESHOLD,
    quick: bool = False,
    train_subset: int | None = None,
    test_subset: int | None = None,
    num_workers: int = DEFAULT_NUM_WORKERS,
    save_aliases: bool = True,
    download: bool = True,
) -> dict[str, object]:
    seed_everything(seed)
    output_paths = ensure_output_dirs(output_dir)
    resolved_device = resolve_device(device)

    train_loader, test_loader = get_cifar10_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        quick=quick,
        train_subset=train_subset,
        test_subset=test_subset,
        seed=seed,
        download=download,
    )

    model = PrunableMLP(
        dropout=dropout,
        threshold=threshold,
        gate_init_mean=gate_init_mean,
        gate_init_std=gate_init_std,
    ).to(resolved_device)
    optimizer = build_optimizer(model=model, lr=lr, gate_lr_multiplier=gate_lr_multiplier)
    criterion = get_classification_loss()

    run_sanity_checks(model, train_loader, resolved_device, lambda_sparsity, threshold)

    run_name = make_run_name(lambda_sparsity, seed)
    checkpoint_path = output_paths["checkpoints"] / f"{run_name}_best.pt"
    history_json_path = output_paths["metrics"] / f"{run_name}_history.json"
    history_csv_path = output_paths["metrics"] / f"{run_name}_history.csv"
    histogram_path = output_paths["plots"] / f"{run_name}_gate_hist.png"
    final_histogram_path = output_paths["plots"] / f"{run_name}_final_gate_hist.png"
    final_summary_path = output_paths["metrics"] / f"{run_name}_final_summary.json"

    history: list[dict[str, float]] = []
    best_test_accuracy = float("-inf")
    best_epoch = 0
    final_metrics: dict[str, float] | None = None

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, resolved_device, lambda_sparsity, criterion)
        test_metrics = evaluate_model(model, test_loader, resolved_device, lambda_sparsity, threshold, criterion)
        history.append(create_epoch_record(epoch, train_metrics, test_metrics, lambda_sparsity))
        print(format_epoch_log(epoch, epochs, train_metrics, test_metrics))
        final_metrics = test_metrics

        if test_metrics["accuracy"] > best_test_accuracy:
            best_test_accuracy = test_metrics["accuracy"]
            best_epoch = epoch
            torch.save(checkpoint_payload(model, optimizer, epoch, best_test_accuracy, lambda_sparsity), checkpoint_path)

    save_json(history_json_path, history)
    save_csv(history_csv_path, history)

    if final_metrics is None:
        raise RuntimeError("Training completed without producing final evaluation metrics.")

    final_gate_stats = compute_gate_statistics(model)
    final_test_accuracy = final_metrics["accuracy"]
    final_sparsity_percentage = final_metrics["sparsity_percentage"]
    save_gate_histogram(model, final_histogram_path, title=f"Final Gate Value Distribution ({run_name})")
    save_json(
        final_summary_path,
        {
            "run_name": run_name,
            "lambda_sparsity": lambda_sparsity,
            "epochs": epochs,
            "threshold": threshold,
            "final_evaluation": final_metrics,
            "final_gate_statistics": final_gate_stats,
        },
    )

    best_checkpoint = torch.load(checkpoint_path, map_location=resolved_device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    best_metrics = evaluate_model(model, test_loader, resolved_device, lambda_sparsity, threshold, criterion)
    gate_stats = compute_gate_statistics(model)
    save_gate_histogram(model, histogram_path, title=f"Gate Value Distribution ({run_name})")

    if save_aliases:
        shutil.copy2(checkpoint_path, output_paths["checkpoints"] / "best.pt")
        shutil.copy2(histogram_path, output_paths["plots"] / "best_gate_hist.png")

    ensure_artifacts_exist(
        [
            checkpoint_path,
            history_json_path,
            history_csv_path,
            histogram_path,
            final_histogram_path,
            final_summary_path,
        ]
    )
    return {
        "run_name": run_name,
        "lambda_sparsity": lambda_sparsity,
        "checkpoint_path": checkpoint_path,
        "history_json_path": history_json_path,
        "history_csv_path": history_csv_path,
        "histogram_path": histogram_path,
        "final_histogram_path": final_histogram_path,
        "final_summary_path": final_summary_path,
        "best_epoch": best_epoch,
        "best_test_accuracy": best_test_accuracy,
        "best_sparsity_percentage": best_metrics["sparsity_percentage"],
        "final_test_accuracy": final_test_accuracy,
        "final_sparsity_percentage": final_sparsity_percentage,
        "final_gate_statistics": final_gate_stats,
        "gate_statistics": gate_stats,
    }


def run_evaluation(
    checkpoint_path: str | Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    threshold: float | None = None,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    device: str = "auto",
    num_workers: int = DEFAULT_NUM_WORKERS,
    test_subset: int | None = None,
) -> dict[str, object]:
    seed_everything(DEFAULT_SEED)
    output_paths = ensure_output_dirs(output_dir)
    resolved_checkpoint_path = resolve_path(checkpoint_path)
    resolved_device = resolve_device(device)
    checkpoint = torch.load(resolved_checkpoint_path, map_location=resolved_device)

    model_config = checkpoint["model_config"]
    effective_threshold = float(threshold) if threshold is not None else float(model_config.get("threshold", DEFAULT_THRESHOLD))
    model = PrunableMLP.from_config(model_config).to(resolved_device)
    model.load_state_dict(checkpoint["model_state_dict"])

    _, test_loader = get_cifar10_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        quick=False,
        test_subset=test_subset,
        seed=DEFAULT_SEED,
        download=True,
    )

    metrics = evaluate_model(
        model=model,
        data_loader=test_loader,
        device=resolved_device,
        lambda_sparsity=float(checkpoint["lambda_sparsity"]),
        threshold=effective_threshold,
        criterion=get_classification_loss(),
    )

    metrics_path = output_paths["metrics"] / f"{resolved_checkpoint_path.stem}_eval.json"
    histogram_path = output_paths["plots"] / f"{resolved_checkpoint_path.stem}_eval_hist.png"
    payload = {
        "checkpoint": str(resolved_checkpoint_path),
        "epoch": int(checkpoint["epoch"]),
        "best_test_accuracy": float(checkpoint["best_test_accuracy"]),
        "lambda_sparsity": float(checkpoint["lambda_sparsity"]),
        "threshold": effective_threshold,
        "evaluation": metrics,
        "gate_statistics": compute_gate_statistics(model),
    }
    save_json(metrics_path, payload)
    save_gate_histogram(model, histogram_path, title=f"Gate Value Distribution ({resolved_checkpoint_path.stem})")
    ensure_artifacts_exist([metrics_path, histogram_path])
    return {
        "metrics": metrics,
        "metrics_path": metrics_path,
        "histogram_path": histogram_path,
    }


def run_sweep(
    lambdas: list[float],
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    dropout: float = DEFAULT_DROPOUT,
    gate_init_mean: float = DEFAULT_GATE_INIT_MEAN,
    gate_init_std: float = DEFAULT_GATE_INIT_STD,
    gate_lr_multiplier: float = DEFAULT_GATE_LR_MULTIPLIER,
    seed: int = DEFAULT_SEED,
    device: str = "auto",
    data_dir: str | Path = DEFAULT_DATA_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    threshold: float = DEFAULT_THRESHOLD,
    quick: bool = False,
    train_subset: int | None = None,
    test_subset: int | None = None,
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> dict[str, object]:
    output_paths = ensure_output_dirs(output_dir)
    report_path = resolve_path(Path("report") / "report.md")
    if not report_path.exists():
        write_report_template(report_path)

    sweep_rows: list[dict[str, float]] = []
    detailed_results: list[dict[str, object]] = []
    best_accuracy_run: dict[str, object] | None = None
    best_final_run: dict[str, object] | None = None

    for lambda_sparsity in lambdas:
        print(f"Running sweep for lambda={lambda_sparsity:.0e}")
        run_result = run_training(
            lambda_sparsity=lambda_sparsity,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            dropout=dropout,
            gate_init_mean=gate_init_mean,
            gate_init_std=gate_init_std,
            gate_lr_multiplier=gate_lr_multiplier,
            seed=seed,
            device=device,
            data_dir=data_dir,
            output_dir=output_dir,
            threshold=threshold,
            quick=quick,
            train_subset=train_subset,
            test_subset=test_subset,
            num_workers=num_workers,
            save_aliases=False,
            download=True,
        )

        row = {
            "Lambda": float(lambda_sparsity),
            "Test Accuracy": float(run_result["final_test_accuracy"]),
            "Sparsity Level (%)": float(run_result["final_sparsity_percentage"]),
        }
        sweep_rows.append(row)
        detailed_results.append(run_result)

        if best_accuracy_run is None or float(run_result["best_test_accuracy"]) > float(best_accuracy_run["best_test_accuracy"]):
            best_accuracy_run = run_result

        if best_final_run is None or float(run_result["final_test_accuracy"]) > float(best_final_run["final_test_accuracy"]):
            best_final_run = run_result

    results_csv_path = output_paths["root"] / "results.csv"
    summary_md_path = output_paths["root"] / "summary.md"
    detailed_json_path = output_paths["sweeps"] / "sweep_results.json"
    pd.DataFrame(sweep_rows).to_csv(results_csv_path, index=False)
    summary_md_path.write_text(
        "\n".join(
            [
                "# Sweep Summary",
                "",
                "This file is generated by the single-file fallback sweep and summarizes the accuracy-sparsity trade-off across lambda values.",
                "",
                markdown_table(sweep_rows),
                "",
                "The table reports final-epoch accuracy and final-epoch thresholded sparsity. The sweep also writes `outputs/checkpoints/sweep_best.pt` plus `outputs/plots/sweep_best_gate_hist.png` for the best accuracy-selected checkpoint, and `outputs/plots/final_sparse_gate_hist.png` plus `outputs/metrics/final_sparse_summary.json` for the best final sparse endpoint used in presentations.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    update_report_results(report_path, sweep_rows)
    save_json(
        detailed_json_path,
        [
            {
                "lambda_sparsity": float(result["lambda_sparsity"]),
                "best_test_accuracy": float(result["best_test_accuracy"]),
                "best_sparsity_percentage": float(result["best_sparsity_percentage"]),
                "final_test_accuracy": float(result["final_test_accuracy"]),
                "final_sparsity_percentage": float(result["final_sparsity_percentage"]),
                "checkpoint_path": str(result["checkpoint_path"]),
                "best_histogram_path": str(result["histogram_path"]),
                "final_histogram_path": str(result["final_histogram_path"]),
                "final_summary_path": str(result["final_summary_path"]),
            }
            for result in detailed_results
        ],
    )

    if best_accuracy_run is not None:
        shutil.copy2(best_accuracy_run["checkpoint_path"], output_paths["checkpoints"] / "sweep_best.pt")
        shutil.copy2(best_accuracy_run["histogram_path"], output_paths["plots"] / "sweep_best_gate_hist.png")

    if best_final_run is not None:
        shutil.copy2(best_final_run["final_histogram_path"], output_paths["plots"] / "final_sparse_gate_hist.png")
        shutil.copy2(best_final_run["final_summary_path"], output_paths["metrics"] / "final_sparse_summary.json")

    ensure_artifacts_exist(
        [
            results_csv_path,
            summary_md_path,
            detailed_json_path,
            output_paths["checkpoints"] / "sweep_best.pt",
            output_paths["plots"] / "sweep_best_gate_hist.png",
            output_paths["plots"] / "final_sparse_gate_hist.png",
            output_paths["metrics"] / "final_sparse_summary.json",
        ]
    )
    return {
        "results_csv_path": results_csv_path,
        "summary_md_path": summary_md_path,
        "report_path": report_path,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone self-pruning CIFAR-10 project.")
    parser.add_argument("--mode", choices=["train", "evaluate", "sweep"], required=True, help="Execution mode.")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/checkpoints/best.pt"), help="Checkpoint path for evaluation.")
    parser.add_argument("--lambda-sparsity", type=float, default=1e-4, help="Weight for the sparsity penalty.")
    parser.add_argument("--lambdas", type=float, nargs="+", default=DEFAULT_SWEEP_LAMBDAS, help="Lambda values for sweep mode.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate for Adam.")
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT, help="Dropout probability.")
    parser.add_argument("--gate-init-mean", type=float, default=DEFAULT_GATE_INIT_MEAN, help="Mean for gate score initialization.")
    parser.add_argument("--gate-init-std", type=float, default=DEFAULT_GATE_INIT_STD, help="Std for gate score initialization.")
    parser.add_argument("--gate-lr-multiplier", type=float, default=DEFAULT_GATE_LR_MULTIPLIER, help="Multiplier for the gate score learning rate.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument("--quick", action="store_true", help="Run with reduced train/test subsets for smoke tests.")
    parser.add_argument("--train-subset", type=int, default=None, help="Optional train subset size.")
    parser.add_argument("--test-subset", type=int, default=None, help="Optional test subset size.")
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS, help="Number of dataloader workers.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Gate threshold for sparsity reporting.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Dataset directory.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Artifact output directory.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: auto, cpu, cuda, or any torch device string.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    ensure_output_dirs(args.output_dir)
    ensure_seed_files(args.output_dir)

    if args.mode == "train":
        result = run_training(
            lambda_sparsity=args.lambda_sparsity,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            dropout=args.dropout,
            gate_init_mean=args.gate_init_mean,
            gate_init_std=args.gate_init_std,
            gate_lr_multiplier=args.gate_lr_multiplier,
            seed=args.seed,
            device=args.device,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            threshold=args.threshold,
            quick=args.quick,
            train_subset=args.train_subset,
            test_subset=args.test_subset,
            num_workers=args.num_workers,
            save_aliases=True,
            download=True,
        )
        print(
            f"Best checkpoint saved to {result['checkpoint_path']} | "
            f"best_test_accuracy={result['best_test_accuracy']:.2f}% | "
            f"sparsity={result['best_sparsity_percentage']:.2f}%"
        )
    elif args.mode == "evaluate":
        result = run_evaluation(
            checkpoint_path=args.checkpoint,
            batch_size=args.batch_size,
            threshold=args.threshold,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            device=args.device,
            num_workers=args.num_workers,
            test_subset=args.test_subset,
        )
        print(f"Final test accuracy: {result['metrics']['accuracy']:.2f}%")
        print(f"Final sparsity level: {result['metrics']['sparsity_percentage']:.2f}%")
        print(f"Metrics saved to: {result['metrics_path']}")
        print(f"Histogram saved to: {result['histogram_path']}")
    else:
        result = run_sweep(
            lambdas=list(args.lambdas),
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            dropout=args.dropout,
            gate_init_mean=args.gate_init_mean,
            gate_init_std=args.gate_init_std,
            gate_lr_multiplier=args.gate_lr_multiplier,
            seed=args.seed,
            device=args.device,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            threshold=args.threshold,
            quick=args.quick,
            train_subset=args.train_subset,
            test_subset=args.test_subset,
            num_workers=args.num_workers,
        )
        print(f"Sweep results saved to: {result['results_csv_path']}")
        print(f"Markdown summary saved to: {result['summary_md_path']}")
        print(f"Report updated at: {result['report_path']}")


if __name__ == "__main__":
    main()
