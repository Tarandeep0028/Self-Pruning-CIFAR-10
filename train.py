from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DATA_DIR,
    DEFAULT_DROPOUT,
    DEFAULT_EPOCHS,
    DEFAULT_GATE_INIT_MEAN,
    DEFAULT_GATE_INIT_STD,
    DEFAULT_GATE_LR_MULTIPLIER,
    DEFAULT_LR,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_NUM_WORKERS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SEED,
    DEFAULT_THRESHOLD,
    ensure_output_dirs,
    resolve_path,
)
from models import (
    PrunableMLP,
    collect_gate_tensors,
    compute_sparsity_percentage,
    compute_total_sparsity_loss,
    iter_prunable_layers,
)
from utils import (
    RunningAverage,
    count_correct_predictions,
    compute_gate_statistics,
    create_epoch_record,
    ensure_artifacts_exist,
    format_epoch_log,
    get_cifar10_loaders,
    get_classification_loss,
    make_run_name,
    save_csv,
    save_gate_histogram,
    save_json,
    seed_everything,
    summarize_run_result,
)
from utils.losses import compute_total_loss


def resolve_device(device: str) -> torch.device:
    """Resolve a CLI device string to a torch.device."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def run_sanity_checks(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    lambda_sparsity: float,
    threshold: float,
) -> None:
    """Validate the most important model and pruning invariants before training."""
    model.train()
    batch_inputs, batch_targets = next(iter(train_loader))
    batch_inputs = batch_inputs.to(device)
    batch_targets = batch_targets.to(device)

    criterion = get_classification_loss()
    logits = model(batch_inputs)
    expected_output_dim = int(getattr(model, "output_dim", DEFAULT_MODEL_CONFIG["output_dim"]))
    expected_shape = (batch_inputs.size(0), expected_output_dim)
    if logits.shape != expected_shape:
        raise AssertionError(f"Expected logits shape {expected_shape}, but received {tuple(logits.shape)}")

    classification_loss, sparsity_loss, total_loss = compute_total_loss(
        logits=logits,
        targets=batch_targets,
        model=model,
        lambda_sparsity=lambda_sparsity,
        criterion=criterion,
    )

    gate_tensors = collect_gate_tensors(model)
    if not gate_tensors:
        raise AssertionError("Expected at least one prunable layer, but no gates were collected.")

    manual_sparsity_loss = torch.stack([gate.sum() for gate in gate_tensors]).sum()
    if not torch.allclose(sparsity_loss, manual_sparsity_loss):
        raise AssertionError("Sparsity loss helper does not match the manual gate sum.")

    total_loss.backward()
    for layer in iter_prunable_layers(model):
        if layer.weight.grad is None:
            raise AssertionError("Expected gradients for weight parameters during the sanity check.")
        if layer.gate_scores.grad is None:
            raise AssertionError("Expected gradients for gate_scores during the sanity check.")

    manual_sparsity_pct = 100.0 * sum(int((gate.detach() < threshold).sum().item()) for gate in gate_tensors) / sum(
        gate.numel() for gate in gate_tensors
    )
    helper_sparsity_pct = compute_sparsity_percentage(model, threshold=threshold)
    if abs(manual_sparsity_pct - helper_sparsity_pct) > 1e-9:
        raise AssertionError("Sparsity percentage helper does not match the manual threshold count.")

    recomputed_loss = compute_total_sparsity_loss(model)
    if not torch.allclose(recomputed_loss, manual_sparsity_loss):
        raise AssertionError("Total sparsity helper does not match the manual gate sum.")

    model.zero_grad(set_to_none=True)


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    lambda_sparsity: float,
    criterion: nn.Module,
) -> dict[str, float]:
    """Train the model for one epoch and return averaged metrics."""
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
    """Evaluate the model on a dataloader and report losses, accuracy, and sparsity."""
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
        "sparsity_percentage": compute_sparsity_percentage(model, threshold=threshold),
    }


def _checkpoint_payload(
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
    """Build Adam with an optional higher learning rate for gate score parameters."""
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
    """Train one self-pruning MLP run and persist its artifacts."""
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
        input_dim=int(DEFAULT_MODEL_CONFIG["input_dim"]),
        hidden_dims=tuple(DEFAULT_MODEL_CONFIG["hidden_dims"]),
        output_dim=int(DEFAULT_MODEL_CONFIG["output_dim"]),
        dropout=dropout,
        threshold=threshold,
        gate_init_mean=gate_init_mean,
        gate_init_std=gate_init_std,
    ).to(resolved_device)
    optimizer = build_optimizer(model=model, lr=lr, gate_lr_multiplier=gate_lr_multiplier)
    criterion = get_classification_loss()

    run_sanity_checks(
        model=model,
        train_loader=train_loader,
        device=resolved_device,
        lambda_sparsity=lambda_sparsity,
        threshold=threshold,
    )

    run_name = make_run_name(lambda_sparsity=lambda_sparsity, seed=seed)
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
        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=resolved_device,
            lambda_sparsity=lambda_sparsity,
            criterion=criterion,
        )
        test_metrics = evaluate_model(
            model=model,
            data_loader=test_loader,
            device=resolved_device,
            lambda_sparsity=lambda_sparsity,
            threshold=threshold,
            criterion=criterion,
        )
        history.append(create_epoch_record(epoch, train_metrics, test_metrics, lambda_sparsity))
        print(format_epoch_log(epoch, epochs, train_metrics, test_metrics))
        final_metrics = test_metrics

        if test_metrics["accuracy"] > best_test_accuracy:
            best_test_accuracy = test_metrics["accuracy"]
            best_epoch = epoch
            torch.save(
                _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_test_accuracy=best_test_accuracy,
                    lambda_sparsity=lambda_sparsity,
                ),
                checkpoint_path,
            )

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
    best_metrics = evaluate_model(
        model=model,
        data_loader=test_loader,
        device=resolved_device,
        lambda_sparsity=lambda_sparsity,
        threshold=threshold,
        criterion=criterion,
    )
    gate_stats = compute_gate_statistics(model)
    save_gate_histogram(model, histogram_path, title=f"Gate Value Distribution ({run_name})")

    if save_aliases:
        alias_checkpoint = output_paths["checkpoints"] / "best.pt"
        alias_histogram = output_paths["plots"] / "best_gate_hist.png"
        shutil.copy2(checkpoint_path, alias_checkpoint)
        shutil.copy2(histogram_path, alias_histogram)

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
        "gate_statistics": gate_stats,
        "final_gate_statistics": final_gate_stats,
        "model_config": model.to_config(),
        "summary": summarize_run_result(
            lambda_sparsity=lambda_sparsity,
            best_test_accuracy=best_test_accuracy,
            sparsity_percentage=best_metrics["sparsity_percentage"],
            checkpoint_path=checkpoint_path,
            histogram_path=histogram_path,
        ),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a self-pruning MLP on CIFAR-10.")
    parser.add_argument("--lambda-sparsity", type=float, required=True, help="Weight for the sparsity penalty.")
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
    results = run_training(
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
        data_dir=resolve_path(args.data_dir),
        output_dir=resolve_path(args.output_dir),
        threshold=args.threshold,
        quick=args.quick,
        train_subset=args.train_subset,
        test_subset=args.test_subset,
        num_workers=args.num_workers,
        save_aliases=True,
        download=True,
    )
    print(
        f"Best checkpoint saved to {results['checkpoint_path']} | "
        f"best_test_accuracy={results['best_test_accuracy']:.2f}% | "
        f"sparsity={results['best_sparsity_percentage']:.2f}%"
    )


if __name__ == "__main__":
    main()
