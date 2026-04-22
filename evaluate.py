from __future__ import annotations

import argparse
from pathlib import Path

import torch

from config import DEFAULT_BATCH_SIZE, DEFAULT_DATA_DIR, DEFAULT_NUM_WORKERS, DEFAULT_OUTPUT_DIR, DEFAULT_THRESHOLD, ensure_output_dirs, resolve_path
from models import PrunableMLP
from train import evaluate_model, resolve_device
from utils import (
    compute_gate_statistics,
    ensure_artifacts_exist,
    get_cifar10_loaders,
    get_classification_loss,
    save_gate_histogram,
    save_json,
    seed_everything,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained self-pruning MLP checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a saved checkpoint.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Evaluation batch size.")
    parser.add_argument("--threshold", type=float, default=None, help="Threshold for sparsity reporting.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Dataset directory.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Artifact output directory.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: auto, cpu, cuda, or any torch device string.",
    )
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS, help="Number of dataloader workers.")
    parser.add_argument("--test-subset", type=int, default=None, help="Optional test subset size for smoke tests.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    seed_everything(42)
    output_paths = ensure_output_dirs(resolve_path(args.output_dir))

    checkpoint_path = resolve_path(args.checkpoint)
    device = resolve_device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint["model_config"]

    threshold = float(args.threshold) if args.threshold is not None else float(model_config.get("threshold", DEFAULT_THRESHOLD))
    model = PrunableMLP.from_config(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    _, test_loader = get_cifar10_loaders(
        data_dir=resolve_path(args.data_dir),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        quick=False,
        train_subset=None,
        test_subset=args.test_subset,
        seed=42,
        download=True,
    )

    metrics = evaluate_model(
        model=model,
        data_loader=test_loader,
        device=device,
        lambda_sparsity=float(checkpoint["lambda_sparsity"]),
        threshold=threshold,
        criterion=get_classification_loss(),
    )

    eval_payload = {
        "checkpoint": str(checkpoint_path),
        "epoch": int(checkpoint["epoch"]),
        "best_test_accuracy": float(checkpoint["best_test_accuracy"]),
        "lambda_sparsity": float(checkpoint["lambda_sparsity"]),
        "threshold": threshold,
        "evaluation": metrics,
        "gate_statistics": compute_gate_statistics(model),
    }
    metrics_path = output_paths["metrics"] / f"{checkpoint_path.stem}_eval.json"
    histogram_path = output_paths["plots"] / f"{checkpoint_path.stem}_eval_hist.png"
    save_json(metrics_path, eval_payload)
    save_gate_histogram(model, histogram_path, title=f"Gate Value Distribution ({checkpoint_path.stem})")
    ensure_artifacts_exist([metrics_path, histogram_path])

    print(f"Final test accuracy: {metrics['accuracy']:.2f}%")
    print(f"Final sparsity level: {metrics['sparsity_percentage']:.2f}%")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Histogram saved to: {histogram_path}")


if __name__ == "__main__":
    main()
