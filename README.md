# Self-Pruning CIFAR-10 PyTorch Project

This repository contains my solution for the Tredence self-pruning neural network case study.

The core idea is simple: instead of pruning a trained model afterward, the network learns during training which connections it can afford to suppress. I implemented that by attaching a learnable gate to every weight in each linear layer, then adding a sparsity penalty so the model is encouraged to close unnecessary gates over time.

## What Is Included

- A modular PyTorch implementation with separate model, data, loss, metric, plotting, and training files.
- A standalone fallback script, [main_single_file.py](/D:/AI%20intern%20project/main_single_file.py), in case a one-file submission is preferred.
- Checked-in lightweight artifacts that show the final sweep table and one representative gate-distribution plot.

## Setup

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Main Commands

Train one run:

```bash
python train.py --lambda-sparsity 1e-4 --epochs 15
```

Evaluate a saved checkpoint:

```bash
python evaluate.py --checkpoint outputs/checkpoints/best.pt
```

The checkpoint itself is not checked into the repository anymore, to keep the repo lightweight. Running `train.py` will create `outputs/checkpoints/best.pt`, after which the evaluation command works as shown.

Run the sweep:

```bash
python sweep.py
```

Use the single-file fallback:

```bash
python main_single_file.py --mode train --lambda-sparsity 1e-4 --epochs 15
python main_single_file.py --mode sweep
```

## Quick Smoke Runs

For fast checks on a smaller subset:

```bash
python train.py --lambda-sparsity 1e-4 --epochs 1 --quick
python sweep.py --epochs 1 --quick
python main_single_file.py --mode train --lambda-sparsity 1e-4 --epochs 1 --quick
```

You can also override the subset sizes directly:

```bash
python train.py --lambda-sparsity 1e-4 --epochs 1 --train-subset 512 --test-subset 256
```

## How the Model Works

- `PrunableLinear` keeps the usual `weight` and `bias`, but also adds a `gate_scores` tensor with the same shape as the weight matrix.
- During the forward pass, `sigmoid(gate_scores)` produces gate values between `0` and `1`.
- The layer uses `weight * gates` instead of the raw weight matrix.
- Training optimizes:

```text
CrossEntropyLoss + lambda_sparsity * sparsity_loss
```

- `sparsity_loss` is the sum of all sigmoid gate values across all prunable layers.
- Reported sparsity is the percentage of gates below the threshold `1e-2`.

The classifier itself is a small MLP:

```text
3072 -> 1024 -> 512 -> 10
```

with ReLU activations and optional dropout.

## Checked-In Results

The checked-in numbers are still lightweight validation runs, not full-dataset benchmark results. I kept them small enough to reproduce on a normal laptop while still showing the sparsity-versus-accuracy trade-off clearly.

The current curated sweep used:

- `15` epochs
- batch size `64`
- train subset `4096`
- test subset `1000`
- lambdas `2e-4`, `5e-4`, and `1e-3`
- gate init mean `0.5`
- gate-score learning-rate multiplier `15`

Exact command:

```bash
python sweep.py --epochs 15 --batch-size 64 --train-subset 4096 --test-subset 1000 --lambdas 2e-4 5e-4 1e-3 --gate-init-mean 0.5 --gate-lr-multiplier 15 --device cpu
```

Current sweep table:

| Lambda | Test Accuracy | Sparsity Level (%) |
| --- | ---: | ---: |
| 2e-04 | 40.90 | 93.84 |
| 5e-04 | 37.10 | 98.09 |
| 1e-03 | 34.20 | 99.33 |

## How To Read the Checked-In Artifacts

The repository now keeps only the lighter checked-in outputs that are useful for review:

- [outputs/results.csv](/D:/AI%20intern%20project/outputs/results.csv) is the sweep table in CSV form.
- [outputs/summary.md](/D:/AI%20intern%20project/outputs/summary.md) is the same result in a quick readable Markdown format.
- [outputs/metrics/final_sparse_summary.json](/D:/AI%20intern%20project/outputs/metrics/final_sparse_summary.json) records the final sparse endpoint for the best-performing run in the checked-in sweep.
- [outputs/plots/final_sparse_gate_hist.png](/D:/AI%20intern%20project/outputs/plots/final_sparse_gate_hist.png) is the representative histogram for that final sparse endpoint.

I removed the checked-in checkpoint and duplicate evaluation artifacts because they were not required by the case study itself and made the repository heavier than it needed to be. They can still be regenerated locally by rerunning training or evaluation.

## Folder Structure

```text
.
|-- README.md
|-- requirements.txt
|-- config.py
|-- train.py
|-- evaluate.py
|-- sweep.py
|-- main_single_file.py
|-- models/
|   |-- __init__.py
|   |-- prunable_layers.py
|   `-- mlp.py
|-- utils/
|   |-- __init__.py
|   |-- data.py
|   |-- losses.py
|   |-- metrics.py
|   |-- plotting.py
|   `-- seed.py
|-- outputs/
|   |-- checkpoints/
|   |-- plots/
|   |-- metrics/
|   |-- sweeps/
|   |-- results.csv
|   `-- summary.md
`-- report/
    `-- report.md
```

## Files To Check First

- Report: [report/report.md](/D:/AI%20intern%20project/report/report.md)
- Sweep table: [outputs/results.csv](/D:/AI%20intern%20project/outputs/results.csv)
- Human-readable sweep summary: [outputs/summary.md](/D:/AI%20intern%20project/outputs/summary.md)
- Final sparse endpoint: [outputs/metrics/final_sparse_summary.json](/D:/AI%20intern%20project/outputs/metrics/final_sparse_summary.json)
- Representative plot: [outputs/plots/final_sparse_gate_hist.png](/D:/AI%20intern%20project/outputs/plots/final_sparse_gate_hist.png)

## Submission Checklist

- Code location: project root plus [models](/D:/AI%20intern%20project/models) and [utils](/D:/AI%20intern%20project/utils)
- Single-file fallback: [main_single_file.py](/D:/AI%20intern%20project/main_single_file.py)
- Report location: [report/report.md](/D:/AI%20intern%20project/report/report.md)
- Plot location: [outputs/plots](/D:/AI%20intern%20project/outputs/plots)
- Sweep reproduction command:

```bash
python sweep.py --epochs 15 --batch-size 64 --train-subset 4096 --test-subset 1000 --lambdas 2e-4 5e-4 1e-3 --gate-init-mean 0.5 --gate-lr-multiplier 15 --device cpu
```
