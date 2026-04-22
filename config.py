from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = Path("data")
DEFAULT_OUTPUT_DIR = Path("outputs")
OUTPUT_SUBDIRS = ("checkpoints", "plots", "metrics", "sweeps")

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

DEFAULT_THRESHOLD = 1e-2
DEFAULT_INPUT_DIM = 3 * 32 * 32
DEFAULT_HIDDEN_DIMS = (1024, 512)
DEFAULT_OUTPUT_DIM = 10
DEFAULT_DROPOUT = 0.1
DEFAULT_GATE_INIT_MEAN = 2.0
DEFAULT_GATE_INIT_STD = 0.01

DEFAULT_MODEL_CONFIG = {
    "input_dim": DEFAULT_INPUT_DIM,
    "hidden_dims": list(DEFAULT_HIDDEN_DIMS),
    "output_dim": DEFAULT_OUTPUT_DIM,
    "dropout": DEFAULT_DROPOUT,
    "threshold": DEFAULT_THRESHOLD,
    "gate_init_mean": DEFAULT_GATE_INIT_MEAN,
    "gate_init_std": DEFAULT_GATE_INIT_STD,
}

DEFAULT_SWEEP_LAMBDAS = [1e-5, 5e-5, 1e-4]
DEFAULT_EPOCHS = 15
DEFAULT_BATCH_SIZE = 128
DEFAULT_LR = 1e-3
DEFAULT_GATE_LR_MULTIPLIER = 1.0
DEFAULT_SEED = 42
DEFAULT_NUM_WORKERS = 0

QUICK_TRAIN_SUBSET = 2048
QUICK_TEST_SUBSET = 1024


def prepare_runtime_environment() -> dict[str, Path]:
    """Configure project-local runtime directories for temp and cache files."""
    temp_dir = PROJECT_ROOT / ".tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    torch_cache_dir = PROJECT_ROOT / ".torch_cache"
    torch_cache_dir.mkdir(parents=True, exist_ok=True)

    mpl_config_dir = PROJECT_ROOT / ".mplconfig"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)

    for key in ("TMP", "TEMP", "TMPDIR"):
        os.environ.setdefault(key, str(temp_dir))
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(torch_cache_dir))
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    return {
        "temp_dir": temp_dir,
        "torch_cache_dir": torch_cache_dir,
        "mpl_config_dir": mpl_config_dir,
    }


RUNTIME_DIRS = prepare_runtime_environment()


def resolve_path(path: str | Path) -> Path:
    """Resolve a possibly relative path from the project root."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def ensure_output_dirs(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Path]:
    """Create and return the output directory tree used by the project."""
    root = resolve_path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    directories = {"root": root}
    for name in OUTPUT_SUBDIRS:
        path = root / name
        path.mkdir(parents=True, exist_ok=True)
        directories[name] = path
    return directories
