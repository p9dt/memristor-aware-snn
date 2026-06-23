"""Small helpers: reproducibility, metrics I/O, and plotting."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_metrics(metrics: dict, path: str | Path) -> None:
    """Write a metrics dict to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def plot_history(history: dict, path: str | Path) -> None:
    """Plot training loss and test accuracy over epochs to ``path``."""
    import matplotlib

    matplotlib.use("Agg")  # headless-safe
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(epochs, history["train_loss"], marker="o")
    ax1.set_title("Training loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")

    ax2.plot(epochs, history["test_acc"], marker="o", color="tab:green")
    ax2.set_title("Test accuracy")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
