"""Train and evaluate the spiking network.

Phase 1 (ideal synapses):
    python -m src.train --epochs 5 --num-steps 25

Phase 2 (memristor-aware training):
    python -m src.train --epochs 5 --device-preset synthetic

For the naive-vs-aware comparison and the accuracy-vs-variability sweep, see
``python -m src.experiment``.
"""

from __future__ import annotations

import argparse
import time

import torch
import snntorch.functional as SF
from tqdm import tqdm

from .data import load_dataset
from .memristor import PRESETS, DeviceConfig
from .model import SpikingMLP
from .utils import plot_history, save_metrics, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a LIF SNN (ideal or memristor-aware).")
    p.add_argument("--dataset", default="mnist")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-hidden", type=int, default=256)
    p.add_argument("--num-steps", type=int, default=25)
    p.add_argument("--beta", type=float, default=0.95)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--results-dir", default="./results")
    # -- memristor-aware (Phase 2) --
    p.add_argument(
        "--device-preset",
        default=None,
        choices=sorted(PRESETS),
        help="memristor device preset; omit for the ideal nn.Linear baseline.",
    )
    p.add_argument("--n-levels", type=int, default=None, help="override preset n_levels.")
    p.add_argument("--sigma", type=float, default=None, help="override preset sigma.")
    # -- fast iteration --
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--max-test-batches", type=int, default=None)
    return p.parse_args()


def build_device_cfg(
    preset: str | None,
    n_levels: int | None = None,
    sigma: float | None = None,
) -> DeviceConfig | None:
    """Resolve a DeviceConfig from a preset name + optional overrides (None = ideal baseline)."""
    if preset is None:
        return None
    cfg = PRESETS[preset]
    cfg = DeviceConfig(
        n_levels=n_levels if n_levels is not None else cfg.n_levels,
        sigma=sigma if sigma is not None else cfg.sigma,
        sigma_read=cfg.sigma_read,
        name=cfg.name,
    )
    return cfg


@torch.no_grad()
def evaluate(model, loader, device: str, max_batches: int | None = None) -> float:
    """Return classification accuracy (spike-count rule) over a loader."""
    model.eval()
    correct, total = 0.0, 0
    for i, (data, targets) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        data = data.view(data.size(0), -1).to(device)
        targets = targets.to(device)
        spk_rec, _ = model(data)
        # predict the class whose output neuron spiked most across time
        correct += SF.accuracy_rate(spk_rec, targets) * targets.size(0)
        total += targets.size(0)
    return correct / max(total, 1)


def train_model(
    model,
    train_loader,
    test_loader,
    *,
    epochs: int,
    lr: float,
    device: str,
    max_train_batches: int | None = None,
    max_test_batches: int | None = None,
    label: str = "",
) -> dict:
    """Train ``model`` end-to-end (surrogate-gradient BPTT) and return a history dict."""
    loss_fn = SF.ce_rate_loss()  # cross-entropy on output spike counts
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    history = {"train_loss": [], "test_acc": []}
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, n_batches = 0.0, 0
        desc = f"{label + ' ' if label else ''}epoch {epoch}/{epochs}"
        for i, (data, targets) in enumerate(tqdm(train_loader, desc=desc)):
            if max_train_batches is not None and i >= max_train_batches:
                break
            data = data.view(data.size(0), -1).to(device)
            targets = targets.to(device)

            spk_rec, _ = model(data)
            loss = loss_fn(spk_rec, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        train_loss = running_loss / max(n_batches, 1)
        test_acc = evaluate(model, test_loader, device, max_batches=max_test_batches)
        history["train_loss"].append(train_loss)
        history["test_acc"].append(test_acc)
        print(f"{desc}: train_loss={train_loss:.4f}  test_acc={test_acc:.4f}")
    return history


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = args.device
    print(f"Device: {device}")

    cfg = build_device_cfg(args.device_preset, args.n_levels, args.sigma)
    print(f"Synapse: {'ideal nn.Linear' if cfg is None else f'memristor ({cfg})'}")

    train_loader, test_loader = load_dataset(
        args.dataset, data_dir=args.data_dir, batch_size=args.batch_size
    )

    model = SpikingMLP(
        num_hidden=args.num_hidden,
        num_steps=args.num_steps,
        beta=args.beta,
        device_cfg=cfg,
    ).to(device)

    start = time.time()
    history = train_model(
        model,
        train_loader,
        test_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        max_train_batches=args.max_train_batches,
        max_test_batches=args.max_test_batches,
    )
    elapsed = time.time() - start
    best_acc = max(history["test_acc"]) if history["test_acc"] else 0.0

    metrics = {
        "dataset": args.dataset,
        "synapse": "ideal" if cfg is None else cfg.name,
        "config": vars(args),
        "history": history,
        "best_test_acc": best_acc,
        "train_seconds": round(elapsed, 1),
    }
    save_metrics(metrics, f"{args.results_dir}/metrics.json")
    plot_history(history, f"{args.results_dir}/training_curve.png")
    print(f"\nBest test accuracy: {best_acc:.4f}")
    print(f"Saved metrics + curve to {args.results_dir}/")


if __name__ == "__main__":
    main()
