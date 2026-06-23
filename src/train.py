"""Train and evaluate the Phase-1 spiking baseline.

Example:
    python -m src.train --epochs 5 --num-steps 25
"""

from __future__ import annotations

import argparse
import time

import torch
import snntorch.functional as SF
from tqdm import tqdm

from .data import load_dataset
from .model import SpikingMLP
from .utils import plot_history, save_metrics, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a Phase-1 LIF SNN baseline.")
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
    return p.parse_args()


@torch.no_grad()
def evaluate(model: SpikingMLP, loader, device: str) -> float:
    """Return classification accuracy (spike-count rule) over a loader."""
    model.eval()
    correct, total = 0, 0
    for data, targets in loader:
        data = data.view(data.size(0), -1).to(device)
        targets = targets.to(device)
        spk_rec, _ = model(data)
        # predict the class whose output neuron spiked most across time
        correct += SF.accuracy_rate(spk_rec, targets) * targets.size(0)
        total += targets.size(0)
    return correct / total


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = args.device
    print(f"Device: {device}")

    train_loader, test_loader = load_dataset(
        args.dataset, data_dir=args.data_dir, batch_size=args.batch_size
    )

    model = SpikingMLP(
        num_hidden=args.num_hidden, num_steps=args.num_steps, beta=args.beta
    ).to(device)

    loss_fn = SF.ce_rate_loss()  # cross-entropy on output spike counts
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    history = {"train_loss": [], "test_acc": []}
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for data, targets in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
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
        test_acc = evaluate(model, test_loader, device)
        history["train_loss"].append(train_loss)
        history["test_acc"].append(test_acc)
        print(f"epoch {epoch}: train_loss={train_loss:.4f}  test_acc={test_acc:.4f}")

    elapsed = time.time() - start
    best_acc = max(history["test_acc"]) if history["test_acc"] else 0.0

    metrics = {
        "dataset": args.dataset,
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
