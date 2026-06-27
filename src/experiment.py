"""Phase-2 headline experiment: naive vs. memristor-aware training.

Two networks with identical memristor-crossbar architecture:

* **naive**  -- trained with the crossbar in *ideal* mode (the Phase-1 story:
  train clean, deploy on hardware and hope).
* **aware**  -- trained with device non-idealities injected every forward pass.

Both are then deployed on the *same* non-ideal device and evaluated across a sweep
of programming-variability ``sigma``. The gap between the curves is the whole point
of memristor-aware training.

    python -m src.experiment --epochs 3 --device-preset synthetic

Outputs (to ``results/``):
    * accuracy_vs_sigma.png   -- the headline figure
    * experiment_metrics.json -- every number behind it
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .data import load_dataset
from .memristor import PRESETS, DeviceConfig
from .model import SpikingMLP
from .train import build_device_cfg, evaluate, train_model
from .utils import save_metrics, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Naive vs memristor-aware SNN experiment.")
    p.add_argument("--dataset", default="mnist")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-hidden", type=int, default=256)
    p.add_argument("--num-steps", type=int, default=20)
    p.add_argument("--beta", type=float, default=0.95)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--results-dir", default="./results")
    p.add_argument("--device-preset", default="synthetic", choices=sorted(PRESETS))
    p.add_argument(
        "--sigma-sweep",
        type=float,
        nargs="+",
        default=[0.0, 0.02, 0.05, 0.08, 0.12, 0.18, 0.25],
        help="test-time programming-variability values to sweep.",
    )
    p.add_argument("--sweep-trials", type=int, default=3, help="MC repeats per sigma.")
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--max-test-batches", type=int, default=None)
    return p.parse_args()


def sweep_accuracy(
    model: SpikingMLP,
    loader,
    device: str,
    sigmas: list[float],
    trials: int,
    max_batches: int | None,
) -> dict[str, list[float]]:
    """Deploy ``model`` on the non-ideal crossbar; sweep sigma, averaging over trials."""
    model.set_ideal(False)
    means, stds = [], []
    for s in sigmas:
        model.set_sigma(s)
        accs = [
            evaluate(model, loader, device, max_batches=max_batches)
            for _ in range(trials)
        ]
        t = torch.tensor(accs)
        means.append(t.mean().item())
        stds.append(t.std(unbiased=False).item())
        print(f"  sigma={s:<5} acc={means[-1]:.4f} +/- {stds[-1]:.4f}")
    return {"sigma": sigmas, "acc_mean": means, "acc_std": stds}


def plot_sweep(naive: dict, aware: dict, ideal_refs: dict, path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, hist, color in [
        ("naive (trained ideal)", naive, "tab:red"),
        ("memristor-aware", aware, "tab:green"),
    ]:
        sig = hist["sigma"]
        mean = hist["acc_mean"]
        std = hist["acc_std"]
        ax.plot(sig, mean, marker="o", color=color, label=name)
        lo = [m - s for m, s in zip(mean, std)]
        hi = [m + s for m, s in zip(mean, std)]
        ax.fill_between(sig, lo, hi, color=color, alpha=0.15)

    ax.axhline(ideal_refs["naive"], ls="--", color="tab:red", alpha=0.5,
               label="naive, ideal crossbar")
    ax.axhline(ideal_refs["aware"], ls="--", color="tab:green", alpha=0.5,
               label="aware, ideal crossbar")

    ax.set_xlabel("test-time programming variability  sigma")
    ax.set_ylabel("test accuracy")
    ax.set_title("Memristor-aware vs. naive SNN under device variability")
    # auto-zoom to the data so the gap is visible (with a little padding)
    lows = naive["acc_mean"] + aware["acc_mean"]
    lo = max(0.0, min(lows) - 0.03)
    ax.set_ylim(lo, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = args.device
    print(f"Device: {device}")

    cfg: DeviceConfig = build_device_cfg(args.device_preset)
    print(f"Crossbar device: {cfg}\n")

    train_loader, test_loader = load_dataset(
        args.dataset, data_dir=args.data_dir, batch_size=args.batch_size
    )

    def make_model() -> SpikingMLP:
        return SpikingMLP(
            num_hidden=args.num_hidden,
            num_steps=args.num_steps,
            beta=args.beta,
            device_cfg=cfg,
        ).to(device)

    common = dict(
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        max_train_batches=args.max_train_batches,
        max_test_batches=args.max_test_batches,
    )

    start = time.time()

    # --- naive: same crossbar, but trained with the device in IDEAL mode ---
    print("=== Training NAIVE model (ideal synapses during training) ===")
    set_seed(args.seed)
    naive_model = make_model()
    naive_model.set_ideal(True)
    naive_hist = train_model(naive_model, train_loader, test_loader, label="naive", **common)

    # --- aware: device non-idealities injected during training ---
    print("\n=== Training AWARE model (device non-idealities in the loop) ===")
    set_seed(args.seed)
    aware_model = make_model()
    aware_model.set_ideal(False)
    aware_hist = train_model(aware_model, train_loader, test_loader, label="aware", **common)

    # --- ideal-crossbar reference accuracy for each ---
    naive_model.set_ideal(True)
    aware_model.set_ideal(True)
    ideal_refs = {
        "naive": evaluate(naive_model, test_loader, device, max_batches=args.max_test_batches),
        "aware": evaluate(aware_model, test_loader, device, max_batches=args.max_test_batches),
    }
    print(f"\nIdeal-crossbar accuracy -> naive={ideal_refs['naive']:.4f} "
          f"aware={ideal_refs['aware']:.4f}")

    # --- deploy on the non-ideal device; sweep variability ---
    print("\n=== Sweep: naive on non-ideal crossbar ===")
    naive_sweep = sweep_accuracy(
        naive_model, test_loader, device, args.sigma_sweep, args.sweep_trials,
        args.max_test_batches,
    )
    print("=== Sweep: aware on non-ideal crossbar ===")
    aware_sweep = sweep_accuracy(
        aware_model, test_loader, device, args.sigma_sweep, args.sweep_trials,
        args.max_test_batches,
    )

    elapsed = round(time.time() - start, 1)

    out_dir = Path(args.results_dir)
    plot_sweep(naive_sweep, aware_sweep, ideal_refs, out_dir / "accuracy_vs_sigma.png")

    metrics = {
        "device": cfg.__dict__,
        "config": vars(args),
        "ideal_crossbar_acc": ideal_refs,
        "train_history": {"naive": naive_hist, "aware": aware_hist},
        "sweep": {"naive": naive_sweep, "aware": aware_sweep},
        "elapsed_seconds": elapsed,
    }
    save_metrics(metrics, out_dir / "experiment_metrics.json")

    # headline summary at the nominal device sigma
    nominal = cfg.sigma
    print("\n" + "=" * 56)
    print(f"Device preset '{cfg.name}'  (n_levels={cfg.n_levels}, nominal sigma={nominal})")
    print(f"Ideal crossbar : naive={ideal_refs['naive']:.3f}  aware={ideal_refs['aware']:.3f}")
    if nominal in args.sigma_sweep:
        idx = args.sigma_sweep.index(nominal)
        print(f"At sigma={nominal}: naive={naive_sweep['acc_mean'][idx]:.3f}  "
              f"aware={aware_sweep['acc_mean'][idx]:.3f}  "
              f"(+{aware_sweep['acc_mean'][idx] - naive_sweep['acc_mean'][idx]:.3f})")
    print(f"Saved figure + metrics to {out_dir}/   ({elapsed}s)")
    print("=" * 56)


if __name__ == "__main__":
    main()
