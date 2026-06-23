"""Phase 2: deploy the trained SNN on a simulated memristor crossbar and
measure what the non-idealities cost.

Loads a checkpoint written by ``src.train``, evaluates it three ways and writes
``results/memristor_eval.json``:

1. **software** - the original floating-point network (upper bound);
2. **ideal crossbar** - mapped to ``MemristorLinear`` with every non-ideality
   switched off (sanity check: should match software);
3. **memristor crossbar** - finite conductance levels + programming variability
   + read noise, averaged over several independent "device instantiations" so
   the spread from write variability is visible (mean +/- std).

Example:
    python -m src.eval_memristor --checkpoint results/model.pt \\
        --num-levels 32 --weight-noise 0.05 --read-noise 0.02 --trials 5
"""

from __future__ import annotations

import argparse
import statistics

import torch

from .data import load_dataset
from .memristor import memristorize
from .model import SpikingMLP
from .train import evaluate
from .utils import save_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained SNN on a memristor crossbar.")
    p.add_argument("--checkpoint", default="./results/model.pt")
    p.add_argument("--dataset", default="mnist")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--results-dir", default="./results")
    p.add_argument("--limit-batches", type=int, default=0, help="cap eval batches (0 = all)")
    # device model
    p.add_argument("--g-min", type=float, default=1.0)
    p.add_argument("--g-max", type=float, default=100.0)
    p.add_argument("--num-levels", type=int, default=32, help="programmable conductance states")
    p.add_argument("--weight-noise", type=float, default=0.05, help="write spread (frac of range)")
    p.add_argument("--read-noise", type=float, default=0.02, help="read spread (frac of range)")
    p.add_argument("--trials", type=int, default=5, help="independent device instantiations")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_model(checkpoint: str, device: str) -> SpikingMLP:
    ckpt = torch.load(checkpoint, map_location=device)
    model = SpikingMLP(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model.eval()


def main() -> None:
    args = parse_args()
    device = args.device
    print(f"Device: {device}")

    _, test_loader = load_dataset(
        args.dataset, data_dir=args.data_dir, batch_size=args.batch_size
    )
    model = load_model(args.checkpoint, device)

    cfg = dict(
        g_min=args.g_min,
        g_max=args.g_max,
        num_levels=args.num_levels,
        weight_noise=args.weight_noise,
        read_noise=args.read_noise,
    )

    # 1. software baseline
    sw_acc = evaluate(model, test_loader, device, limit_batches=args.limit_batches)
    print(f"software          : {sw_acc:.4f}")

    # 2. ideal crossbar (control)
    ideal = memristorize(model, ideal=True, **cfg).to(device)
    ideal_acc = evaluate(ideal, test_loader, device, limit_batches=args.limit_batches)
    print(f"ideal crossbar    : {ideal_acc:.4f}")

    # 3. memristor crossbar, averaged over independent device instantiations
    trial_accs = []
    for t in range(args.trials):
        gen = torch.Generator().manual_seed(args.seed + t)
        crossbar = memristorize(model, generator=gen, **cfg).to(device)
        acc = evaluate(crossbar, test_loader, device, limit_batches=args.limit_batches)
        trial_accs.append(acc)
        print(f"  memristor trial {t + 1}/{args.trials}: {acc:.4f}")

    mem_mean = statistics.fmean(trial_accs)
    mem_std = statistics.pstdev(trial_accs) if len(trial_accs) > 1 else 0.0
    print(f"memristor crossbar: {mem_mean:.4f} +/- {mem_std:.4f}  (over {args.trials} devices)")
    print(f"accuracy cost     : {(sw_acc - mem_mean) * 100:.2f} pp")

    results = {
        "checkpoint": args.checkpoint,
        "device_model": cfg,
        "trials": args.trials,
        "software_acc": sw_acc,
        "ideal_crossbar_acc": ideal_acc,
        "memristor_acc_mean": mem_mean,
        "memristor_acc_std": mem_std,
        "memristor_acc_trials": trial_accs,
        "accuracy_cost_pp": (sw_acc - mem_mean) * 100,
    }
    save_metrics(results, f"{args.results_dir}/memristor_eval.json")
    print(f"Saved comparison to {args.results_dir}/memristor_eval.json")


if __name__ == "__main__":
    main()
