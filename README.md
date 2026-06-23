# Memristor-Aware Spiking Neural Networks

**Device → Algorithm.** Train a spiking neural network (SNN) with surrogate-gradient
descent in [snnTorch](https://snntorch.readthedocs.io), then deploy it on a *simulated
memristor crossbar* using **measured device non-idealities** — and quantify how much
accuracy real hardware costs you.

> The goal is the rare full-stack neuromorphic story: from a fabricated, characterized
> memristive device (SnS₂ synaptic devices measured at CSIR–NPL) all the way up to a
> trained spiking classifier running on a crossbar model of that same device.

---

## Why this project

Most SNN repos stop at software accuracy. Most memristor papers stop at a single device.
This project connects the two: it asks **"if I take a real, variable, non-ideal device and
build a network out of it, what actually happens to accuracy and energy?"**

That bridge — algorithm side (snnTorch, surrogate gradients, BPTT) meeting the device side
(conductance variability, drift, finite LTP/LTD levels) — is the contribution.

## Project status & roadmap

This repo is built in phases. See [`ROADMAP.md`](ROADMAP.md) for detail.

| Phase | Goal | Status |
|------:|------|--------|
| **1** | Clean, reproducible SNN baseline (MNIST → N-MNIST) | ✅ in this repo |
| **2** | Map trained network onto a memristor crossbar (differential-pair `G+−G−`) and inject device non-idealities; ideal-vs-memristor inference comparison | 🟡 layer + comparison in this repo; **measured** SnS₂ data lands in 2.5 |
| 3 | Accuracy vs. device variability/noise (sweeps); energy estimate (NeuroSim) | 🔜 |
| 4 | Package + short workshop paper / preprint | 🔜 |

## Quickstart

```bash
# 1. (optional) create an environment
python -m venv .venv && . .venv/Scripts/activate     # Windows
# python -m venv .venv && source .venv/bin/activate  # Linux/macOS

# 2. install
pip install -r requirements.txt

# 3. train the Phase-1 baseline on MNIST (writes metrics, curve, and a
#    reloadable checkpoint to results/model.pt)
python -m src.train --epochs 5 --num-steps 25

# 4. Phase 2: deploy that checkpoint on a simulated memristor crossbar and
#    measure the accuracy cost of the device non-idealities
python -m src.eval_memristor --num-levels 32 --weight-noise 0.05 --read-noise 0.02
```

Run `python -m src.train --help` / `python -m src.eval_memristor --help` for all
options (hidden size, LIF decay `beta`, time steps, conductance levels, write/read
noise, number of device trials, …). Pass `--limit-batches N` to either entry point
for a fast smoke run on a subset.

## What's inside

**Phase 1 — SNN baseline**
- A 2-layer **Leaky Integrate-and-Fire** SNN, trained end-to-end with **fast-sigmoid
  surrogate gradients** and backpropagation-through-time.
- Rate-coded MNIST input over a configurable number of time steps.
- Reproducible training loop with seeded runs, accuracy logging, a saved loss/accuracy
  curve, and a reloadable checkpoint.

**Phase 2 — memristor crossbar**
- `MemristorLinear`: an inference-time drop-in for `nn.Linear` that realizes each signed
  weight as a **differential conductance pair** (`G+ − G−`) and injects **finite conductance
  levels**, **programming (write) variability**, and **read noise**.
- `memristorize(model)` swaps every `nn.Linear` in a trained network for a crossbar, so the
  same model can be re-mapped across many independent "device instantiations".
- `src.eval_memristor` compares **software vs. ideal-crossbar vs. memristor-crossbar** accuracy
  (mean ± std over device trials) and reports the accuracy cost in percentage points.

```
memristor-aware-snn/
├── src/
│   ├── model.py          # LIF SNN definition (surrogate gradients)
│   ├── data.py           # dataset loaders (MNIST now; N-MNIST hook for later)
│   ├── train.py          # training / evaluation loop (entry point)
│   ├── memristor.py      # Phase 2: differential-pair crossbar layer + non-idealities
│   ├── eval_memristor.py # Phase 2: ideal-vs-memristor inference comparison (entry point)
│   └── utils.py          # seeding, plotting, metrics I/O
├── tests/                # fast unit tests (no dataset download)
├── .github/workflows/    # CI: unit tests + train/eval smoke
├── results/              # metrics, figures, checkpoint land here
├── requirements.txt
└── ROADMAP.md
```

## Results

Numbers are populated by running the entry points above (no results are checked in).

**Phase 1 — software baseline** (`python -m src.train`)

| Dataset | Model | Test accuracy |
|---------|-------|---------------|
| MNIST   | 784–256–10 LIF SNN, 25 steps | _run `src.train` to populate_ |

**Phase 2 — what the device costs** (`python -m src.eval_memristor`)

| Configuration | Test accuracy |
|---------------|---------------|
| software (float) | _run to populate_ |
| ideal crossbar (control, should match software) | _run to populate_ |
| memristor crossbar (32 levels, 5% write / 2% read noise) | _run to populate_ (mean ± std over device trials) |

The headline number is the **accuracy cost** — software minus memristor accuracy, in
percentage points. Phase 2.5 replaces the synthetic non-ideality knobs with **measured SnS₂
device statistics** from the CSIR–NPL characterization.

## Background

This work builds on measured memristive-device characteristics from my research, including
synaptic plasticity (PPF/PPD, LTP/LTD) of SnS₂-based devices:

- *Structural engineering of SnS₂ nanoflowers for neuromorphic applications*,
  J. Mater. Sci.: Mater. Electron. (2026), DOI: 10.1007/s10854-026-16751-w
- *Resistive Switching & Synapse Properties of Bilayered CuO/MAPbI₃ Films*,
  ACS Appl. Nano Mater. (2025), DOI: 10.1021/acsanm.5c04416

## License

MIT — see [`LICENSE`](LICENSE).
