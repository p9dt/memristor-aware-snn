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
| 2 | Map trained network onto a memristor crossbar (MemTorch / IBM aihwkit); inject measured SnS₂ I–V non-idealities | 🔜 |
| 3 | Accuracy vs. device variability/noise; energy estimate (NeuroSim) | 🔜 |
| 4 | Package + short workshop paper / preprint | 🔜 |

## Quickstart

```bash
# 1. (optional) create an environment
python -m venv .venv && . .venv/Scripts/activate     # Windows
# python -m venv .venv && source .venv/bin/activate  # Linux/macOS

# 2. install
pip install -r requirements.txt

# 3. train the Phase-1 baseline on MNIST
python -m src.train --epochs 5 --num-steps 25

# results (metrics + training curve) are written to results/
```

Run `python -m src.train --help` for all options (hidden size, LIF decay `beta`,
time steps, learning rate, device).

## What's inside (Phase 1)

- A 2-layer **Leaky Integrate-and-Fire** SNN, trained end-to-end with **fast-sigmoid
  surrogate gradients** and backpropagation-through-time.
- Rate-coded MNIST input over a configurable number of time steps.
- Reproducible training loop with seeded runs, accuracy logging, and a saved loss/accuracy curve.

```
memristor-aware-snn/
├── src/
│   ├── model.py      # LIF SNN definition (surrogate gradients)
│   ├── data.py       # dataset loaders (MNIST now; N-MNIST hook for later)
│   ├── train.py      # training / evaluation loop (entry point)
│   └── utils.py      # seeding, plotting, metrics I/O
├── results/          # metrics + figures land here
├── requirements.txt
└── ROADMAP.md
```

## Results

| Dataset | Model | Test accuracy |
|---------|-------|---------------|
| MNIST   | 784–256–10 LIF SNN, 25 steps | _run `src.train` to populate_ |

(Phase 2+ will add columns for ideal-crossbar vs. measured-SnS₂-crossbar accuracy.)

## Background

This work builds on measured memristive-device characteristics from my research, including
synaptic plasticity (PPF/PPD, LTP/LTD) of SnS₂-based devices:

- *Structural engineering of SnS₂ nanoflowers for neuromorphic applications*,
  J. Mater. Sci.: Mater. Electron. (2026), DOI: 10.1007/s10854-026-16751-w
- *Resistive Switching & Synapse Properties of Bilayered CuO/MAPbI₃ Films*,
  ACS Appl. Nano Mater. (2025), DOI: 10.1021/acsanm.5c04416

## License

MIT — see [`LICENSE`](LICENSE).
