# Roadmap

A build toward a single first-author artifact: a spiking neural network trained
in software and deployed on a simulated memristor crossbar — first on a synthetic
device (v1, done), then with **measured** SnS₂ non-idealities (future scope).

## ✅ v1 — COMPLETE (this repo)

### Phase 1 — SNN baseline
- [x] 2-layer LIF SNN in snnTorch with fast-sigmoid surrogate gradients.
- [x] Rate-coded MNIST, BPTT training, seeded/reproducible runs.
- [x] Metrics + training-curve output.

### Phase 2 — Memristor-aware training (synthetic device)
- [x] Differentiable memristor-crossbar synapse (`src/memristor.py`): differential-pair
      conductance encoding, finite conductance levels (STE), programming/read variability.
- [x] Memristor-*aware* training (non-idealities injected every forward via surrogate-grad/BPTT).
- [x] Naive-vs-aware comparison + accuracy-vs-variability sweep (`src/experiment.py`).

## 🔭 Future scope — device side

Each item has an entry point already in the code, so none requires restructuring.

- [ ] **Measured SnS₂ parameters** via `DeviceConfig.from_measurement` (stub in
      `src/memristor.py`): real conductance-state count, variability, on/off ratio
      from NPL characterization. *The project's true differentiator.*
- [ ] **Richer non-idealities**: conductance drift + LTP/LTD asymmetry in `_conductance`.
- [ ] **N-MNIST / DVS-Gesture** (event-based) via the `load_dataset` hook in `src/data.py`.
- [ ] **Energy estimate** via NeuroSim / DNN+NeuroSim; identify the dominant non-ideality.
- [ ] **Cross-check** against IBM aihwkit inference, or real hardware (SpiNNaker/Loihi).
- [ ] **Writeup**: short workshop paper / arXiv preprint.

## Notes
- Keep every result reproducible (fixed seeds, pinned deps, logged configs).
- Real measured device data is the differentiator — synthetic noise is the fallback only.
  v1 results are explicitly synthetic and labelled as such.
