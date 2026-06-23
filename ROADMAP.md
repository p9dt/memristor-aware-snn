# Roadmap

A phased build toward a single first-author artifact: a spiking neural network trained
in software and deployed on a simulated memristor crossbar with **measured** device
non-idealities.

## Phase 1 — SNN baseline ✅ (this repo)
- [x] 2-layer LIF SNN in snnTorch with fast-sigmoid surrogate gradients.
- [x] Rate-coded MNIST, BPTT training, seeded/reproducible runs.
- [x] Metrics + training-curve output.
- [ ] Swap MNIST → **N-MNIST** (event-based) to move past the field's "hello world".

## Phase 2 — Hardware-aware mapping 🟡 (layer + comparison in this repo)
- [x] Map the trained network onto a differential-pair (`G+−G−`) memristor crossbar
      (`src/memristor.py`, `memristorize()`) — a self-contained model so the accuracy
      drop is attributable to a specific non-ideality.
- [x] Inject **synthetic** device non-idealities: finite/quantized LTP–LTD conductance
      levels, programming (write) variability, and read noise.
- [x] Compare software vs. ideal-crossbar vs. memristor-crossbar inference, averaged over
      independent device instantiations (`src/eval_memristor.py`).
- [ ] **Phase 2.5:** swap the synthetic knobs for **measured SnS₂ I–V statistics**
      (conductance variability, drift, asymmetric LTP/LTD) from NPL characterization data.
- [ ] Cross-check against an established library (**MemTorch** / **IBM aihwkit**).

## Phase 3 — Analysis
- [ ] Accuracy vs. device variability and read/write noise (sweep).
- [ ] Energy estimate via **NeuroSim / DNN+NeuroSim**.
- [ ] Identify which non-ideality dominates the accuracy drop.

## Phase 4 — Package
- [ ] Clean repo, reproducible scripts, figures.
- [ ] 4-page workshop paper / arXiv preprint.

## Notes
- Keep every result reproducible (fixed seeds, pinned deps, logged configs).
- Real measured device data is the differentiator — synthetic noise is the fallback only.
