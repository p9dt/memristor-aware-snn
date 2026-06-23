# Roadmap

A phased build toward a single first-author artifact: a spiking neural network trained
in software and deployed on a simulated memristor crossbar with **measured** device
non-idealities.

## Phase 1 — SNN baseline ✅ (this repo)
- [x] 2-layer LIF SNN in snnTorch with fast-sigmoid surrogate gradients.
- [x] Rate-coded MNIST, BPTT training, seeded/reproducible runs.
- [x] Metrics + training-curve output.
- [ ] Swap MNIST → **N-MNIST** (event-based) to move past the field's "hello world".

## Phase 2 — Hardware-aware mapping
- [ ] Map the trained network onto a memristor crossbar with **MemTorch** and/or **IBM aihwkit**.
- [ ] Inject **measured SnS₂ I–V non-idealities**: conductance variability, drift,
      finite/asymmetric LTP–LTD levels (from NPL characterization data).
- [ ] Compare ideal-crossbar vs. measured-device-crossbar inference.

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
