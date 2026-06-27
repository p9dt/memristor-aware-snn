"""Differentiable memristor-crossbar device model (Phase 2).

This module implements *memristor-aware training*: a drop-in replacement for
``nn.Linear`` whose weights are passed through a simulated memristor crossbar on
**every forward pass**, so the network learns weights that are robust to real
device non-idealities instead of discovering them only after deployment.

Why not MemTorch directly?
--------------------------
MemTorch's ``convert()`` path is built for *inference* and is not differentiable,
so it cannot sit inside a surrogate-gradient / BPTT training loop. For *training*
the standard approach (and the one here) is a lightweight, differentiable
non-ideality injection with straight-through gradients. The device parameters
below are the same quantities MemTorch / IBM aihwkit expose, so measured SnS2
characterization data (Phase 2 of the roadmap) drops straight into ``DeviceConfig``.

Device model
------------
A signed weight cannot be stored on a single memristor (conductance ``G >= 0``),
so each weight is encoded as a **differential pair** ``w ~ (G+ - G-)``. We work in
a normalized conductance ``g = w / w_max`` in ``[-1, 1]`` (``w_max`` = the layer's
programmed full-scale range) and apply, in order:

1. **Finite conductance levels** -- real devices have a limited number of stable
   states (``n_levels``). Quantized with a straight-through estimator (STE).
2. **Programming variability** -- write/cycle-to-cycle conductance spread
   (``sigma`` as a fraction of full scale), re-sampled every forward so the
   network sees many device realizations during training.
3. **Read noise** -- optional per-read jitter (``sigma_read``).

Drift and LTP/LTD asymmetry are intentionally left as Phase-3 hooks.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

_EPS = 1e-12


@dataclass
class DeviceConfig:
    """Memristor crossbar device parameters.

    Args:
        n_levels:   number of stable conductance states per device (finite
                    LTP/LTD resolution). ``None`` or <= 1 disables quantization.
        sigma:      programming / cycle-to-cycle conductance variability, as a
                    fraction of the full-scale conductance range. Re-sampled every
                    forward pass (the core "awareness" knob).
        sigma_read: per-read conductance noise, fraction of full scale.
        name:       label for logging / plots.
    """

    n_levels: int | None = 16
    sigma: float = 0.05
    sigma_read: float = 0.0
    name: str = "device"


# Convenience presets. IMPORTANT: these are *synthetic* parameter sets chosen to be
# plausible for a generic memristor -- they are NOT fitted to any measured device.
# Real SnS2 characterization data (conductance levels, variability, LTP/LTD) is a
# Phase-3 deliverable; when available, add it via ``DeviceConfig.from_measurement``
# below rather than inventing numbers here.
PRESETS = {
    "ideal": DeviceConfig(n_levels=None, sigma=0.0, sigma_read=0.0, name="ideal"),
    "mild": DeviceConfig(n_levels=32, sigma=0.03, sigma_read=0.0, name="synthetic-mild"),
    "synthetic": DeviceConfig(n_levels=16, sigma=0.05, sigma_read=0.01, name="synthetic"),
    "harsh": DeviceConfig(n_levels=8, sigma=0.12, sigma_read=0.02, name="synthetic-harsh"),
}


def from_measurement(*_args, **_kwargs) -> DeviceConfig:
    """Build a DeviceConfig from measured device data (e.g. SnS2 NPL characterization).

    Placeholder for Phase 3: real conductance-state count, programming variability,
    and LTP/LTD asymmetry extracted from I-V / pulse measurements go here. Until that
    data is wired in, the project uses the *synthetic* presets above and says so.
    """
    raise NotImplementedError(
        "No measured device data is wired in yet. Use a synthetic preset "
        f"({', '.join(sorted(PRESETS))}) and report results as synthetic."
    )


def quantize_ste(g: torch.Tensor, n_levels: int | None) -> torch.Tensor:
    """Uniformly quantize ``g`` in [-1, 1] to ``n_levels`` states (STE backward).

    Forward rounds to the nearest conductance level; backward passes the gradient
    through unchanged (straight-through estimator), exactly like the surrogate
    gradient used at the spike.
    """
    if n_levels is None or n_levels <= 1:
        return g
    step = 2.0 / (n_levels - 1)
    gq = torch.round((g + 1.0) / step) * step - 1.0
    gq = gq.clamp(-1.0, 1.0)
    return g + (gq - g).detach()  # value of gq, gradient of g


class MemristorLinear(nn.Module):
    """``nn.Linear`` whose weights pass through a memristor crossbar each forward.

    Drop-in for ``nn.Linear`` (same ``weight``/``bias`` parameters, so a trained
    checkpoint or an ideal baseline maps over directly). When ``ideal`` is True the
    layer behaves exactly like ``nn.Linear``; otherwise the device non-idealities
    in ``DeviceConfig`` are injected.

    The bias is treated as an ideal peripheral (kept digital), which matches how
    crossbar accelerators usually handle biases.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        cfg: DeviceConfig | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Same parameterization as nn.Linear for easy interop.
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

        cfg = cfg or DeviceConfig()
        # Stored as plain attributes (not a frozen dataclass) so eval-time sweeps
        # can mutate them in place via the setters on the model.
        self.n_levels = cfg.n_levels
        self.sigma = cfg.sigma
        self.sigma_read = cfg.sigma_read
        self.device_name = cfg.name
        self.ideal = False

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1.0 / fan_in ** 0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def _conductance(self, w: torch.Tensor) -> torch.Tensor:
        """Map weights -> noisy, quantized normalized conductance -> effective weights."""
        # Programmed full-scale range for this crossbar tile (detached: it sets the
        # mapping, it is not itself a trained quantity).
        w_max = w.abs().amax().clamp(min=_EPS).detach()
        g = w / w_max  # differential-pair normalized conductance in [-1, 1]

        g = quantize_ste(g, self.n_levels)  # finite conductance levels

        if self.sigma > 0 and (self.training or not self.training):
            # programming variability, re-sampled every forward (device-to-device
            # + cycle-to-cycle spread). Kept active at eval too so accuracy sweeps
            # are Monte-Carlo estimates over device realizations.
            g = g + self.sigma * torch.randn_like(g)
        if self.sigma_read > 0:
            g = g + self.sigma_read * torch.randn_like(g)

        g = g.clamp(-1.0, 1.0)  # device cannot exceed G_on / fall below -G_on
        return g * w_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ideal:
            return F.linear(x, self.weight, self.bias)
        w_eff = self._conductance(self.weight)
        return F.linear(x, w_eff, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"device={self.device_name}, n_levels={self.n_levels}, "
            f"sigma={self.sigma}, sigma_read={self.sigma_read}, ideal={self.ideal}"
        )
