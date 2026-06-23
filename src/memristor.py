"""Phase 2: a memristor-crossbar-aware linear layer.

This is the piece that makes the repo live up to its name. It replaces a
trained ``nn.Linear`` with a **differential-pair memristor crossbar** (each
signed weight is realized as ``G+ - G-`` of two conductances) and injects the
device non-idealities that actually move the needle on inference accuracy:

* **finite conductance levels** - a real analog synapse programs to a limited
  number of distinguishable LTP/LTD states, not a continuum;
* **programming (write) variability** - the conductance you ask for is not the
  conductance you get (device-to-device and cycle-to-cycle spread, fixed once
  the array is written);
* **read noise** - the conductance fluctuates slightly on every read, so it is
  re-sampled on every forward pass.

The mapping is deliberately simple and inspectable so an accuracy drop can be
attributed to a *specific* non-ideality. Conductance values are normalized:
only the ratios (number of levels, noise as a fraction of the usable range)
affect the result, so absolute units (uS) are deferred to the measured-SnS2
device-data integration in Phase 2.5.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def _quantize(g: torch.Tensor, g_min: float, g_max: float, num_levels: int) -> torch.Tensor:
    """Snap conductances to ``num_levels`` evenly spaced states in [g_min, g_max]."""
    if not num_levels or num_levels <= 1:
        return g
    step = (g_max - g_min) / (num_levels - 1)
    return g_min + torch.round((g - g_min) / step) * step


class MemristorLinear(nn.Module):
    """Inference-only drop-in for ``nn.Linear`` backed by a memristor crossbar.

    The trained weight is frozen and mapped to a fixed pair of programmed
    conductances ``(Gp, Gn)`` once, at construction time (this captures write
    quantization + programming variability). Read noise, if any, is added fresh
    on every ``forward`` call.

    Args:
        weight:       trained weight, shape (out_features, in_features).
        bias:         trained bias, shape (out_features,), or ``None``.
        g_min, g_max: normalized conductance range of the device.
        num_levels:   number of programmable conductance states (LTP/LTD levels).
                      ``0`` or ``1`` disables quantization (continuous).
        weight_noise: programming spread, as a fraction of (g_max - g_min).
        read_noise:   per-read spread, as a fraction of (g_max - g_min).
        ideal:        if ``True``, bypass every non-ideality and behave exactly
                      like the original ``nn.Linear`` (useful as a control).
        generator:    optional ``torch.Generator`` for reproducible programming.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        *,
        g_min: float = 1.0,
        g_max: float = 100.0,
        num_levels: int = 32,
        weight_noise: float = 0.05,
        read_noise: float = 0.0,
        ideal: bool = False,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        self.g_min = float(g_min)
        self.g_max = float(g_max)
        self.read_noise = float(read_noise)
        self.ideal = bool(ideal)

        weight = weight.detach().clone()
        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias.detach().clone() if bias is not None else None)

        # Symmetric weight->conductance scale: the largest-magnitude weight uses
        # the full usable conductance swing of one device in the differential pair.
        g_range = self.g_max - self.g_min
        w_abs_max = weight.abs().max().item()
        self.scale = (g_range / w_abs_max) if w_abs_max > 0 else 1.0

        # Differential mapping: positive weights raise G+, negative raise G-.
        gp = self.g_min + torch.relu(weight) * self.scale
        gn = self.g_min + torch.relu(-weight) * self.scale

        # Write-time non-idealities, applied once (the array is now "programmed").
        gp = _quantize(gp, self.g_min, self.g_max, num_levels)
        gn = _quantize(gn, self.g_min, self.g_max, num_levels)
        if weight_noise > 0:
            std = weight_noise * g_range
            gp = gp + torch.randn(gp.shape, generator=generator) * std
            gn = gn + torch.randn(gn.shape, generator=generator) * std
        gp = gp.clamp(self.g_min, self.g_max)
        gn = gn.clamp(self.g_min, self.g_max)

        self.register_buffer("Gp", gp)
        self.register_buffer("Gn", gn)

    @classmethod
    def from_linear(cls, linear: nn.Linear, **kwargs) -> "MemristorLinear":
        """Build a crossbar layer from an existing trained ``nn.Linear``."""
        return cls(linear.weight, linear.bias, **kwargs)

    def effective_weight(self, *, with_read_noise: bool = False) -> torch.Tensor:
        """Reconstruct the weight the crossbar actually implements."""
        gp, gn = self.Gp, self.Gn
        if with_read_noise and self.read_noise > 0:
            std = self.read_noise * (self.g_max - self.g_min)
            gp = (gp + torch.randn_like(gp) * std).clamp(self.g_min, self.g_max)
            gn = (gn + torch.randn_like(gn) * std).clamp(self.g_min, self.g_max)
        return (gp - gn) / self.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ideal:
            return F.linear(x, self.weight, self.bias)
        w_eff = self.effective_weight(with_read_noise=True)
        return F.linear(x, w_eff, self.bias)

    def extra_repr(self) -> str:
        out, inp = self.weight.shape
        return (
            f"in_features={inp}, out_features={out}, g=[{self.g_min},{self.g_max}], "
            f"read_noise={self.read_noise}, ideal={self.ideal}"
        )


def memristorize(model: nn.Module, **cfg) -> nn.Module:
    """Return a deep copy of ``model`` with every ``nn.Linear`` swapped for a
    :class:`MemristorLinear` crossbar.

    The copy is returned in eval mode; the original model is left untouched, so
    one trained network can be re-mapped many times (e.g. to sample programming
    variability across independent "device instantiations").
    """
    mapped = copy.deepcopy(model)
    for module in mapped.modules():
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                setattr(module, name, MemristorLinear.from_linear(child, **cfg))
    return mapped.eval()
