"""Unit tests for the Phase-2 memristor crossbar layer.

These use small random tensors only - no dataset download, no training - so
they stay fast and deterministic in CI.
"""

import torch
import torch.nn as nn

from src.memristor import MemristorLinear, memristorize


def test_ideal_matches_linear_exactly():
    lin = nn.Linear(8, 4)
    mem = MemristorLinear.from_linear(lin, ideal=True)
    x = torch.randn(3, 8)
    assert torch.allclose(mem(x), lin(x), atol=1e-6)


def test_noise_free_continuous_reconstructs_weight():
    lin = nn.Linear(8, 4)
    mem = MemristorLinear.from_linear(lin, num_levels=0, weight_noise=0.0, read_noise=0.0)
    assert torch.allclose(mem.effective_weight(), lin.weight, atol=1e-5)


def test_quantization_limits_conductance_states():
    lin = nn.Linear(16, 8)
    mem = MemristorLinear.from_linear(lin, num_levels=8, weight_noise=0.0, read_noise=0.0)
    # finite LTP/LTD levels: each device takes at most num_levels distinct states
    assert torch.unique(mem.Gp).numel() <= 8
    assert torch.unique(mem.Gn).numel() <= 8


def test_conductances_stay_within_range():
    lin = nn.Linear(16, 8)
    mem = MemristorLinear.from_linear(
        lin, g_min=1.0, g_max=100.0, num_levels=16, weight_noise=0.1
    )
    for g in (mem.Gp, mem.Gn):
        assert g.min().item() >= 1.0 - 1e-6
        assert g.max().item() <= 100.0 + 1e-6


def test_forward_shape_and_runs_with_noise():
    lin = nn.Linear(8, 4)
    mem = MemristorLinear.from_linear(lin, num_levels=16, weight_noise=0.05, read_noise=0.02)
    out = mem(torch.randn(3, 8))
    assert out.shape == (3, 4)


def test_programming_is_reproducible_with_generator():
    lin = nn.Linear(8, 4)
    a = MemristorLinear.from_linear(lin, weight_noise=0.1, generator=torch.Generator().manual_seed(0))
    b = MemristorLinear.from_linear(lin, weight_noise=0.1, generator=torch.Generator().manual_seed(0))
    assert torch.allclose(a.Gp, b.Gp)


def test_memristorize_replaces_linears_and_leaves_original():
    model = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 2))
    mapped = memristorize(model, num_levels=8)
    assert isinstance(mapped[0], MemristorLinear)
    assert isinstance(mapped[2], MemristorLinear)
    # original model must be untouched (deep copy)
    assert type(model[0]) is nn.Linear
