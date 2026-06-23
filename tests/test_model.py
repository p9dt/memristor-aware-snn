"""Smoke test for the SNN forward pass (shape contract)."""

import torch

from src.model import SpikingMLP


def test_forward_output_shapes():
    num_steps, batch = 5, 3
    model = SpikingMLP(num_inputs=784, num_hidden=16, num_outputs=10, num_steps=num_steps)
    spk_rec, mem_rec = model(torch.randn(batch, 784))
    assert spk_rec.shape == (num_steps, batch, 10)
    assert mem_rec.shape == (num_steps, batch, 10)
