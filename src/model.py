"""A simple Leaky Integrate-and-Fire (LIF) spiking neural network.

Two fully-connected layers with LIF dynamics, trained end-to-end with
surrogate-gradient backpropagation-through-time. Kept deliberately small and
readable so that Phase 2 can replace the ``nn.Linear`` layers with a simulated
memristor crossbar without restructuring the model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class SpikingMLP(nn.Module):
    """784 -> hidden -> 10 spiking classifier.

    Args:
        num_inputs:  flattened input size (784 for MNIST).
        num_hidden:  hidden layer width.
        num_outputs: number of classes.
        num_steps:   simulation time steps per forward pass.
        beta:        LIF membrane decay (0 < beta < 1).
    """

    def __init__(
        self,
        num_inputs: int = 784,
        num_hidden: int = 256,
        num_outputs: int = 10,
        num_steps: int = 25,
        beta: float = 0.95,
    ) -> None:
        super().__init__()
        self.num_steps = num_steps

        spike_grad = surrogate.fast_sigmoid()  # surrogate gradient for the spike non-linearity

        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the network for ``num_steps`` and return recorded output spikes/membrane.

        Args:
            x: input of shape (batch, num_inputs). The same static input is
               presented at every time step; the LIF layers turn it into spikes
               (rate coding emerges from the membrane dynamics).

        Returns:
            spk_rec: (num_steps, batch, num_outputs) output spike train.
            mem_rec: (num_steps, batch, num_outputs) output membrane potential.
        """
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk_rec, mem_rec = [], []
        for _ in range(self.num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk_rec.append(spk2)
            mem_rec.append(mem2)

        return torch.stack(spk_rec), torch.stack(mem_rec)
