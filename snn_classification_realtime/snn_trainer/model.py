"""SNN classifier model."""

import torch.nn as nn
import snntorch as snn


HIDDEN_SIZE = 512


class SNNClassifier(nn.Module):
    """Spiking Neural Network classifier for activity time-series."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
    ) -> None:
        super().__init__()
        beta = 0.9

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.lif2 = snn.Leaky(beta=beta)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.lif3 = snn.Leaky(beta=beta)
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.lif4 = snn.Leaky(beta=beta)

    def forward(
        self,
        x: "torch.Tensor",
        mem1: "torch.Tensor | None" = None,
        mem2: "torch.Tensor | None" = None,
        mem3: "torch.Tensor | None" = None,
        mem4: "torch.Tensor | None" = None,
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """Forward pass for a single time step.

        Args:
            x: Input tensor of shape (batch_size, input_size)
            mem1-mem4: Previous membrane potentials (optional)

        Returns:
            spk4: Output spikes for current time step
            mem1-mem4: Updated membrane potentials
        """
        if mem1 is None:
            mem1 = self.lif1.init_leaky()
        if mem2 is None:
            mem2 = self.lif2.init_leaky()
        if mem3 is None:
            mem3 = self.lif3.init_leaky()
        if mem4 is None:
            mem4 = self.lif4.init_leaky()

        cur1 = self.fc1(x)
        spk1, mem1 = self.lif1(cur1, mem1)
        cur2 = self.fc2(spk1)
        spk2, mem2 = self.lif2(cur2, mem2)
        cur3 = self.fc3(spk2)
        spk3, mem3 = self.lif3(cur3, mem3)
        cur4 = self.fc4(spk3)
        spk4, mem4 = self.lif4(cur4, mem4)

        return spk4, mem1, mem2, mem3, mem4
