"""
Ablation variant: r and b thresholds update disabled.
The excitability thresholds r and b remain at their base values and are not
updated by neuromodulation.
"""

from neuron.neuron import (
    Neuron as BaseNeuron,
    NeuronParameters,
    setup_neuron_logger,
    PostsynapticInputVector,
    PresynapticOutputVector,
    PostsynapticPoint,
    PresynapticPoint,
    NeuronEvent,
    PresynapticReleaseEvent,
    RetrogradeSignalEvent,
)

__all__ = ["Neuron", "NeuronParameters", "setup_neuron_logger"]


class Neuron(BaseNeuron):
    """Neuron with r and b thresholds frozen - no neuromodulatory update."""

    def __init__(
        self,
        neuron_id: int,
        params: NeuronParameters,
        log_level: str = "INFO",
        metadata: dict | None = None,
        **kwargs,
    ):
        super().__init__(
            neuron_id,
            params,
            log_level=log_level,
            metadata=metadata or {},
            ablation="thresholds_frozen",
            **kwargs,
        )
