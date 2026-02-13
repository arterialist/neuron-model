"""
Ablation variant: Weight update disabled.
Postsynaptic synaptic weights (u_i.info) are not updated by plasticity.
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
    """Neuron with weight update disabled."""

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
            ablation="weight_update_disabled",
            **kwargs,
        )
