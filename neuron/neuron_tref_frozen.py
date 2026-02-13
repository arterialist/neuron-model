"""
Ablation variant: t_ref frozen (no update logic).
The learning window t_ref remains at its initial value and is not updated.
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
    """Neuron with t_ref frozen - no update logic for the learning window."""

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
            ablation="tref_frozen",
            **kwargs,
        )
