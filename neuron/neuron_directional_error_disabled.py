"""
Ablation variant: Directional error calculation disabled.
Temporal correlation (direction = 1 if delta_t <= t_ref else -1) is disabled;
direction is always 1.0 (treat all inputs as causal).
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
    """Neuron with directional error calculation disabled."""


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
            ablation="directional_error_disabled",
            **kwargs,
        )
