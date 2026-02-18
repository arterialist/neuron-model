"""Map images to neural network input signals."""

from activity_dataset_builder.input_mapping import image_to_signals as _image_to_signals
from activity_dataset_builder.config import DatasetConfig
from neuron.network import NeuronNetwork
import torch


def image_to_signals(
    image_tensor: torch.Tensor,
    network_sim: NeuronNetwork,
    input_layer_ids: list[int],
    synapses_per_neuron: int,
    dataset_config: DatasetConfig,
) -> list[tuple[int, int, float]]:
    """Map an image tensor to (neuron_id, synapse_id, strength) signals."""
    return _image_to_signals(
        image_tensor,
        input_layer_ids,
        synapses_per_neuron,
        network_sim,
        dataset_config,
    )
