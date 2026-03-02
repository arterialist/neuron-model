"""Signal grid computation and visualization."""

from typing import Any

from neuron.network import NeuronNetwork


def collect_tick_snapshot(
    network_sim: NeuronNetwork,
    layers: list[list[int]],
    tick_index: int,
) -> dict[str, Any]:
    """Collect per-layer arrays of neuron metrics for this tick."""
    net = network_sim.network
    layer_snapshots: list[dict[str, Any]] = []
    for layer_idx, layer_ids in enumerate(layers):
        layer_S: list[float] = []
        layer_F_avg: list[float] = []
        layer_t_ref: list[float] = []
        layer_fire: list[int] = []
        for nid in layer_ids:
            neuron = net.neurons[nid]
            layer_S.append(float(neuron.S))
            layer_F_avg.append(float(neuron.F_avg))
            layer_t_ref.append(float(neuron.t_ref))
            layer_fire.append(1 if neuron.O > 0 else 0)
        layer_snapshots.append({
            "layer_index": layer_idx,
            "neuron_ids": layer_ids,
            "S": layer_S,
            "F_avg": layer_F_avg,
            "t_ref": layer_t_ref,
            "fired": layer_fire,
        })
    return {"tick": tick_index, "layers": layer_snapshots}
