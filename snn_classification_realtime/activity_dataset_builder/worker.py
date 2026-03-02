"""Multiprocessing worker for single-image simulation."""

import time

import numpy as np

from neuron.nn_core import NNCore
from neuron.network_config import NetworkConfig
from neuron.ablation_registry import get_neuron_class_for_ablation
from neuron import setup_neuron_logger

from snn_classification_realtime.activity_dataset_builder.network_utils import (
    infer_layers_from_metadata,
    determine_input_mapping,
)
from snn_classification_realtime.core.config import DatasetConfig
from snn_classification_realtime.core.input_mapping import image_to_signals


def process_single_image_worker(args: tuple) -> tuple:
    """Worker function for multiprocessing image processing.

    Returns: (img_idx, label, u_buf, t_ref_buf, fr_buf, spikes)
    """
    (
        img_idx,
        actual_label,
        ticks_per_image,
        net_path,
        input_layer_ids,
        input_synapses_per_neuron,
        layers,
        tick_ms,
        ds_train,
        dataset_config,
        ablation_name,
        process_id,
        progress_queue,
    ) = args

    neuron_cls = get_neuron_class_for_ablation(ablation_name)
    network_sim = NetworkConfig.load_network_config(net_path, neuron_class=neuron_cls)
    nn_core = NNCore()
    nn_core.neural_net = network_sim
    setup_neuron_logger("CRITICAL")
    nn_core.set_log_level("CRITICAL")

    # Recompute input mapping from this process's network (neuron IDs can differ across processes)
    layers = infer_layers_from_metadata(network_sim)
    input_layer_ids, input_synapses_per_neuron = determine_input_mapping(
        network_sim, layers
    )

    network_sim.reset_simulation()
    nn_core.state.current_tick = 0
    network_sim.current_tick = 0

    image_tensor, _ = ds_train[img_idx]
    signals = image_to_signals(
        image_tensor,
        input_layer_ids,
        input_synapses_per_neuron,
        network_sim,
        dataset_config,
    )

    num_neurons = len(network_sim.network.neurons)
    u_buf = np.zeros((ticks_per_image, num_neurons), dtype=np.float32)
    t_ref_buf = np.zeros((ticks_per_image, num_neurons), dtype=np.float32)
    fr_buf = np.zeros((ticks_per_image, num_neurons), dtype=np.float32)
    spikes: list[tuple[int, int]] = []

    if progress_queue is not None:
        progress_queue.put({
            "process_id": process_id,
            "current_tick": 0,
            "total_ticks": ticks_per_image,
            "img_idx": img_idx,
            "label": actual_label,
            "completed": False,
        })

    uuid_to_idx = {
        uid: i for i, uid in enumerate(network_sim.network.neurons.keys())
    }

    for local_tick in range(ticks_per_image):
        nn_core.send_batch_signals(signals)
        nn_core.do_tick()

        for uid, neuron in network_sim.network.neurons.items():
            idx = uuid_to_idx[uid]
            u_buf[local_tick, idx] = neuron.S
            t_ref_buf[local_tick, idx] = neuron.t_ref
            fr_buf[local_tick, idx] = neuron.F_avg
            if neuron.O > 0:
                spikes.append((local_tick, idx))

        if progress_queue is not None and (local_tick + 1) % 10 == 0:
            progress_queue.put({
                "process_id": process_id,
                "current_tick": local_tick + 1,
                "total_ticks": ticks_per_image,
                "img_idx": img_idx,
                "label": actual_label,
                "completed": False,
            })

        if tick_ms > 0:
            time.sleep(tick_ms / 1000.0)

    if progress_queue is not None:
        progress_queue.put({
            "process_id": process_id,
            "current_tick": ticks_per_image,
            "total_ticks": ticks_per_image,
            "img_idx": img_idx,
            "label": actual_label,
            "completed": True,
        })

    return (img_idx, actual_label, u_buf, t_ref_buf, fr_buf, spikes)
