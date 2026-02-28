"""
Core network builder for SNN classification.

Builds NeuronNetwork from config: supports conv (receptive-field connections)
and dense (connectivity-based) layers. Used by interactive_training, pipeline,
and the standalone build_network.py entrypoint.
"""

import logging
import math
import random
from typing import Any

import numpy as np

from neuron.network import NeuronNetwork
from neuron.neuron import Neuron, NeuronParameters

# Dataset name -> (channels, height, width). Avoids pipeline dependency when used standalone.
DATASET_DIMENSIONS: dict[str, tuple[int, int, int]] = {
    "mnist": (1, 28, 28),
    "fashionmnist": (1, 28, 28),
    "usps": (1, 28, 28),
    "cifar10": (3, 32, 32),
    "cifar10_grayscale": (1, 32, 32),
    "cifar10_color": (3, 32, 32),
    "cifar100": (3, 32, 32),
    "svhn": (3, 32, 32),
}


def get_dataset_dimensions(dataset: Any) -> tuple[int, int, int]:
    """Get (channels, height, width) for a dataset.

    Args:
        dataset: Dataset name (str) or enum with .value (e.g. DatasetType).

    Returns:
        Tuple of (channels, height, width).
    """
    name = getattr(dataset, "value", dataset)
    if isinstance(name, str) and name in DATASET_DIMENSIONS:
        return DATASET_DIMENSIONS[name]
    raise ValueError(f"Unknown dataset: {dataset}")


def _normalize_config(config: Any) -> dict[str, Any]:
    """Convert config to a plain dict with normalized keys."""
    if hasattr(config, "model_dump"):
        raw = config.model_dump()
    elif hasattr(config, "dict"):
        raw = config.dict()
    else:
        raw = dict(config) if not isinstance(config, dict) else config.copy()

    # Normalize layers: support both pipeline (kernel_size) and interactive (kernel) keys
    layers = raw.get("layers", [])
    out_layers = []
    for i, layer in enumerate(layers):
        if hasattr(layer, "model_dump"):
            L = layer.model_dump()
        elif hasattr(layer, "dict"):
            L = layer.dict()
        else:
            L = dict(layer) if not isinstance(layer, dict) else layer.copy()
        if "kernel_size" in L and "kernel" not in L:
            L["kernel"] = L["kernel_size"]
        if "kernel" in L and "kernel_size" not in L:
            L["kernel_size"] = L["kernel"]
        L.setdefault("kernel_size", 3)
        L.setdefault("kernel", 3)
        L.setdefault("stride", 1)
        L.setdefault("connectivity", 0.8)
        out_layers.append(L)
    raw["layers"] = out_layers
    raw.setdefault("inhibitory_signals", False)
    raw.setdefault("rgb_separate_neurons", False)
    raw.setdefault("input_size", 100)
    return raw


def build_network(
    config: Any,
    input_shape: tuple[int, int, int] | None = None,
    shortcut_specs: list[tuple[int, int, float]] | None = None,
    logger: logging.Logger | None = None,
) -> NeuronNetwork:
    """Build a NeuronNetwork from config.

    Supports conv (receptive-field connections) and dense layers. Conv layers
    must come first; then dense. All-conv, all-dense, or mixed.

    Args:
        config: NetworkBuildConfig (Pydantic), or dict with dataset, layers,
            optional inhibitory_signals, rgb_separate_neurons, input_size.
        input_shape: (C, H, W) for input. If None, derived from config.dataset.
        shortcut_specs: Optional list of (src_layer_idx, dst_layer_idx, percent).
        logger: Optional logger.

    Returns:
        Constructed NeuronNetwork.
    """
    log = logger or logging.getLogger(__name__)
    cfg = _normalize_config(config)

    if input_shape is None:
        input_shape = get_dataset_dimensions(cfg["dataset"])
    prev_channels, prev_h, prev_w = input_shape
    is_colored = cfg["dataset"] in ("cifar10_color",)
    rgb_separate = bool(cfg.get("rgb_separate_neurons"))

    log.info(
        f"Building network for {cfg['dataset']} "
        f"(C={prev_channels}, H={prev_h}, W={prev_w})"
    )

    network_sim = NeuronNetwork(num_neurons=0, synapses_per_neuron=0)
    net_topology = network_sim.network
    all_layers: list[list[int]] = []
    prev_coord_to_id: dict[tuple[int, ...], int] | None = None
    conv_layer_idx = 0
    # O(1) lookup for connection deduplication (avoids O(n) list scans)
    connections_set: set[tuple[int, int, int, int]] = set()

    def create_neuron(
        layer_idx: int,
        layer_name: str,
        num_synapses: int,
        num_terminals: int = 10,
        metadata_extra: dict | None = None,
        synapse_distance_fn: Any = None,
    ) -> int:
        neuron_id = random.randint(0, 2**36 - 1)
        params = NeuronParameters(
            num_inputs=num_synapses,
            num_neuromodulators=2,
            r_base=np.random.uniform(0.9, 1.3),
            b_base=np.random.uniform(1.1, 1.5),
            c=10,
            lambda_param=20.0,
            p=1.0,
            delta_decay=0.96,
            beta_avg=0.999,
            eta_post=0.005,
            eta_retro=0.002,
            gamma=np.array([0.99, 0.995]),
            w_r=np.array([-0.2, 0.05]),
            w_b=np.array([-0.2, 0.05]),
            w_tref=np.array([-20.0, 10.0]),
        )
        metadata = {"layer": layer_idx, "layer_name": layer_name}
        if metadata_extra:
            metadata.update(metadata_extra)
        neuron = Neuron(
            neuron_id,
            params,
            log_level="CRITICAL",
            metadata=metadata,
        )
        for s_id in range(num_synapses):
            distance = (
                synapse_distance_fn(s_id)
                if synapse_distance_fn
                else random.randint(2, 8)
            )
            neuron.add_synapse(s_id, distance_to_hillock=distance)
        for t_id in range(num_terminals):
            neuron.add_axon_terminal(t_id, distance_from_hillock=random.randint(2, 8))
        net_topology.neurons[neuron_id] = neuron
        return neuron_id

    layers_cfg = cfg["layers"]
    if not layers_cfg:
        raise ValueError("config.layers must be non-empty")

    first_type = layers_cfg[0].get("type", "dense")
    if first_type not in ("conv", "dense"):
        first_type = "dense"

    if first_type == "conv":
        pass
    else:
        input_size = int(cfg.get("input_size", 100))
        if is_colored:
            pixels_per_image = prev_h * prev_w
            if rgb_separate:
                synapses_per_input = math.ceil(pixels_per_image / (input_size * 3))
                actual_input_neurons = input_size * 3
            else:
                synapses_per_input = math.ceil(pixels_per_image / input_size) * 3
                actual_input_neurons = input_size
        else:
            vector_size = prev_channels * prev_h * prev_w
            synapses_per_input = math.ceil(vector_size / max(1, input_size))
            actual_input_neurons = input_size

        input_layer: list[int] = []
        for i in range(actual_input_neurons):
            metadata_extra = {}
            if is_colored and rgb_separate:
                metadata_extra = {"color_channel": i % 3, "spatial_idx": i // 3}
            nid = create_neuron(
                0, "input", synapses_per_input, metadata_extra=metadata_extra
            )
            input_layer.append(nid)
        all_layers.append(input_layer)
        log.info(f"Created input layer with {len(input_layer)} neurons")

    for li, layer_cfg in enumerate(layers_cfg):
        ltype = layer_cfg.get("type", "dense")
        if ltype not in ("conv", "dense"):
            ltype = "dense"

        if ltype == "conv":
            if prev_coord_to_id is None and all_layers and first_type == "conv":
                pass
            elif prev_coord_to_id is None and all_layers:
                raise ValueError(
                    "Conv layers must precede dense layers; conv after dense is not supported."
                )

            k = int(layer_cfg.get("kernel_size", layer_cfg.get("kernel", 3)))
            s = int(layer_cfg.get("stride", 1))
            filters = int(layer_cfg.get("filters", 1))
            p = float(layer_cfg.get("connectivity", 0.8))

            if prev_channels is None or prev_h is None or prev_w is None:
                raise ValueError(
                    "Previous spatial dimensions are undefined for convolution."
                )
            prev_channels_int = int(prev_channels)

            out_h = int(math.floor((prev_h - k) / s) + 1)
            out_w = int(math.floor((prev_w - k) / s) + 1)
            if out_h <= 0 or out_w <= 0:
                raise ValueError(
                    f"Conv layer {li} has invalid output dims (k={k}, s={s}) from ({prev_h},{prev_w})."
                )

            layer_neurons = []
            next_coord_to_id: dict[tuple[int, ...], int] = {}
            rgb_multiplier = (
                3 if li == 0 and is_colored and rgb_separate else 1
            )
            rgb_separate_this = li == 0 and is_colored and rgb_separate

            center = (k - 1) / 2.0
            prev_layer_separate = (
                len(next(iter(prev_coord_to_id.keys()))) == 4
                if prev_coord_to_id
                else False
            )
            # Cache presynaptic terminals for O(1) lookup in connection loop
            prev_src_terms_cache = {
                nid: list(net_topology.neurons[nid].presynaptic_points.keys())
                for nid in set(prev_coord_to_id.values())
            } if prev_coord_to_id else {}

            def conv_distance(s_id: int, kk: int = k) -> float:
                c_idx = s_id // (kk * kk)
                rem = s_id % (kk * kk)
                ky = rem // kk
                kx = rem % kk
                dx = (kx - center) * 5.0
                dy = (ky - center) * 5.0
                base = 5.0
                return float(math.sqrt(base * base + dx * dx + dy * dy))

            for f_idx in range(filters):
                for y_out in range(out_h):
                    for x_out in range(out_w):
                        if rgb_separate_this:
                            for color_channel in range(3):
                                num_synapses = max(1, k * k * 1)
                                nid = create_neuron(
                                    conv_layer_idx,
                                    "conv",
                                    num_synapses=num_synapses,
                                    num_terminals=10,
                                    metadata_extra={
                                        "layer_type": "conv",
                                        "filter": f_idx,
                                        "y": y_out,
                                        "x": x_out,
                                        "kernel_size": k,
                                        "stride": s,
                                        "in_channels": 1,
                                        "in_height": prev_h,
                                        "in_width": prev_w,
                                        "out_height": out_h,
                                        "out_width": out_w,
                                        "color_channel": color_channel,
                                        "spatial_idx": f_idx * out_h * out_w
                                        + y_out * out_w
                                        + x_out,
                                    },
                                    synapse_distance_fn=conv_distance,
                                )
                                layer_neurons.append(nid)
                                next_coord_to_id[(f_idx, y_out, x_out, color_channel)] = nid
                        else:
                            num_synapses = max(1, k * k * prev_channels_int)
                            nid = create_neuron(
                                conv_layer_idx,
                                "conv",
                                num_synapses=num_synapses,
                                num_terminals=10,
                                metadata_extra={
                                    "layer_type": "conv",
                                    "filter": f_idx,
                                    "y": y_out,
                                    "x": x_out,
                                    "kernel_size": k,
                                    "stride": s,
                                    "in_channels": prev_channels_int,
                                    "in_height": prev_h,
                                    "in_width": prev_w,
                                    "out_height": out_h,
                                    "out_width": out_w,
                                },
                                synapse_distance_fn=conv_distance,
                            )
                            layer_neurons.append(nid)
                            next_coord_to_id[(f_idx, y_out, x_out)] = nid

                        if prev_coord_to_id is not None:
                            dst_neuron = net_topology.neurons[nid]
                            current_layer_separate = rgb_separate_this

                            if prev_layer_separate:
                                for ky in range(k):
                                    for kx in range(k):
                                        in_y = y_out * s + ky
                                        in_x = x_out * s + kx
                                        if in_y >= prev_h or in_x >= prev_w:
                                            continue
                                        for c in range(3):
                                            if random.random() > p:
                                                continue
                                            src_id = prev_coord_to_id[
                                                (0, int(in_y), int(in_x), int(c))
                                            ]
                                            src_terms = prev_src_terms_cache.get(src_id, [])
                                            if not src_terms:
                                                continue
                                            if current_layer_separate:
                                                dst_synapse_id = c * k * k + ky * k + kx
                                            else:
                                                dst_synapse_id = min(
                                                    (c * k + ky) * k + kx,
                                                    len(dst_neuron.postsynaptic_points) - 1,
                                                )
                                            conn = (
                                                src_id,
                                                random.choice(src_terms),
                                                nid,
                                                dst_synapse_id,
                                            )
                                            if conn not in connections_set:
                                                connections_set.add(conn)
                                                net_topology.connections.append(conn)
                            else:
                                for c in range(prev_channels_int):
                                    for ky in range(k):
                                        for kx in range(k):
                                            in_y = y_out * s + ky
                                            in_x = x_out * s + kx
                                            if in_y >= prev_h or in_x >= prev_w:
                                                continue
                                            if random.random() > p:
                                                continue
                                            src_id = prev_coord_to_id[
                                                (int(c), int(in_y), int(in_x))
                                            ]
                                            src_terms = prev_src_terms_cache.get(src_id, [])
                                            if not src_terms:
                                                continue
                                            if current_layer_separate:
                                                dst_synapse_id = c * k * k + ky * k + kx
                                            else:
                                                dst_synapse_id = min(
                                                    (c * k + ky) * k + kx,
                                                    len(dst_neuron.postsynaptic_points) - 1,
                                                )
                                            conn = (
                                                src_id,
                                                random.choice(src_terms),
                                                nid,
                                                dst_synapse_id,
                                            )
                                            if conn not in connections_set:
                                                connections_set.add(conn)
                                                net_topology.connections.append(conn)

            all_layers.append(layer_neurons)
            log.info(
                f"Created conv layer {conv_layer_idx}: {len(layer_neurons)} neurons "
                f"(filters={filters}, out_h={out_h}, out_w={out_w})"
            )
            if conv_layer_idx == 0:
                for nid in layer_neurons:
                    neuron = net_topology.neurons[nid]
                    for s_id in neuron.postsynaptic_points:
                        net_topology.external_inputs[(nid, s_id)] = {
                            "info": 0.0,
                            "mod": np.array([0.0, 0.0]),
                        }
            prev_coord_to_id = next_coord_to_id
            prev_channels = filters
            prev_h, prev_w = out_h, out_w
            conv_layer_idx += 1

        else:
            size = int(layer_cfg.get("size", 128))
            p = float(layer_cfg.get("connectivity", 0.5))
            layer_name = "output" if li == len(layers_cfg) - 1 else "dense"
            layer_neurons = []

            if prev_coord_to_id is not None:
                prev_layer_ids = list(prev_coord_to_id.values())
            else:
                prev_layer_ids = all_layers[-1] if all_layers else []

            syn_per = layer_cfg.get("synapses_per")
            if syn_per is not None:
                num_synapses = max(1, int(syn_per))
            else:
                num_synapses = max(1, len(prev_layer_ids)) if prev_layer_ids else 10

            for _ in range(size):
                nid = create_neuron(
                    conv_layer_idx + (li - conv_layer_idx),
                    layer_name,
                    num_synapses=num_synapses,
                    num_terminals=10,
                )
                layer_neurons.append(nid)
            all_layers.append(layer_neurons)

            src_ids = (
                list(prev_coord_to_id.values())
                if prev_coord_to_id is not None
                else (all_layers[-2] if len(all_layers) >= 2 else [])
            )
            # Precompute terminal/synapse lists to avoid repeated dict.keys() and list() in inner loop
            src_terms_cache = {
                sid: list(net_topology.neurons[sid].presynaptic_points.keys())
                for sid in src_ids
            }
            dst_syns_cache = {
                did: list(net_topology.neurons[did].postsynaptic_points.keys())
                for did in layer_neurons
            }
            # Batch random draws per source to reduce Python overhead
            for src_neuron_id in src_ids:
                src_terms = src_terms_cache[src_neuron_id]
                if not src_terms:
                    continue
                mask = np.random.random(len(layer_neurons)) < p
                for idx, dst_neuron_id in enumerate(layer_neurons):
                    if not mask[idx]:
                        continue
                    dst_syns = dst_syns_cache[dst_neuron_id]
                    if not dst_syns:
                        continue
                    conn = (
                        src_neuron_id,
                        random.choice(src_terms),
                        dst_neuron_id,
                        random.choice(dst_syns),
                    )
                    if conn not in connections_set:
                        connections_set.add(conn)
                        net_topology.connections.append(conn)
            prev_coord_to_id = None
            prev_channels = None
            prev_h = None
            prev_w = None

    input_layer_ids = all_layers[0] if all_layers else []
    for neuron_id in input_layer_ids:
        neuron = net_topology.neurons[neuron_id]
        for s_id in neuron.postsynaptic_points:
            if (neuron_id, s_id) not in net_topology.external_inputs:
                net_topology.external_inputs[(neuron_id, s_id)] = {
                    "info": 0.0,
                    "mod": np.array([0.0, 0.0]),
                }

    if shortcut_specs:
        for src_layer_idx, dst_layer_idx, percent in shortcut_specs:
            if (
                0 <= src_layer_idx < len(all_layers)
                and 0 <= dst_layer_idx < len(all_layers)
                and src_layer_idx + 1 <= dst_layer_idx
            ):
                immediate_dst = src_layer_idx + 1
                out_conns = [
                    c
                    for c in net_topology.connections
                    if c[0] in all_layers[src_layer_idx]
                    and c[2] in all_layers[immediate_dst]
                ]
                in_conns = (
                    [
                        c
                        for c in net_topology.connections
                        if c[0] in all_layers[dst_layer_idx - 1]
                        and c[2] in all_layers[dst_layer_idx]
                    ]
                    if dst_layer_idx > 0
                    else []
                )
                num_remove_out = int(len(out_conns) * percent)
                num_remove_in = int(len(in_conns) * percent)
                num_shortcuts = min(num_remove_out, num_remove_in)
                if num_shortcuts > 0:
                    out_remove = (
                        set(random.sample(out_conns, num_shortcuts))
                        if len(out_conns) >= num_shortcuts
                        else set(out_conns)
                    )
                    in_remove = (
                        set(random.sample(in_conns, num_shortcuts))
                        if len(in_conns) >= num_shortcuts
                        else set(in_conns)
                    )
                    net_topology.connections = [
                        c
                        for c in net_topology.connections
                        if c not in out_remove and c not in in_remove
                    ]
                    added = 0
                    attempts = 0
                    while added < num_shortcuts and attempts < num_shortcuts * 5:
                        attempts += 1
                        src = random.choice(all_layers[src_layer_idx])
                        dst = random.choice(all_layers[dst_layer_idx])
                        src_terms = list(
                            net_topology.neurons[src].presynaptic_points.keys()
                        )
                        dst_syns = list(
                            net_topology.neurons[dst].postsynaptic_points.keys()
                        )
                        if not src_terms or not dst_syns:
                            continue
                        conn = (
                            src,
                            random.choice(src_terms),
                            dst,
                            random.choice(dst_syns),
                        )
                        if conn not in connections_set:
                            connections_set.add(conn)
                            net_topology.connections.append(conn)
                            added += 1

    log.info(
        f"Network built with {len(net_topology.neurons)} neurons and "
        f"{len(net_topology.connections)} connections"
    )
    return network_sim
