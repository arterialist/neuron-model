"""
Direct JSON network builder - builds config dict without Neuron/Synapse objects.

Use this when the goal is only to produce a JSON file. Avoids creating Python
objects (Neuron, PostsynapticPoint, PresynapticPoint) entirely.
"""

import logging
import math
import random
from typing import Any

from snn_classification_realtime.network_builder import (
    _normalize_config,
    get_dataset_dimensions,
)


def _make_neuron_params(num_inputs: int, num_neuromodulators: int = 2) -> dict[str, Any]:
    """Return JSON-serializable neuron params (native Python types only)."""
    return {
        "num_inputs": num_inputs,
        "num_neuromodulators": num_neuromodulators,
        "r_base": random.uniform(0.9, 1.3),
        "b_base": random.uniform(1.1, 1.5),
        "c": 10,
        "lambda": 20.0,
        "p": 1.0,
        "delta_decay": 0.96,
        "beta_avg": 0.999,
        "eta_post": 0.005,
        "eta_retro": 0.002,
        "gamma": [0.99, 0.995],
        "w_r": [-0.2, 0.05],
        "w_b": [-0.2, 0.05],
        "w_tref": [-20.0, 10.0],
    }


def _make_synapse_point(
    neuron_id: int,
    synapse_id: int,
    distance: int | float,
    num_neuromodulators: int = 2,
) -> dict[str, Any]:
    """Return postsynaptic point dict (native Python types)."""
    return {
        "neuron_id": neuron_id,
        "synapse_id": synapse_id,
        "type": "postsynaptic",
        "distance_to_hillock": distance,
        "potential": 0.0,
        "u_i": {
            "info": random.uniform(0.5, 1.5),
            "plast": random.uniform(0.5, 1.5),
            "adapt": [random.uniform(0.1, 0.5) for _ in range(num_neuromodulators)],
        },
    }


def _make_terminal_point(
    neuron_id: int,
    terminal_id: int,
    distance: int | float,
    p: float = 1.0,
    num_neuromodulators: int = 2,
) -> dict[str, Any]:
    """Return presynaptic point dict (native Python types)."""
    return {
        "neuron_id": neuron_id,
        "terminal_id": terminal_id,
        "type": "presynaptic",
        "distance_from_hillock": distance,
        "u_o": {
            "info": p,
            "mod": [random.uniform(0.1, 0.5) for _ in range(num_neuromodulators)],
        },
        "u_i_retro": random.uniform(0.5, 1.5),
    }


def build_network_config_direct(
    config: Any,
    input_shape: tuple[int, int, int] | None = None,
    shortcut_specs: list[tuple[int, int, float]] | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Build network config dict directly (no Neuron objects).

    Returns the same structure as NetworkConfig.save_network_config would produce,
    suitable for json.dump. Use for JSON-only builds (e.g. build_network.py).
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

    neurons_list: list[dict[str, Any]] = []
    synaptic_points_list: list[dict[str, Any]] = []
    connections_list: list[tuple[int, int, int, int]] = []
    external_inputs_set: set[tuple[int, int]] = set()

    # neuron_id -> list of terminal_ids (for connection source)
    neuron_terminals: dict[int, list[int]] = {}
    # neuron_id -> list of synapse_ids (for connection target)
    neuron_synapses: dict[int, list[int]] = {}
    all_layers: list[list[int]] = []
    prev_coord_to_id: dict[tuple[int, ...], int] | None = None
    conv_layer_idx = 0
    connections_set: set[tuple[int, int, int, int]] = set()

    def create_neuron_data(
        layer_idx: int,
        layer_name: str,
        num_synapses: int,
        num_terminals: int = 10,
        metadata_extra: dict | None = None,
        synapse_distance_fn: Any = None,
    ) -> int:
        neuron_id = random.randint(0, 2**36 - 1)
        params = _make_neuron_params(num_synapses)
        metadata = {"layer": layer_idx, "layer_name": layer_name}
        if metadata_extra:
            metadata.update(metadata_extra)

        neurons_list.append({"id": neuron_id, "params": params, "metadata": metadata})

        syn_ids: list[int] = []
        term_ids: list[int] = []
        num_nm = params["num_neuromodulators"]
        p_val = params["p"]

        for s_id in range(num_synapses):
            dist = (
                synapse_distance_fn(s_id)
                if synapse_distance_fn
                else random.randint(2, 8)
            )
            synaptic_points_list.append(
                _make_synapse_point(neuron_id, s_id, dist, num_nm)
            )
            syn_ids.append(s_id)

        for t_id in range(num_terminals):
            dist = random.randint(2, 8)
            synaptic_points_list.append(
                _make_terminal_point(neuron_id, t_id, dist, p_val, num_nm)
            )
            term_ids.append(t_id)

        neuron_terminals[neuron_id] = term_ids
        neuron_synapses[neuron_id] = syn_ids
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
            nid = create_neuron_data(
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

            layer_neurons: list[int] = []
            next_coord_to_id: dict[tuple[int, ...], int] = {}
            rgb_separate_this = li == 0 and is_colored and rgb_separate

            center = (k - 1) / 2.0
            prev_layer_separate = (
                len(next(iter(prev_coord_to_id.keys()))) == 4
                if prev_coord_to_id
                else False
            )
            prev_src_terms_cache = {
                nid: neuron_terminals[nid]
                for nid in set(prev_coord_to_id.values())
            } if prev_coord_to_id else {}

            def conv_distance(s_id: int, kk: int = k) -> float:
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
                                nid = create_neuron_data(
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
                            nid = create_neuron_data(
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
                            dst_syns = neuron_synapses[nid]
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
                                                    len(dst_syns) - 1,
                                                )
                                            conn = (
                                                src_id,
                                                random.choice(src_terms),
                                                nid,
                                                dst_synapse_id,
                                            )
                                            if conn not in connections_set:
                                                connections_set.add(conn)
                                                connections_list.append(conn)
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
                                                    len(dst_syns) - 1,
                                                )
                                            conn = (
                                                src_id,
                                                random.choice(src_terms),
                                                nid,
                                                dst_synapse_id,
                                            )
                                            if conn not in connections_set:
                                                connections_set.add(conn)
                                                connections_list.append(conn)

            all_layers.append(layer_neurons)
            log.info(
                f"Created conv layer {conv_layer_idx}: {len(layer_neurons)} neurons "
                f"(filters={filters}, out_h={out_h}, out_w={out_w})"
            )
            if conv_layer_idx == 0:
                for nid in layer_neurons:
                    for s_id in neuron_synapses[nid]:
                        external_inputs_set.add((nid, s_id))
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
                nid = create_neuron_data(
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
            src_terms_cache = {sid: neuron_terminals[sid] for sid in src_ids}
            dst_syns_cache = {did: neuron_synapses[did] for did in layer_neurons}

            for src_neuron_id in src_ids:
                src_terms = src_terms_cache[src_neuron_id]
                if not src_terms:
                    continue
                mask = [random.random() < p for _ in layer_neurons]
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
                        connections_list.append(conn)
            prev_coord_to_id = None
            prev_channels = None
            prev_h = None
            prev_w = None

    input_layer_ids = all_layers[0] if all_layers else []
    for neuron_id in input_layer_ids:
        for s_id in neuron_synapses[neuron_id]:
            external_inputs_set.add((neuron_id, s_id))

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
                    for c in connections_list
                    if c[0] in all_layers[src_layer_idx]
                    and c[2] in all_layers[immediate_dst]
                ]
                in_conns = (
                    [
                        c
                        for c in connections_list
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
                    connections_list = [
                        c
                        for c in connections_list
                        if c not in out_remove and c not in in_remove
                    ]
                    connections_set -= out_remove
                    connections_set -= in_remove
                    added = 0
                    attempts = 0
                    while added < num_shortcuts and attempts < num_shortcuts * 5:
                        attempts += 1
                        src = random.choice(all_layers[src_layer_idx])
                        dst = random.choice(all_layers[dst_layer_idx])
                        src_terms = neuron_terminals[src]
                        dst_syns = neuron_synapses[dst]
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
                            connections_list.append(conn)
                            added += 1

    log.info(
        f"Network built with {len(neurons_list)} neurons and "
        f"{len(connections_list)} connections"
    )

    # Build final config dict (matches NetworkConfig.save_network_config output)
    config_out = {
        "metadata": {
            "name": "Untitled Network",
            "description": "Network configuration",
            "version": "1.0",
            "created_by": "neuron-model",
        },
        "global_params": _make_neuron_params(10),
        "simulation_params": {
            "max_history": 1000,
            "default_signal_strength": 1.5,
            "default_travel_time_range": [5, 25],
        },
        "neurons": neurons_list,
        "synaptic_points": synaptic_points_list,
        "connections": [
            {
                "source_neuron": c[0],
                "source_terminal": c[1],
                "target_neuron": c[2],
                "target_synapse": c[3],
                "properties": {},
            }
            for c in connections_list
        ],
        "external_inputs": [
            {"target_neuron": nid, "target_synapse": sid, "info": 0.0, "mod": [0.0, 0.0]}
            for nid, sid in external_inputs_set
        ],
    }
    return config_out
