#!/usr/bin/env python3
"""
Data transformation layer for converting neural network state to the web viewer
payload consumed by the canvas renderer.
"""

import math
import copy
import traceback
from typing import Dict, List, Any, Tuple, Optional
import numpy as np


class NetworkDataTransformer:
    """Transforms neural network data for the browser visualization."""

    def __init__(self):
        self.node_colors = {
            "firing": "#ff4444",  # Red - firing neurons (output > 0)
            "high_potential": "#ff8800",  # Orange - high potential (> 0.5)
            "some_potential": "#ffdd00",  # Yellow - some potential (> 0)
            "inactive": "#87ceeb",  # Light blue - inactive
            "external": "#90ee90",  # Light green - external inputs
        }

        self.edge_colors = {
            "neuron": "#808080",  # Gray - neuron connections
            "external": "#90ee90",  # Light green - external connections
        }

        self.event_colors = {
            "PresynapticReleaseEvent": "#ff8800",  # Orange
            "RetrogradeSignalEvent": "#8a2be2",  # Purple
            "default": "#0066cc",  # Blue
        }

    def transform_network_state(self, network_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform complete network state to the browser graph payload.

        Args:
            network_state: Output from nn_core.get_network_state()

        Returns:
            Dictionary with graph elements, activity state, and statistics
        """
        if "error" in network_state:
            return {"error": network_state["error"]}

        network = network_state.get("network", {})
        core_state = network_state.get("core_state", {})

        # Transform nodes and edges
        nodes = self._transform_nodes(network)
        edges = self._transform_edges(network, network.get("neurons", {}))

        # Transform traveling signals for animation
        traveling_signals = self._transform_traveling_signals(
            network.get("traveling_signals", []),
            network.get("connections", []),
            core_state.get("current_tick", 0),
        )

        # Calculate network statistics
        stats = self._calculate_statistics(network, core_state, network_state)

        # Calculate layer-based layout positions
        self._assign_layer_positions(nodes)

        result = {
            "elements": {"nodes": nodes, "edges": edges},
            "traveling_signals": traveling_signals,
            "statistics": stats,
            "current_tick": core_state.get("current_tick", 0),
            "is_running": core_state.get("is_running", False),
        }

        # Convert all numpy types to native Python types for JSON serialization
        return self._convert_numpy_types(result)

    def _transform_nodes(self, network: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Transform neurons and external inputs to graph nodes."""
        nodes = []

        # Transform neurons
        neurons = network.get("neurons", {})
        for neuron_id, neuron_data in neurons.items():
            node = self._create_neuron_node(neuron_id, neuron_data)
            nodes.append(node)

        # Transform external inputs as special nodes
        external_inputs = network.get("external_inputs", {})
        for ext_key, ext_data in external_inputs.items():
            parsed = self._parse_external_key(ext_key)
            if parsed is None:
                continue
            neuron_id, synapse_id = parsed
            node = self._create_external_node(neuron_id, synapse_id, ext_data)

            # Assign the same layer as the target neuron when metadata exists.
            target_neuron = self._get_neuron_data(neurons, neuron_id)
            if target_neuron:
                metadata = target_neuron.get("metadata", {}) or {}
                node["data"]["layer"] = metadata.get("layer", -1)
                node["data"]["layer_name"] = metadata.get("layer_name", "external")
                node["data"]["layer_key"] = self._layer_key(metadata.get("layer", -1))

            nodes.append(node)

        return nodes

    def _create_neuron_node(
        self, neuron_id: int, neuron_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a graph node for a neuron."""
        output = self._safe_float(neuron_data.get("output", 0))
        potential = self._safe_float(neuron_data.get("membrane_potential", 0))
        firing_rate = self._safe_float(neuron_data.get("firing_rate", 0))
        t_ref = self._safe_float(
            neuron_data.get(
                "t_ref",
                neuron_data.get("refractory_period", neuron_data.get("tref", 0)),
            )
        )
        threshold = self._safe_float(
            neuron_data.get("r", neuron_data.get("threshold", neuron_data.get("r_base", 0)))
        )
        bias = self._safe_float(
            neuron_data.get("b", neuron_data.get("bias", neuron_data.get("b_base", 0)))
        )
        last_fire_tick = self._safe_float_or_none(
            neuron_data.get("last_fire_tick", neuron_data.get("t_last_fire"))
        )
        m_vector = self._safe_float_list(
            neuron_data.get("M_vector", neuron_data.get("m_vector", []))
        )
        pq_len = self._safe_int(
            neuron_data.get(
                "pq_len",
                neuron_data.get("propagation_queue_length", neuron_data.get("queue_len", 0)),
            )
        )

        # Determine node color based on activity
        if output > 0:
            color = self.node_colors["firing"]
            activity_level = "firing"
        elif potential > 0.5:
            color = self.node_colors["high_potential"]
            activity_level = "high_potential"
        elif potential > 0:
            color = self.node_colors["some_potential"]
            activity_level = "some_potential"
        else:
            color = self.node_colors["inactive"]
            activity_level = "inactive"

        # Calculate node size based on activity (30-50px range)
        base_size = 30
        activity_bonus = min(20, abs(potential) * 40)  # Scale potential to size bonus
        node_size = base_size + activity_bonus

        # Extract layer information from metadata
        layer_info = neuron_data.get("metadata", {}) or {}
        layer_index = layer_info.get("layer", -1)
        layer_name = layer_info.get("layer_name", "unknown")
        
        return {
            "data": {
                "id": f"neuron_{neuron_id}",
                "label": str(neuron_id),
                "type": "neuron",
                "neuron_id": neuron_id,
                "membrane_potential": potential,
                "firing_rate": firing_rate,
                "t_ref": t_ref,
                "output": output,
                "r": threshold,
                "b": bias,
                "t_last_fire": last_fire_tick,
                "M_vector": m_vector,
                "pq_len": pq_len,
                "activity_level": activity_level,
                "synapses": list(neuron_data.get("synapses", []) or []),
                "terminals": list(neuron_data.get("terminals", []) or []),
                "postsynaptic": neuron_data.get("postsynaptic", []),
                "presynaptic": neuron_data.get("presynaptic", []),
                "params": self._normalize_params(
                    neuron_data.get("params", neuron_data.get("parameters", {}))
                ),
                "layer": layer_index,
                "layer_name": layer_name,
                "layer_key": self._layer_key(layer_index),
                # CNN/layout metadata passthrough so frontend can render spatially
                "layer_type": layer_info.get("layer_type"),
                "filter": layer_info.get("filter"),
                "kernel_size": layer_info.get("kernel_size"),
                "stride": layer_info.get("stride"),
                "in_channels": layer_info.get("in_channels"),
                "in_height": layer_info.get("in_height"),
                "in_width": layer_info.get("in_width"),
                "out_height": layer_info.get("out_height"),
                "out_width": layer_info.get("out_width"),
                "x_index": layer_info.get("x", layer_info.get("x_index")),
                "y_index": layer_info.get("y", layer_info.get("y_index")),
                "base_color": color,
                "base_size": node_size,
            }
        }

    def _create_external_node(
        self, neuron_id: int, synapse_id: int, ext_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a graph node for an external input."""
        info_signal = self._safe_float(ext_data.get("info", 0))

        return {
            "data": {
                "id": f"ext_{neuron_id}_{synapse_id}",
                "label": "EXT",
                "type": "external",
                "target_neuron": neuron_id,
                "target_synapse": synapse_id,
                "info_signal": info_signal,
                "layer": -1,  # Will be set to match target neuron's layer
                "layer_name": "external",
                "layer_key": "external",
                "base_color": self.node_colors["external"],
                "base_size": 20,
            }
        }

    def _transform_edges(self, network: Dict[str, Any], neurons: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Transform connections to graph edges."""
        edges = []
        metric_by_edge = self._connection_metric_map(network.get("connection_metrics", []))

        # Transform neuron connections
        connections = network.get("connections", [])
        for i, connection in enumerate(connections):
            if len(connection) >= 4:
                source_neuron, source_terminal, target_neuron, target_synapse = connection[:4]
                metrics = metric_by_edge.get(
                    (str(source_neuron), str(source_terminal), str(target_neuron), str(target_synapse)),
                    {},
                )
                
                # Check if source neuron is firing to determine edge color.
                source_neuron_data = self._get_neuron_data(neurons, source_neuron)
                is_source_firing = self._safe_float(source_neuron_data.get("output", 0)) > 0
                weight = self._safe_float(metrics.get("weight", 0.0))
                base_width = 2 + min(3.5, abs(weight) * 2.5)
                
                # Use light red for firing neurons, default gray for others
                edge_color = "#ff9999" if is_source_firing else self.edge_colors["neuron"]
                
                edge = {
                    "data": {
                        "id": f"conn_{i}",
                        "source": f"neuron_{source_neuron}",
                        "target": f"neuron_{target_neuron}",
                        "type": "neuron",
                        "source_terminal": source_terminal,
                        "target_synapse": target_synapse,
                        "source_firing": is_source_firing,  # Add firing state for styling
                        "weight": weight,
                        "info": self._safe_float(metrics.get("info", 0.0)),
                        "plast": self._safe_float(metrics.get("plast", 0.0)),
                        "potential": self._safe_float(metrics.get("potential", 0.0)),
                        "base_color": edge_color,
                        "base_width": max(3, base_width) if is_source_firing else base_width,
                    }
                }
                edges.append(edge)

        # Transform external input connections
        external_inputs = network.get("external_inputs", {})
        for ext_key in external_inputs.keys():
            parsed = self._parse_external_key(ext_key)
            if parsed is None:
                continue
            neuron_id, synapse_id = parsed
            edge = {
                "data": {
                    "id": f"ext_conn_{neuron_id}_{synapse_id}",
                    "source": f"ext_{neuron_id}_{synapse_id}",
                    "target": f"neuron_{neuron_id}",
                    "type": "external",
                    "base_color": self.edge_colors["external"],
                    "base_width": 2,
                }
            }
            edges.append(edge)

        return edges

    def _connection_metric_map(
        self, metrics: List[Dict[str, Any]]
    ) -> Dict[Tuple[str, str, str, str], Dict[str, Any]]:
        out: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
        for row in metrics or []:
            key = (
                str(row.get("source_neuron")),
                str(row.get("source_terminal")),
                str(row.get("target_neuron")),
                str(row.get("target_synapse")),
            )
            out[key] = row
        return out

    def _transform_traveling_signals(
        self,
        traveling_signals: List[Dict[str, Any]],
        connections: List[Tuple[Any, Any, Any, Any]],
        current_tick: int,
    ) -> List[Dict[str, Any]]:
        """Transform traveling signals for animation."""
        transformed_signals = []

        for signal in traveling_signals:
            source_neuron = signal.get("source_neuron_id", signal.get("source_neuron"))
            source_terminal = signal.get("source_terminal_id")
            target_neuron = signal.get("target_neuron_id", signal.get("target_neuron"))
            event_type = signal.get("event_type", "default")
            arrival_tick = signal.get("arrival_tick", current_tick)
            start_tick = signal.get("start_tick", current_tick)

            if source_neuron is None:
                continue

            target_neurons = [target_neuron] if target_neuron is not None else []
            if not target_neurons and source_terminal is not None:
                for src, src_term, tgt, _tgt_syn in connections:
                    if str(src) == str(source_neuron) and str(src_term) == str(source_terminal):
                        target_neurons.append(tgt)

            if not target_neurons:
                continue

            duration = max(1, int(arrival_tick) - int(start_tick))
            progress = max(0, min(1, (int(current_tick) - int(start_tick)) / duration))

            color = self.event_colors.get(event_type, self.event_colors["default"])
            size = 100 if event_type == "PresynapticReleaseEvent" else 50

            for target in target_neurons:
                transformed_signals.append(
                    {
                        "id": f"signal_{source_neuron}_{target}_{arrival_tick}_{event_type}",
                        "source": f"neuron_{source_neuron}",
                        "target": f"neuron_{target}",
                        "progress": progress,
                        "event_type": event_type,
                        "color": color,
                        "size": size,
                        "arrival_tick": arrival_tick,
                        "start_tick": start_tick,
                    }
                )

        return transformed_signals

    def _calculate_statistics(
        self,
        network: Dict[str, Any],
        core_state: Dict[str, Any],
        root_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate network statistics for display."""
        neurons = network.get("neurons", {})
        connections = network.get("connections", [])
        external_inputs = network.get("external_inputs", {})
        traveling_signals = network.get("traveling_signals", [])

        # Count active neurons
        active_neurons = sum(
            1
            for neuron_data in neurons.values()
            if self._safe_float(neuron_data.get("output", 0)) > 0
        )

        # Calculate average potential and firing rate
        if neurons:
            potentials = [
                self._safe_float(neuron_data.get("membrane_potential", 0))
                for neuron_data in neurons.values()
            ]
            firing_rates = [
                self._safe_float(neuron_data.get("firing_rate", 0))
                for neuron_data in neurons.values()
            ]
            t_refs = [
                self._safe_float(
                    neuron_data.get(
                        "t_ref",
                        neuron_data.get(
                            "refractory_period", neuron_data.get("tref", 0)
                        ),
                    )
                )
                for neuron_data in neurons.values()
            ]
            avg_potential = sum(potentials) / len(potentials)
            avg_firing_rate = sum(firing_rates) / len(firing_rates)
            avg_t_ref = sum(t_refs) / len(t_refs)
            max_potential = max(potentials)
            state_energy = math.sqrt(
                avg_potential * avg_potential
                + avg_firing_rate * avg_firing_rate
                + (active_neurons / len(neurons)) ** 2
            )
        else:
            avg_potential = avg_firing_rate = avg_t_ref = max_potential = state_energy = 0

        # Calculate densities (from network state if available)
        synaptic_density = network.get("synaptic_density", 0.0)
        graph_density = network.get("graph_density", 0.0)

        return {
            "current_tick": core_state.get("current_tick", 0),
            "is_running": core_state.get("is_running", False),
            "tick_rate": core_state.get("tick_rate", 0.0),
            "num_neurons": len(neurons),
            "num_connections": len(connections),
            "num_external_inputs": len(external_inputs),
            "num_traveling_signals": len(traveling_signals),
            "active_neurons": active_neurons,
            "avg_potential": avg_potential,
            "avg_firing_rate": avg_firing_rate,
            "avg_t_ref": avg_t_ref,
            "max_potential": max_potential,
            "free_energy": self._extract_free_energy(root_state, network, core_state),
            "state_energy": state_energy,
            "synaptic_density": synaptic_density,
            "graph_density": graph_density,
        }

    def _assign_layer_positions(self, nodes: List[Dict[str, Any]]) -> None:
        """
        Assign positions to neurons based on their layer information.
        Organizes neurons in layers from left to right, with vertical alignment within each layer.
        """
        # Group neurons by layer
        layers = {}
        for node in nodes:
            if node["data"]["type"] == "neuron":
                layer = node["data"].get("layer", -1)
                if layer not in layers:
                    layers[layer] = []
                layers[layer].append(node)
        
        if not layers:
            return
        
        # Sort layers by index
        sorted_layers = sorted(layers.keys(), key=self._layer_sort_key)
        
        # Layout parameters
        canvas_width = 1200
        canvas_height = 800
        cell_w = 40.0
        cell_h = 40.0
        filter_gap = 40.0
        
        # Calculate required spacing for each layer type to prevent overlap
        layer_widths = []
        layer_meta = {}
        for layer_idx, layer_index in enumerate(sorted_layers):
            layer_neurons = layers[layer_index]
            layer_neurons_only = [n for n in layer_neurons if n["data"]["type"] == "neuron"]
            conv_nodes = [n for n in layer_neurons_only if n["data"].get("layer_type") == "conv"]
            
            if conv_nodes:
                filters = max(self._safe_int(n["data"].get("filter", 0), 0) for n in conv_nodes) + 1
                out_w = max(self._safe_int(n["data"].get("out_width", 1), 1) for n in conv_nodes)
                out_h = max(self._safe_int(n["data"].get("out_height", 1), 1) for n in conv_nodes)
                layer_width = filters * (out_w * cell_w + filter_gap)
                layer_meta[layer_index] = {
                    "type": "conv",
                    "filters": filters,
                    "out_w": out_w,
                    "out_h": out_h,
                }
            else:
                if len(layer_neurons_only) <= 16:
                    layer_width = 60
                else:
                    cols = int(math.ceil(math.sqrt(len(layer_neurons_only))))
                    cell_width = min(80, canvas_width / (len(sorted_layers) + 2))
                    layer_width = cols * cell_width
                layer_meta[layer_index] = {"type": "dense", "count": len(layer_neurons_only)}
            
            layer_widths.append(layer_width)
        
        # Calculate spacing
        min_spacing = max(layer_widths) + 120 if layer_widths else 200
        layer_spacing = min_spacing
        
        # Position neurons in each layer
        for layer_idx, layer_index in enumerate(sorted_layers):
            layer_neurons = layers[layer_index]
            x_pos = (layer_idx + 1) * layer_spacing
            
            # Separate neurons and external inputs for this layer
            layer_neurons_only = [n for n in layer_neurons if n["data"]["type"] == "neuron"]
            layer_externals = [n for n in layer_neurons if n["data"]["type"] == "external"]
            
            # Position external inputs above the layer
            if layer_externals:
                ext_y_offset = 80
                ext_spacing = min(40, layer_spacing / (len(layer_externals) + 1))
                ext_start_x = x_pos - (len(layer_externals) - 1) * ext_spacing / 2
                
                for i, ext_node in enumerate(layer_externals):
                    ext_x = ext_start_x + i * ext_spacing
                    ext_y = ext_y_offset
                    ext_node["position"] = {"x": ext_x, "y": ext_y}
                    ext_node["data"]["position"] = {"x": ext_x, "y": ext_y}
            
            # Conv layout (grid per filter)
            layer_info = layer_meta.get(layer_index, {"type": "dense"})
            if layer_info.get("type") == "conv" and layer_neurons_only:
                filters = layer_info["filters"]
                out_w = layer_info["out_w"]
                out_h = layer_info["out_h"]
                
                # Center the entire block vertically
                block_height = out_h * cell_h
                start_y = (canvas_height - block_height) / 2.0 + (cell_h / 2.0)
                
                filter_block_w = out_w * cell_w + filter_gap
                filter_start = -(filters - 1) * filter_block_w / 2.0
                
                for neuron in layer_neurons_only:
                    d = neuron["data"]
                    f_idx = self._safe_int(d.get("filter", 0), 0)
                    gx = self._safe_int(d.get("x_index", d.get("x", 0)), 0)
                    gy = self._safe_int(d.get("y_index", d.get("y", 0)), 0)
                    
                    base_x = x_pos + filter_start + f_idx * filter_block_w
                    neuron_x = base_x + gx * cell_w
                    neuron_y = start_y + gy * cell_h
                    
                    neuron["position"] = {"x": neuron_x, "y": neuron_y}
                    neuron["data"]["position"] = {"x": neuron_x, "y": neuron_y}
            else:
                # Dense layout (existing behavior)
                if len(layer_neurons_only) <= 16:
                    neuron_spacing = min(60, canvas_height / (len(layer_neurons_only) + 1))
                    start_y = (canvas_height - (len(layer_neurons_only) - 1) * neuron_spacing) / 2
                    
                    for i, neuron in enumerate(layer_neurons_only):
                        y_pos = start_y + i * neuron_spacing
                        neuron["position"] = {"x": x_pos, "y": y_pos}
                        neuron["data"]["position"] = {"x": x_pos, "y": y_pos}
                else:
                    cols = int(math.ceil(math.sqrt(len(layer_neurons_only))))
                    rows = int(math.ceil(len(layer_neurons_only) / cols))
                    
                    cell_width = min(80, layer_spacing / 3)
                    cell_height = min(60, canvas_height / (rows + 1))
                    
                    start_x = x_pos - (cols - 1) * cell_width / 2
                    start_y = (canvas_height - (rows - 1) * cell_height) / 2
                    
                    for i, neuron in enumerate(layer_neurons_only):
                        col = i % cols
                        row = i // cols
                        neuron_x = start_x + col * cell_width
                        neuron_y = start_y + row * cell_height
                        neuron["position"] = {"x": neuron_x, "y": neuron_y}
                        neuron["data"]["position"] = {"x": neuron_x, "y": neuron_y}

        neuron_positions = {
            node["data"].get("neuron_id"): node.get("position")
            for node in nodes
            if node["data"].get("type") == "neuron" and node.get("position")
        }
        for node in nodes:
            if node["data"].get("type") != "external":
                continue
            target = node["data"].get("target_neuron")
            target_pos = neuron_positions.get(target)
            if not target_pos:
                continue
            synapse_id = int(node["data"].get("target_synapse", 0) or 0)
            ext_x = float(target_pos["x"]) - 45.0
            ext_y = float(target_pos["y"]) - 30.0 + (synapse_id % 5) * 12.0
            node["position"] = {"x": ext_x, "y": ext_y}
            node["data"]["position"] = {"x": ext_x, "y": ext_y}

    def get_canvas_style(self) -> Dict[str, Any]:
        """Return style tokens consumed by the canvas app."""
        return {
            "renderer": "canvas",
            "node_colors": self.node_colors,
            "edge_colors": self.edge_colors,
            "event_colors": self.event_colors,
        }

    def update_edge_colors(self, edges: List[Dict[str, Any]], neurons: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Update edge colors based on current neuron firing states."""
        updated_edges = []
        
        for edge in edges:
            try:
                # Create a deep copy to avoid modifying the original
                edge_copy = copy.deepcopy(edge)
                
                if edge_copy["data"]["type"] == "neuron":
                    # Extract source neuron ID from the edge source
                    source_id = edge_copy["data"]["source"]
                    if source_id.startswith("neuron_"):
                        neuron_id = int(source_id[7:])  # Remove "neuron_" prefix
                        neuron_data = self._get_neuron_data(neurons, neuron_id)
                        is_firing = self._safe_float(neuron_data.get("output", 0)) > 0
                        
                        # Update edge data and style
                        edge_copy["data"]["source_firing"] = is_firing
                        if is_firing:
                            edge_copy["data"]["base_color"] = "#ff9999"
                            edge_copy["data"]["base_width"] = 3
                        else:
                            edge_copy["data"]["base_color"] = self.edge_colors["neuron"]
                            edge_copy["data"]["base_width"] = 2
                
                updated_edges.append(edge_copy)
            except Exception as e:
                # If there's an error processing an edge, skip it and continue
                print(f"Warning: Error processing edge {edge.get('data', {}).get('id', 'unknown')}: {e}")
                updated_edges.append(edge)
        
        return updated_edges

    def get_layout_config(self, layout_name: str = "layers") -> Dict[str, Any]:
        """Get layout metadata for the canvas renderer."""
        layouts = {"layers", "grid", "circle", "concentric"}
        name = layout_name if layout_name in layouts else "layers"
        return {"name": name, "available": sorted(layouts)}

    def _get_neuron_data(self, neurons: Dict[Any, Dict[str, Any]], neuron_id: Any) -> Dict[str, Any]:
        """Look up neuron data across int/string key variants."""
        if neuron_id in neurons:
            return neurons[neuron_id] or {}
        text_id = str(neuron_id)
        if text_id in neurons:
            return neurons[text_id] or {}
        try:
            int_id = int(neuron_id)
        except (TypeError, ValueError):
            return {}
        return neurons.get(int_id, {}) or {}

    def _parse_external_key(self, key: Any) -> Optional[Tuple[Any, Any]]:
        """Parse NNCore external-input keys without assuming one exact shape."""
        if isinstance(key, (tuple, list)) and len(key) >= 2:
            return key[0], key[1]
        if isinstance(key, str):
            parts = key.rsplit("_", 1)
            if len(parts) == 2:
                return self._parse_id(parts[0]), self._parse_id(parts[1])
        return None

    def _parse_id(self, value: Any) -> Any:
        try:
            return int(value)
        except (TypeError, ValueError):
            return value

    def _safe_float(self, value: Any, fallback: float = 0.0) -> float:
        try:
            if value is None:
                return fallback
            number = float(value)
            if math.isnan(number) or math.isinf(number):
                return fallback
            return number
        except (TypeError, ValueError):
            return fallback

    def _safe_float_or_none(self, value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            number = float(value)
            if math.isnan(number) or math.isinf(number):
                return None
            return number
        except (TypeError, ValueError):
            return None

    def _safe_float_list(self, value: Any) -> List[float]:
        if value is None:
            return []
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if not isinstance(value, (list, tuple)):
            return []
        return [self._safe_float(item) for item in value]

    def _normalize_params(self, value: Any) -> Dict[str, Any]:
        if not isinstance(value, dict):
            return {}
        params = dict(value)
        if "lambda" in params and "lambda_param" not in params:
            params["lambda_param"] = params["lambda"]
        return params

    def _safe_int(self, value: Any, fallback: int = 0) -> int:
        try:
            if value is None:
                return fallback
            return int(value)
        except (TypeError, ValueError):
            return fallback

    def _extract_free_energy(
        self,
        root_state: Dict[str, Any],
        network: Dict[str, Any],
        core_state: Dict[str, Any],
    ) -> Optional[float]:
        network_stats = root_state.get("network_stats", {}) or {}
        candidates = (
            root_state.get("free_energy"),
            root_state.get("global_free_energy"),
            root_state.get("fe"),
            core_state.get("free_energy"),
            core_state.get("global_free_energy"),
            core_state.get("fe"),
            network.get("free_energy"),
            network.get("global_free_energy"),
            network.get("fe"),
            network_stats.get("free_energy"),
            network_stats.get("global_free_energy"),
            network_stats.get("fe"),
        )
        for value in candidates:
            parsed = self._safe_float_or_none(value)
            if parsed is not None:
                return parsed
        return None

    def _layer_key(self, layer: Any) -> str:
        return "unlayered" if layer is None else str(layer)

    def _layer_sort_key(self, layer: Any) -> Tuple[int, float, str]:
        if layer is None:
            return (1, 0.0, "")
        try:
            return (0, float(layer), str(layer))
        except (TypeError, ValueError):
            return (0, 0.0, str(layer))

    def _convert_numpy_types(self, obj: Any) -> Any:
        """
        Recursively convert numpy data types to native Python types for JSON serialization.

        Args:
            obj: Object to convert

        Returns:
            Object with all numpy types converted to Python types
        """
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (bool, int, float, str)) or obj is None:
            return obj
        else:
            # For any other type, try to convert to string as fallback
            try:
                return str(obj)
            except:
                return None
