#!/usr/bin/env python3
"""
Data transformation layer for converting neural network state to Cytoscape.js format.
Transforms the output from nn_core.get_network_state() into the format expected by Cytoscape.js.
"""

import math
import copy
import traceback
from typing import Dict, List, Any, Tuple, Optional
import numpy as np


class NetworkDataTransformer:
    """Transforms neural network data for Cytoscape.js visualization."""

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
        Transform complete network state to Cytoscape.js format.

        Args:
            network_state: Output from nn_core.get_network_state()

        Returns:
            Dictionary with Cytoscape.js compatible data structure
        """
        if "error" in network_state:
            return {"error": network_state["error"]}

        network = network_state["network"]
        core_state = network_state["core_state"]

        # Transform nodes and edges
        nodes = self._transform_nodes(network)
        edges = self._transform_edges(network, network.get("neurons", {}))

        # Transform traveling signals for animation
        traveling_signals = self._transform_traveling_signals(
            network.get("traveling_signals", []), core_state.get("current_tick", 0)
        )

        # Calculate network statistics
        stats = self._calculate_statistics(network, core_state)

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
        """Transform neurons and external inputs to Cytoscape.js nodes."""
        nodes = []

        # Transform neurons
        neurons = network.get("neurons", {})
        for neuron_id, neuron_data in neurons.items():
            node = self._create_neuron_node(neuron_id, neuron_data)
            nodes.append(node)

        # Transform external inputs as special nodes
        external_inputs = network.get("external_inputs", {})
        for ext_key, ext_data in external_inputs.items():
            if "_" in ext_key:
                parts = ext_key.split("_")
                if len(parts) >= 2:
                    try:
                        neuron_id = int(parts[0])
                        synapse_id = int(parts[1])
                        node = self._create_external_node(
                            neuron_id, synapse_id, ext_data
                        )
                        
                        # Assign the same layer as the target neuron
                        if neuron_id in neurons:
                            target_layer = neurons[neuron_id].get("metadata", {}).get("layer", -1)
                            node["data"]["layer"] = target_layer
                            node["data"]["layer_name"] = neurons[neuron_id].get("metadata", {}).get("layer_name", "external")
                        
                        nodes.append(node)
                    except ValueError:
                        continue

        return nodes

    def _create_neuron_node(
        self, neuron_id: int, neuron_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a Cytoscape.js node for a neuron."""
        output = neuron_data.get("output", 0)
        potential = neuron_data.get("membrane_potential", 0)
        firing_rate = neuron_data.get("firing_rate", 0)

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
        layer_info = neuron_data.get("metadata", {})
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
                "output": output,
                "activity_level": activity_level,
                "synapses": neuron_data.get("synapses", []),
                "terminals": neuron_data.get("terminals", []),
                "layer": layer_index,
                "layer_name": layer_name,
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
                "x_index": layer_info.get("x"),
                "y_index": layer_info.get("y"),
            },
            "style": {
                "background-color": color,
                "width": node_size,
                "height": node_size,
                "label": str(neuron_id),
                "color": "#000000",
                "text-valign": "center",
                "text-halign": "center",
                "font-size": "12px",
                "font-weight": "bold",
            },
        }

    def _create_external_node(
        self, neuron_id: int, synapse_id: int, ext_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a Cytoscape.js node for an external input."""
        info_signal = ext_data.get("info", 0)

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
            },
            "style": {
                "background-color": self.node_colors["external"],
                "width": 20,
                "height": 20,
                "shape": "square",
                "label": "EXT",
                "color": "#006400",
                "text-valign": "center",
                "text-halign": "center",
                "font-size": "8px",
                "font-weight": "bold",
            },
        }

    def _transform_edges(self, network: Dict[str, Any], neurons: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Transform connections to Cytoscape.js edges."""
        edges = []

        # Transform neuron connections
        connections = network.get("connections", [])
        for i, connection in enumerate(connections):
            if len(connection) >= 4:
                source_neuron, source_terminal, target_neuron, target_synapse = (
                    connection[:4]
                )
                
                # Check if source neuron is firing to determine edge color
                source_neuron_data = neurons.get(source_neuron, {})
                is_source_firing = source_neuron_data.get("output", 0) > 0
                
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
                    },
                    "style": {
                        "line-color": edge_color,
                        "target-arrow-color": edge_color,
                        "target-arrow-shape": "triangle",
                        "curve-style": "bezier",
                        "width": 3 if is_source_firing else 2,  # Thicker edges for firing neurons
                        "opacity": 0.9 if is_source_firing else 0.8,  # Higher opacity for firing neurons
                    },
                }
                edges.append(edge)

        # Transform external input connections
        external_inputs = network.get("external_inputs", {})
        for ext_key in external_inputs.keys():
            if "_" in ext_key:
                parts = ext_key.split("_")
                if len(parts) >= 2:
                    try:
                        neuron_id = int(parts[0])
                        synapse_id = int(parts[1])
                        edge = {
                            "data": {
                                "id": f"ext_conn_{ext_key}",
                                "source": f"ext_{neuron_id}_{synapse_id}",
                                "target": f"neuron_{neuron_id}",
                                "type": "external",
                            },
                            "style": {
                                "line-color": self.edge_colors["external"],
                                "target-arrow-color": self.edge_colors["external"],
                                "target-arrow-shape": "triangle",
                                "curve-style": "straight",
                                "width": 2,
                                "opacity": 0.6,
                            },
                        }
                        edges.append(edge)
                    except ValueError:
                        continue

        return edges

    def _transform_traveling_signals(
        self, traveling_signals: List[Dict[str, Any]], current_tick: int
    ) -> List[Dict[str, Any]]:
        """Transform traveling signals for animation."""
        transformed_signals = []

        for signal in traveling_signals:
            source_neuron = signal.get("source_neuron")
            target_neuron = signal.get("target_neuron")
            event_type = signal.get("event_type", "default")
            arrival_tick = signal.get("arrival_tick", current_tick)

            if source_neuron is None or target_neuron is None:
                continue

            # Calculate animation progress (0 to 1)
            # Assuming signals take 5 ticks to travel
            travel_duration = 5
            progress = max(
                0,
                min(
                    1, (current_tick - arrival_tick + travel_duration) / travel_duration
                ),
            )

            color = self.event_colors.get(event_type, self.event_colors["default"])
            size = 100 if event_type == "PresynapticReleaseEvent" else 50

            transformed_signal = {
                "id": f"signal_{source_neuron}_{target_neuron}_{arrival_tick}",
                "source": f"neuron_{source_neuron}",
                "target": f"neuron_{target_neuron}",
                "progress": progress,
                "event_type": event_type,
                "color": color,
                "size": size,
                "arrival_tick": arrival_tick,
            }
            transformed_signals.append(transformed_signal)

        return transformed_signals

    def _calculate_statistics(
        self, network: Dict[str, Any], core_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate network statistics for display."""
        neurons = network.get("neurons", {})
        connections = network.get("connections", [])
        external_inputs = network.get("external_inputs", {})
        traveling_signals = network.get("traveling_signals", [])

        # Count active neurons
        active_neurons = sum(
            1 for neuron_data in neurons.values() if neuron_data.get("output", 0) > 0
        )

        # Calculate average potential and firing rate
        if neurons:
            avg_potential = sum(
                neuron_data.get("membrane_potential", 0)
                for neuron_data in neurons.values()
            ) / len(neurons)
            avg_firing_rate = sum(
                neuron_data.get("firing_rate", 0) for neuron_data in neurons.values()
            ) / len(neurons)
            max_potential = max(
                neuron_data.get("membrane_potential", 0)
                for neuron_data in neurons.values()
            )
        else:
            avg_potential = avg_firing_rate = max_potential = 0

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
            "max_potential": max_potential,
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
        sorted_layers = sorted(layers.keys())
        
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
                filters = max(int(n["data"].get("filter", 0)) for n in conv_nodes) + 1
                out_w = max(int(n["data"].get("out_width", 1) or 1) for n in conv_nodes)
                out_h = max(int(n["data"].get("out_height", 1) or 1) for n in conv_nodes)
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
                    f_idx = int(d.get("filter", 0) or 0)
                    gx = int(d.get("x_index", d.get("x", 0)) or 0)
                    gy = int(d.get("y_index", d.get("y", 0)) or 0)
                    
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

    def get_cytoscape_style(self) -> List[Dict[str, Any]]:
        """Get the complete Cytoscape.js style definition."""
        return [
            # Default node style
            {
                "selector": "node",
                "style": {
                    "content": "data(label)",
                    "text-valign": "center",
                    "text-halign": "center",
                    "font-size": "12px",
                    "font-weight": "bold",
                    "border-width": 2,
                    "border-color": "#333333",
                    "border-opacity": 0.8,
                },
            },
            # Neuron nodes by activity level
            {
                "selector": "node[type='neuron'][activity_level='firing']",
                "style": {
                    "background-color": self.node_colors["firing"],
                    "border-color": "#cc0000",
                    "width": "mapData(output, 0, 2, 30, 50)",
                    "height": "mapData(output, 0, 2, 30, 50)",
                },
            },
            {
                "selector": "node[type='neuron'][activity_level='high_potential']",
                "style": {
                    "background-color": self.node_colors["high_potential"],
                    "border-color": "#cc6600",
                },
            },
            {
                "selector": "node[type='neuron'][activity_level='some_potential']",
                "style": {
                    "background-color": self.node_colors["some_potential"],
                    "border-color": "#ccaa00",
                },
            },
            {
                "selector": "node[type='neuron'][activity_level='inactive']",
                "style": {
                    "background-color": self.node_colors["inactive"],
                    "border-color": "#4682b4",
                },
            },
            # External input nodes
            {
                "selector": "node[type='external']",
                "style": {
                    "background-color": self.node_colors["external"],
                    "border-color": "#006400",
                    "shape": "square",
                    "width": 20,
                    "height": 20,
                    "font-size": "8px",
                },
            },
            # Default edge style
            {
                "selector": "edge",
                "style": {
                    "curve-style": "bezier",
                    "target-arrow-shape": "triangle",
                    "width": 2,
                    "opacity": 0.8,
                },
            },
            # Neuron connection edges
            {
                "selector": "edge[type='neuron']",
                "style": {
                    "line-color": self.edge_colors["neuron"],
                    "target-arrow-color": self.edge_colors["neuron"],
                },
            },
            # Firing neuron edges (light red)
            {
                "selector": "edge[type='neuron'][source_firing='true']",
                "style": {
                    "line-color": "#ff9999",
                    "target-arrow-color": "#ff9999",
                    "width": 3,
                    "opacity": 0.9,
                },
            },
            # External input edges
            {
                "selector": "edge[type='external']",
                "style": {
                    "line-color": self.edge_colors["external"],
                    "target-arrow-color": self.edge_colors["external"],
                    "curve-style": "straight",
                },
            },
            # Selected/highlighted elements
            {
                "selector": "node:selected",
                "style": {
                    "border-width": 4,
                    "border-color": "#ffff00",
                    "border-opacity": 1.0,
                },
            },
            {
                "selector": "edge:selected",
                "style": {
                    "line-color": "#ffff00",
                    "target-arrow-color": "#ffff00",
                    "width": 4,
                    "opacity": 1.0,
                },
            },
            # Hover effects
            {
                "selector": "node:active",
                "style": {
                    "overlay-color": "#ffff00",
                    "overlay-padding": 5,
                    "overlay-opacity": 0.3,
                },
            },
        ]

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
                        neuron_data = neurons.get(str(neuron_id), {})
                        is_firing = neuron_data.get("output", 0) > 0
                        
                        # Update edge data and style
                        edge_copy["data"]["source_firing"] = is_firing
                        if is_firing:
                            edge_copy["style"]["line-color"] = "#ff9999"
                            edge_copy["style"]["target-arrow-color"] = "#ff9999"
                            edge_copy["style"]["width"] = 3
                            edge_copy["style"]["opacity"] = 0.9
                        else:
                            edge_copy["style"]["line-color"] = self.edge_colors["neuron"]
                            edge_copy["style"]["target-arrow-color"] = self.edge_colors["neuron"]
                            edge_copy["style"]["width"] = 2
                            edge_copy["style"]["opacity"] = 0.8
                
                updated_edges.append(edge_copy)
            except Exception as e:
                # If there's an error processing an edge, skip it and continue
                print(f"Warning: Error processing edge {edge.get('data', {}).get('id', 'unknown')}: {e}")
                updated_edges.append(edge)
        
        return updated_edges

    def get_layout_config(self, layout_name: str = "cose") -> Dict[str, Any]:
        """Get layout configuration for Cytoscape.js."""
        layouts = {
            "cose": {
                "name": "cose",
                "idealEdgeLength": 100,
                "nodeOverlap": 20,
                "refresh": 20,
                "fit": True,
                "padding": 30,
                "randomize": False,
                "componentSpacing": 100,
                "nodeRepulsion": 400000,
                "edgeElasticity": 100,
                "nestingFactor": 5,
                "gravity": 80,
                "numIter": 1000,
                "initialTemp": 200,
                "coolingFactor": 0.95,
                "minTemp": 1.0,
            },
            "grid": {
                "name": "grid",
                "fit": True,
                "padding": 30,
                "boundingBox": None,
                "avoidOverlap": True,
                "avoidOverlapPadding": 10,
                "nodeDimensionsIncludeLabels": False,
                "spacingFactor": 1.75,
                "condense": False,
                "rows": None,
                "cols": None,
                "position": None,
                "sort": None,
                "animate": False,
            },
            "circle": {
                "name": "circle",
                "fit": True,
                "padding": 30,
                "boundingBox": None,
                "avoidOverlap": True,
                "nodeDimensionsIncludeLabels": False,
                "spacingFactor": None,
                "radius": None,
                "startAngle": 1.5 * math.pi,
                "sweep": None,
                "clockwise": True,
                "sort": None,
                "animate": False,
            },
            "concentric": {
                "name": "concentric",
                "fit": True,
                "padding": 30,
                "startAngle": 1.5 * math.pi,
                "sweep": None,
                "clockwise": True,
                "equidistant": False,
                "minNodeSpacing": 10,
                "boundingBox": None,
                "avoidOverlap": True,
                "nodeDimensionsIncludeLabels": False,
                "height": None,
                "width": None,
                "spacingFactor": None,
                "concentric": "function(node){ return node.degree(); }",
                "levelWidth": "function(nodes){ return nodes.maxDegree() / 4; }",
                "animate": False,
            },
        }

        return layouts.get(layout_name, layouts["cose"])

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
