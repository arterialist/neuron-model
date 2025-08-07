#!/usr/bin/env python3
"""
Data transformation layer for converting neural network state to Cytoscape.js format.
Transforms the output from nn_core.get_network_state() into the format expected by Cytoscape.js.
"""

import math
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
        edges = self._transform_edges(network)

        # Transform traveling signals for animation
        traveling_signals = self._transform_traveling_signals(
            network.get("traveling_signals", []), core_state.get("current_tick", 0)
        )

        # Calculate network statistics
        stats = self._calculate_statistics(network, core_state)

        return {
            "elements": {"nodes": nodes, "edges": edges},
            "traveling_signals": traveling_signals,
            "statistics": stats,
            "current_tick": core_state.get("current_tick", 0),
            "is_running": core_state.get("is_running", False),
        }

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

    def _transform_edges(self, network: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Transform connections to Cytoscape.js edges."""
        edges = []

        # Transform neuron connections
        connections = network.get("connections", [])
        for i, connection in enumerate(connections):
            if len(connection) >= 4:
                source_neuron, source_terminal, target_neuron, target_synapse = (
                    connection[:4]
                )
                edge = {
                    "data": {
                        "id": f"conn_{i}",
                        "source": f"neuron_{source_neuron}",
                        "target": f"neuron_{target_neuron}",
                        "type": "neuron",
                        "source_terminal": source_terminal,
                        "target_synapse": target_synapse,
                    },
                    "style": {
                        "line-color": self.edge_colors["neuron"],
                        "target-arrow-color": self.edge_colors["neuron"],
                        "target-arrow-shape": "triangle",
                        "curve-style": "bezier",
                        "width": 2,
                        "opacity": 0.8,
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
