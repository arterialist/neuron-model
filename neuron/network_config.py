#!/usr/bin/env python3
"""
Network Configuration Module
Handles loading and saving multi-neuron networks from/to JSON configuration files.
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional, Union
from pathlib import Path


# Handle imports for both direct execution and module execution
try:
    from .neuron import Neuron, NeuronParameters
    from .network import NetworkTopology, NeuronNetwork
except ImportError:
    from neuron import Neuron, NeuronParameters
    from network import NetworkTopology, NeuronNetwork


class NetworkConfig:
    """Handles network configuration via JSON files."""

    DEFAULT_GLOBAL_PARAMS = NeuronParameters()

    DEFAULT_SIMULATION_PARAMS = {
        "max_history": 1000,
        "default_signal_strength": 1.5,
        "default_travel_time_range": [5, 25],
    }

    @staticmethod
    def create_empty_config() -> Dict[str, Any]:
        """Create an empty network configuration template."""
        return {
            "metadata": {
                "name": "Untitled Network",
                "description": "Network configuration",
                "version": "1.0",
                "created_by": "neuron-model",
            },
            "global_params": NetworkConfig._serialize_neuron_params(
                NetworkConfig.DEFAULT_GLOBAL_PARAMS
            ),
            "simulation_params": NetworkConfig.DEFAULT_SIMULATION_PARAMS.copy(),
            "neurons": [],
            "synaptic_points": [],
            "connections": [],
            "external_inputs": [],
        }

    @staticmethod
    def save_network_config(
        network_sim: NeuronNetwork,
        filepath: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save a NeuronNetwork to a JSON configuration file."""
        filepath = Path(filepath)

        # Extract network configuration
        config = NetworkConfig.create_empty_config()

        # Update metadata if provided
        if metadata:
            config["metadata"].update(metadata)

        # Extract neurons
        neurons_list = []
        for neuron_id, neuron in network_sim.network.neurons.items():
            neuron_config = {
                "id": neuron_id,
                "params": NetworkConfig._serialize_neuron_params(neuron.params),
                "metadata": neuron.metadata if hasattr(neuron, "metadata") else {},
            }
            neurons_list.append(neuron_config)

        config["neurons"] = neurons_list

        # Extract detailed synaptic points data
        synaptic_points_list = []
        for neuron_id, neuron in network_sim.network.neurons.items():
            # Export postsynaptic points (synapses)
            for synapse_id, postsynaptic_point in neuron.postsynaptic_points.items():
                distance = neuron.distances.get(synapse_id, 0)
                synapse_data = {
                    "neuron_id": neuron_id,
                    "synapse_id": synapse_id,
                    "type": "postsynaptic",
                    "distance_to_hillock": distance,
                    "potential": float(postsynaptic_point.potential),
                    "u_i": {
                        "info": float(postsynaptic_point.u_i.info),
                        "plast": float(postsynaptic_point.u_i.plast),
                        "adapt": (
                            postsynaptic_point.u_i.adapt.tolist()
                            if isinstance(postsynaptic_point.u_i.adapt, np.ndarray)
                            else postsynaptic_point.u_i.adapt
                        ),
                    },
                }
                synaptic_points_list.append(synapse_data)

            # Export presynaptic points (axon terminals)
            for terminal_id, presynaptic_point in neuron.presynaptic_points.items():
                distance = neuron.distances.get(terminal_id, 0)
                terminal_data = {
                    "neuron_id": neuron_id,
                    "terminal_id": terminal_id,
                    "type": "presynaptic",
                    "distance_from_hillock": distance,
                    "u_o": {
                        "info": float(presynaptic_point.u_o.info),
                        "mod": (
                            presynaptic_point.u_o.mod.tolist()
                            if isinstance(presynaptic_point.u_o.mod, np.ndarray)
                            else presynaptic_point.u_o.mod
                        ),
                    },
                    "u_i_retro": float(presynaptic_point.u_i_retro),
                }
                synaptic_points_list.append(terminal_data)

        config["synaptic_points"] = synaptic_points_list

        # Extract connections
        connections_list = []
        for (
            source_neuron_id,
            source_terminal_id,
            target_neuron_id,
            target_synapse_id,
        ) in network_sim.network.connections:
            connection = {
                "source_neuron": source_neuron_id,
                "source_terminal": source_terminal_id,
                "target_neuron": target_neuron_id,
                "target_synapse": target_synapse_id,
                "properties": {},  # Can be extended for connection-specific properties
            }
            connections_list.append(connection)

        config["connections"] = connections_list

        # Extract external inputs
        external_inputs_list = []
        for input_key, input_data in network_sim.network.external_inputs.items():
            neuron_id, synapse_id = input_key
            external_input = {
                "target_neuron": neuron_id,
                "target_synapse": synapse_id,
                "info": float(input_data["info"]),
                "mod": (
                    input_data["mod"].tolist()
                    if isinstance(input_data["mod"], np.ndarray)
                    else input_data["mod"]
                ),
            }
            external_inputs_list.append(external_input)

        config["external_inputs"] = external_inputs_list

        # Save to file
        with open(filepath, "w") as f:
            json.dump(config, f, indent=2)

        # print(f"Network configuration saved to {filepath}")

    @staticmethod
    def load_network_config(
        filepath: Union[str, Path],
        neuron_class: Optional[type] = None,
    ) -> NeuronNetwork:
        """Load a NeuronNetwork from a JSON configuration file.

        Args:
            filepath: Path to the JSON configuration file.
            neuron_class: Optional Neuron class to use (e.g. from ablation variant).
                If None, uses the default neuron.neuron.Neuron.
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")

        with open(filepath, "r") as f:
            config = json.load(f)

        return NetworkConfig._build_network_from_config(config, neuron_class=neuron_class)

    @staticmethod
    def _build_network_from_config(
        config: Dict[str, Any],
        neuron_class: Optional[type] = None,
    ) -> NeuronNetwork:
        """Build a NeuronNetwork from configuration dictionary."""
        # Get simulation parameters
        sim_params = config.get(
            "simulation_params", NetworkConfig.DEFAULT_SIMULATION_PARAMS
        )
        max_history = sim_params.get("max_history", 1000)

        NeuronCls = neuron_class if neuron_class is not None else Neuron

        # Create custom NetworkTopology from config
        network_topology = NetworkConfig._create_topology_from_config(
            config, neuron_class=NeuronCls
        )

        # Create NeuronNetwork with custom topology
        sim = NeuronNetwork.__new__(NeuronNetwork)
        sim.current_tick = 0
        sim.max_history = max_history
        sim.network = network_topology

        # Initialize split event wheels (optimization)
        sim.max_delay = 10
        sim.wheel_size = sim.max_delay + 1
        sim.presynaptic_wheel = [[] for _ in range(sim.wheel_size)]
        sim.retrograde_wheel = [[] for _ in range(sim.wheel_size)]

        # Initialize history tracking
        from collections import defaultdict, deque

        sim.history = {
            "ticks": deque(maxlen=sim.max_history),
            "neuron_states": defaultdict(
                lambda: {
                    "membrane_potential": deque(maxlen=sim.max_history),
                    "firing": deque(maxlen=sim.max_history),
                    "firing_rate": deque(maxlen=sim.max_history),
                    "output": deque(maxlen=sim.max_history),
                }
            ),
            "network_activity": deque(maxlen=sim.max_history),
        }

        # Set up external inputs
        for external_input in config.get("external_inputs", []):
            neuron_id = external_input["target_neuron"]
            synapse_id = external_input["target_synapse"]

            # Ensure synapse_id is an integer for consistency
            if isinstance(synapse_id, str):
                try:
                    synapse_id = int(synapse_id)
                except ValueError:
                    # If it's a string like "syn_0", extract the number
                    if synapse_id.startswith("syn_"):
                        synapse_id = int(synapse_id[4:])
                    else:
                        raise ValueError(f"Invalid synapse ID format: {synapse_id}")

            info = external_input.get("info", 0.0)
            mod = np.array(external_input.get("mod", [0.0, 0.0]))
            sim.set_external_input(neuron_id, synapse_id, info, mod)

        return sim

    @staticmethod
    def _create_topology_from_config(
        config: Dict[str, Any],
        neuron_class: Optional[type] = None,
    ) -> NetworkTopology:
        """Create a NetworkTopology from configuration."""
        NeuronCls = neuron_class if neuron_class is not None else Neuron
        global_params = config.get("global_params", None)
        if isinstance(global_params, dict):
            # Convert dict to NeuronParameters
            if "lambda" in global_params:
                global_params["lambda_param"] = global_params.pop("lambda")
            array_keys = ["gamma", "w_r", "w_b", "w_tref"]
            for key in array_keys:
                if key in global_params and isinstance(global_params[key], list):
                    global_params[key] = np.array(global_params[key])
            global_params_obj = NeuronParameters(**global_params)
        elif isinstance(global_params, NeuronParameters):
            global_params_obj = global_params
        else:
            global_params_obj = NetworkConfig.DEFAULT_GLOBAL_PARAMS

        neurons_config = config.get("neurons", [])
        connections_config = config.get("connections", [])
        synaptic_points_config = config.get("synaptic_points", [])
        external_inputs_config = config.get("external_inputs", [])

        # Create empty topology
        topology = NetworkTopology.__new__(NetworkTopology)
        topology.neurons = {}
        # OPTIMIZATION: Initialize connection cache
        from collections import defaultdict

        topology.connection_cache = defaultdict(list)
        topology.fast_connection_cache = defaultdict(list)  # Initialize fast cache
        topology.connections = []
        topology.free_synapses = []
        topology.external_inputs = {}

        # Create neurons from config
        for neuron_config in neurons_config:
            neuron_id = neuron_config["id"]
            neuron_params = neuron_config.get("params", {})
            # Merge global_params_obj with neuron_params
            merged_params_dict = global_params_obj.__dict__.copy()
            merged_params_dict.update(neuron_params)
            if "lambda" in merged_params_dict:
                merged_params_dict["lambda_param"] = merged_params_dict.pop("lambda")
            array_keys = ["gamma", "w_r", "w_b", "w_tref"]
            for key in array_keys:
                if key in merged_params_dict and isinstance(
                    merged_params_dict[key], list
                ):
                    merged_params_dict[key] = np.array(merged_params_dict[key])
            params = NeuronParameters(**merged_params_dict)
            metadata = neuron_config.get("metadata", {})
            neuron = NeuronCls(neuron_id, params, log_level="CRITICAL", metadata=metadata)
            topology.neurons[neuron_id] = neuron

        # Import detailed synaptic points data
        NetworkConfig._restore_synaptic_points(topology.neurons, synaptic_points_config)

        # Create synapses for external inputs (if they don't exist from synaptic_points)
        for external_input in external_inputs_config:
            neuron_id = external_input["target_neuron"]
            synapse_id = external_input["target_synapse"]

            # Ensure synapse_id is an integer for consistency
            if isinstance(synapse_id, str):
                try:
                    synapse_id = int(synapse_id)
                except ValueError:
                    # If it's a string like "syn_0", extract the number
                    if synapse_id.startswith("syn_"):
                        synapse_id = int(synapse_id[4:])
                    else:
                        raise ValueError(f"Invalid synapse ID format: {synapse_id}")

            # Create synapse if it doesn't exist
            if neuron_id in topology.neurons:
                if synapse_id not in topology.neurons[neuron_id].postsynaptic_points:
                    distance = np.random.randint(2, 8)  # Default distance
                    topology.neurons[neuron_id].add_synapse(synapse_id, distance)

        # Set topology counts
        topology.num_neurons = len(topology.neurons)

        # Calculate synapses per neuron from existing synaptic points
        synapse_counts = {}
        for neuron_id, neuron in topology.neurons.items():
            synapse_counts[neuron_id] = len(neuron.postsynaptic_points)

        topology.synapses_per_neuron = (
            max(synapse_counts.values()) if synapse_counts else 5
        )

        # Add connections based on config
        for conn in connections_config:
            source_neuron_id = conn["source_neuron"]
            source_terminal_id = conn.get(
                "source_terminal", 0
            )  # Default to terminal 0 for backwards compatibility
            target_neuron_id = conn["target_neuron"]
            target_synapse_id = conn["target_synapse"]

            # Ensure IDs are integers for consistency
            if isinstance(target_synapse_id, str):
                try:
                    target_synapse_id = int(target_synapse_id)
                except ValueError:
                    # If it's a string like "syn_0", extract the number
                    if target_synapse_id.startswith("syn_"):
                        target_synapse_id = int(target_synapse_id[4:])
                    else:
                        raise ValueError(
                            f"Invalid synapse ID format: {target_synapse_id}"
                        )

            # Verify the target synapse exists (should have been created by synaptic_points import)
            if target_neuron_id in topology.neurons:
                if (
                    target_synapse_id
                    not in topology.neurons[target_neuron_id].postsynaptic_points
                ):
                    # If synaptic_points data wasn't provided, create with default
                    distance = conn.get("distance_to_hillock", np.random.randint(2, 8))
                    topology.neurons[target_neuron_id].add_synapse(
                        target_synapse_id, distance
                    )

                # Add connection with 4-element format
                topology.connections.append(
                    (
                        source_neuron_id,
                        source_terminal_id,
                        target_neuron_id,
                        target_synapse_id,
                    )
                )

                # Populate connection cache and register source for retrograde signaling
                topology.connection_cache[
                    (source_neuron_id, source_terminal_id)
                ].append((target_neuron_id, target_synapse_id))
                topology.neurons[target_neuron_id].register_source(
                    target_synapse_id, source_neuron_id, source_terminal_id
                )

        # Set up external inputs for synapses that were specified in config
        for external_input in external_inputs_config:
            neuron_id = external_input["target_neuron"]
            synapse_id = external_input["target_synapse"]

            # Ensure synapse_id is an integer for consistency
            if isinstance(synapse_id, str):
                try:
                    synapse_id = int(synapse_id)
                except ValueError:
                    # If it's a string like "syn_0", extract the number
                    if synapse_id.startswith("syn_"):
                        synapse_id = int(synapse_id[4:])
                    else:
                        raise ValueError(f"Invalid synapse ID format: {synapse_id}")

            info = external_input.get("info", 0.0)
            mod = np.array(external_input.get("mod", [0.0, 0.0]))

            topology.external_inputs[(neuron_id, synapse_id)] = {
                "info": info,
                "mod": mod,
                "plast": 0.0,
            }

        # Identify any remaining free synapses (not connected and not explicitly marked as external)
        # Only add default external inputs for neurons in the input layer (metadata layer == 0).
        for neuron_id, neuron in topology.neurons.items():
            # Determine if neuron belongs to input layer; if no metadata, skip adding defaults
            meta = getattr(neuron, "metadata", {}) or {}
            layer_idx = meta.get("layer", None)

            connected_synapses = {
                conn[3] for conn in topology.connections if conn[2] == neuron_id
            }
            external_input_synapses = {
                key[1] for key in topology.external_inputs.keys() if key[0] == neuron_id
            }

            for synapse_id in neuron.postsynaptic_points.keys():
                if (
                    synapse_id not in connected_synapses
                    and synapse_id not in external_input_synapses
                ):
                    topology.free_synapses.append((neuron_id, synapse_id))
                    if layer_idx == 0:
                        # Only input layer synapses receive default external input entries
                        topology.external_inputs[(neuron_id, synapse_id)] = {
                            "info": 0.0,
                            "mod": np.array([0.0, 0.0]),
                            "plast": 0.0,
                        }

        # Optimize runtime connections after all neurons and connections are set up
        topology.optimize_runtime_connections()

        return topology

    @staticmethod
    def _restore_synaptic_points(
        neurons: Dict[int, Neuron], synaptic_points_config: List[Dict[str, Any]]
    ) -> None:
        """Restore detailed synaptic points data from configuration."""
        # Handle imports for PostsynapticInputVector, etc.
        try:
            from .neuron import PostsynapticInputVector, PresynapticOutputVector
            from .neuron import PostsynapticPoint, PresynapticPoint
        except ImportError:
            from neuron import PostsynapticInputVector, PresynapticOutputVector
            from neuron import PostsynapticPoint, PresynapticPoint

        for point_data in synaptic_points_config:
            neuron_id = point_data["neuron_id"]

            if neuron_id not in neurons:
                continue  # Skip if neuron doesn't exist

            neuron = neurons[neuron_id]

            if point_data["type"] == "postsynaptic":
                # Restore postsynaptic point (synapse)
                synapse_id = point_data["synapse_id"]
                distance = point_data.get("distance_to_hillock", 0)
                potential = point_data.get("potential", 0.0)

                # Create PostsynapticInputVector with restored data
                u_i_data = point_data.get("u_i", {})
                u_i = PostsynapticInputVector(
                    info=u_i_data.get("info", 1.0),
                    plast=u_i_data.get("plast", 1.0),
                    adapt=np.array(u_i_data.get("adapt", [0.0, 0.0])),
                )

                # Create PostsynapticPoint with restored data
                postsynaptic_point = PostsynapticPoint(u_i=u_i, potential=potential)

                # Add to neuron
                neuron.postsynaptic_points[synapse_id] = postsynaptic_point
                neuron.distances[synapse_id] = distance

            elif point_data["type"] == "presynaptic":
                # Restore presynaptic point (axon terminal)
                terminal_id = point_data["terminal_id"]
                distance = point_data.get("distance_from_hillock", 0)

                # Create PresynapticOutputVector with restored data
                u_o_data = point_data.get("u_o", {})
                u_o = PresynapticOutputVector(
                    info=u_o_data.get("info", 1.0),
                    mod=np.array(u_o_data.get("mod", [0.0, 0.0])),
                )

                # Create PresynapticPoint with restored data
                u_i_retro = point_data.get("u_i_retro", 1.0)
                presynaptic_point = PresynapticPoint(u_o=u_o, u_i_retro=u_i_retro)

                # Add to neuron
                neuron.presynaptic_points[terminal_id] = presynaptic_point
                neuron.distances[terminal_id] = distance

    @staticmethod
    def _serialize_neuron_params(params: NeuronParameters) -> Dict[str, Any]:
        """Convert neuron parameters to JSON-serializable format."""
        serialized = {}

        # Convert dataclass to dict first, handling lambda_param -> lambda conversion
        if hasattr(params, "__dict__"):
            params_dict = params.__dict__

        for key, value in params_dict.items():
            # Convert lambda_param back to lambda for JSON compatibility
            json_key = "lambda" if key == "lambda_param" else key

            if isinstance(value, np.ndarray):
                serialized[json_key] = value.tolist()
            elif isinstance(value, np.floating):
                serialized[json_key] = float(value)
            elif isinstance(value, np.integer):
                serialized[json_key] = int(value)
            else:
                serialized[json_key] = value
        return serialized

    @staticmethod
    def _deserialize_neuron_params(params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert JSON parameters back to appropriate types for NeuronParameters creation."""
        deserialized = {}
        array_keys = ["gamma", "w_r", "w_b", "w_tref"]

        for key, value in params.items():
            if key in array_keys and isinstance(value, list):
                deserialized[key] = np.array(value)
            else:
                deserialized[key] = value

        # Also handle the case where global params are being merged
        # and need to convert lists to arrays
        for key in array_keys:
            if key in deserialized and isinstance(deserialized[key], list):
                deserialized[key] = np.array(deserialized[key])

        return deserialized

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> List[str]:
        """Validate network configuration and return list of errors."""
        errors = []

        # Check required sections
        required_sections = ["neurons", "connections"]
        for section in required_sections:
            if section not in config:
                errors.append(f"Missing required section: {section}")

        # Validate neurons
        if "neurons" in config:
            neuron_ids = set()
            for i, neuron in enumerate(config["neurons"]):
                if "id" not in neuron:
                    errors.append(f"Neuron {i} missing 'id' field")
                else:
                    neuron_id = neuron["id"]
                    if neuron_id in neuron_ids:
                        errors.append(f"Duplicate neuron ID: {neuron_id}")
                    neuron_ids.add(neuron_id)

        # Validate synaptic points
        if "synaptic_points" in config and "neurons" in config:
            neuron_ids = {n["id"] for n in config["neurons"] if "id" in n}

            for i, point in enumerate(config["synaptic_points"]):
                if "neuron_id" not in point:
                    errors.append(f"Synaptic point {i} missing 'neuron_id' field")
                elif point["neuron_id"] not in neuron_ids:
                    errors.append(
                        f"Synaptic point {i} references unknown neuron: {point['neuron_id']}"
                    )

                if "type" not in point:
                    errors.append(f"Synaptic point {i} missing 'type' field")
                elif point["type"] not in ["postsynaptic", "presynaptic"]:
                    errors.append(
                        f"Synaptic point {i} has invalid type: {point['type']}"
                    )

                if point.get("type") == "postsynaptic" and "synapse_id" not in point:
                    errors.append(f"Postsynaptic point {i} missing 'synapse_id' field")
                elif point.get("type") == "presynaptic" and "terminal_id" not in point:
                    errors.append(f"Presynaptic point {i} missing 'terminal_id' field")

        # Validate connections
        if "connections" in config and "neurons" in config:
            # Safely collect neuron IDs (only from neurons that have IDs)
            neuron_ids = {n["id"] for n in config["neurons"] if "id" in n}

            for i, conn in enumerate(config["connections"]):
                required_fields = ["source_neuron", "target_neuron", "target_synapse"]
                for field in required_fields:
                    if field not in conn:
                        errors.append(f"Connection {i} missing '{field}' field")

                if "source_neuron" in conn and conn["source_neuron"] not in neuron_ids:
                    errors.append(
                        f"Connection {i} references unknown source neuron: {conn['source_neuron']}"
                    )

                if "target_neuron" in conn and conn["target_neuron"] not in neuron_ids:
                    errors.append(
                        f"Connection {i} references unknown target neuron: {conn['target_neuron']}"
                    )

        return errors

    @staticmethod
    def create_sample_config(
        num_neurons: int = 5, connectivity: float = 0.3
    ) -> Dict[str, Any]:
        """Create a sample network configuration."""
        config = NetworkConfig.create_empty_config()
        config["metadata"]["name"] = f"Sample Network ({num_neurons} neurons)"
        config["metadata"]["description"] = (
            f"Randomly generated network with {connectivity:.1%} connectivity"
        )

        # Create neurons
        neurons = []
        for i in range(num_neurons):
            neuron = {
                "id": i,  # Use integer ID
                "params": {
                    "r_base": float(np.random.uniform(0.8, 1.2)),
                    "b_base": float(np.random.uniform(1.0, 1.4)),
                },
            }
            neurons.append(neuron)
        config["neurons"] = neurons

        # Create sample synaptic points
        synaptic_points = []
        for neuron_i in range(num_neurons):
            # Add postsynaptic points (synapses) for each neuron
            for syn_idx in range(3):  # 3 synapses per neuron
                synapse_point = {
                    "neuron_id": neuron_i,
                    "synapse_id": syn_idx,
                    "type": "postsynaptic",
                    "distance_to_hillock": int(np.random.randint(2, 8)),
                    "potential": 0.0,
                    "u_i": {
                        "info": float(np.random.uniform(0.5, 1.5)),
                        "plast": float(np.random.uniform(0.5, 1.5)),
                        "adapt": [
                            float(np.random.uniform(0.1, 0.5)),
                            float(np.random.uniform(0.1, 0.5)),
                        ],
                    },
                }
                synaptic_points.append(synapse_point)

            # Add some presynaptic points (axon terminals) for variety
            if np.random.random() < 0.7:  # 70% chance of having terminals
                num_terminals = np.random.randint(1, 3)  # 1-2 terminals
                for term_idx in range(num_terminals):
                    terminal_point = {
                        "neuron_id": neuron_i,
                        "terminal_id": term_idx,
                        "type": "presynaptic",
                        "distance_from_hillock": int(np.random.randint(5, 15)),
                        "u_o": {
                            "info": float(np.random.uniform(0.5, 1.5)),
                            "mod": [
                                float(np.random.uniform(0.1, 0.5)),
                                float(np.random.uniform(0.1, 0.5)),
                            ],
                        },
                        "u_i_retro": float(np.random.uniform(0.5, 1.5)),
                    }
                    synaptic_points.append(terminal_point)

        config["synaptic_points"] = synaptic_points

        # Create random connections
        connections = []
        import random

        for target_i in range(num_neurons):
            for syn_idx in range(3):  # 3 synapses per neuron
                if random.random() < connectivity:
                    source_i = random.choice(
                        [i for i in range(num_neurons) if i != target_i]
                    )
                    connection = {
                        "source_neuron": source_i,  # Use integer ID
                        "target_neuron": target_i,  # Use integer ID
                        "target_synapse": syn_idx,  # Use integer ID
                        "properties": {},
                    }
                    connections.append(connection)
        config["connections"] = connections

        # Add some external inputs
        external_inputs = []
        for i in range(min(2, num_neurons)):  # Add inputs to first 2 neurons
            external_input = {
                "target_neuron": i,  # Use integer ID
                "target_synapse": 0,  # Use integer ID
                "info": 2.0,
                "mod": [0.0, 0.0],
            }
            external_inputs.append(external_input)
        config["external_inputs"] = external_inputs

        return config
