#!/usr/bin/env python3
"""
Neural Network Core Engine
Core simulation state and neural network operations for multi-neuron networks.
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

from neuron.neuron import NeuronParameters

# Network imports
from .network_config import NetworkConfig
from .network import NeuronNetwork
from . import Neuron


@dataclass
class NNCoreState:
    """State container for the NN Core"""

    current_tick: int = 0
    is_running: bool = False
    tick_rate: float = 1.0  # ticks per second
    last_tick_time: float = field(default_factory=time.time)


class NNCore:
    """
    Neural Network Core Engine
    Handles simulation state and neural network operations
    """

    def __init__(self):
        self.state = NNCoreState()
        self.neural_net: Optional[NeuronNetwork] = None
        self.lock = threading.Lock()
        self.time_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start_time_flow(self, tick_rate: float = 1.0):
        """Start autonomous time flow at specified rate"""
        with self.lock:
            if self.state.is_running:
                return False

            self.state.tick_rate = tick_rate
            self.state.is_running = True
            self._stop_event.clear()

            # Start time thread
            self.time_thread = threading.Thread(target=self._time_loop, daemon=True)
            self.time_thread.start()

            return True

    def stop_time_flow(self):
        """Stop autonomous time flow"""
        with self.lock:
            if not self.state.is_running:
                return False

            self.state.is_running = False
            self._stop_event.set()

            if self.time_thread:
                self.time_thread.join(timeout=1.0)

            return True

    def _time_loop(self):
        """Internal time loop for autonomous ticking"""
        while not self._stop_event.is_set():
            start_time = time.time()

            # Execute tick
            self.do_tick()

            # Calculate sleep time
            elapsed = (time.time() - start_time) * 1000  # ms
            sleep_time = max(0, (1000.0 / self.state.tick_rate) - elapsed) / 1000.0

            if sleep_time > 0:
                self._stop_event.wait(sleep_time)

    def do_tick(self) -> Dict[str, Any]:
        """Execute a single simulation tick"""
        with self.lock:
            if self.neural_net is None:
                return {"error": "No simulation loaded"}

            try:
                # Execute the simulation tick
                start_time = time.perf_counter()
                tick_result = self.neural_net.run_tick()
                execution_time_ms = (time.perf_counter() - start_time) * 1000

                # Update state
                self.state.current_tick = self.neural_net.current_tick
                self.state.last_tick_time = time.time()

                # Check for neural activity
                fired_neurons = []
                for neuron_id, neuron in self.neural_net.network.neurons.items():
                    if neuron.O > 0:  # Neuron fired
                        fired_neurons.append(neuron_id)

                return {
                    "tick": self.state.current_tick,
                    "execution_time_ms": execution_time_ms,
                    "fired_neurons": fired_neurons,
                }

            except Exception as e:
                return {"error": str(e)}

    def do_n_ticks(self, n: int) -> List[Dict[str, Any]]:
        """Execute N simulation ticks"""
        results = []
        for _ in range(n):
            result = self.do_tick()
            results.append(result)
            if "error" in result:
                break
        return results

    def import_network(self, config_file: str) -> bool:
        """Import network from config file"""
        with self.lock:
            try:
                self.neural_net = NetworkConfig.load_network_config(config_file)
                self.state.current_tick = 0
                return True
            except Exception as e:
                print(f"Error importing network: {e}")
                return False

    def export_network(self, config_file: str, metadata: Optional[Dict] = None) -> bool:
        """Export current network to config file"""
        with self.lock:
            try:
                if self.neural_net is None:
                    return False

                if metadata is None:
                    metadata = {
                        "name": f"Network Export - Tick {self.state.current_tick}",
                        "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "current_tick": self.state.current_tick,
                    }

                NetworkConfig.save_network_config(
                    self.neural_net, config_file, metadata
                )
                return True
            except Exception as e:
                print(f"Error exporting network: {e}")
                return False

    def add_neuron(self, neuron_id: int, params: NeuronParameters) -> bool:
        """Add a single neuron to the network"""
        with self.lock:
            try:
                if self.neural_net is None:
                    self.neural_net = NeuronNetwork(
                        num_neurons=0, synapses_per_neuron=0
                    )

                # Create neuron with given parameters
                neuron = Neuron(neuron_id, params, log_level="WARNING")
                self.neural_net.network.neurons[neuron_id] = neuron
                return True
            except Exception as e:
                print(f"Error adding neuron: {e}")
                return False

    def get_neuron(self, neuron_id: int) -> Optional[Dict[str, Any]]:
        """Get neuron information by ID"""
        with self.lock:
            if (
                self.neural_net is None
                or neuron_id not in self.neural_net.network.neurons
            ):
                return None

            neuron = self.neural_net.network.neurons[neuron_id]
            return {
                "id": neuron.id,
                "membrane_potential": neuron.S,
                "firing_rate": neuron.F_avg,
                "output": neuron.O,
                "params": neuron.params,
                "synapses": list(neuron.postsynaptic_points.keys()),
                "terminals": list(neuron.presynaptic_points.keys()),
            }

    def delete_neuron(self, neuron_id: int) -> bool:
        """Delete neuron by ID"""
        with self.lock:
            try:
                if (
                    self.neural_net is None
                    or neuron_id not in self.neural_net.network.neurons
                ):
                    return False

                # Remove from neurons
                del self.neural_net.network.neurons[neuron_id]

                # Remove associated connections
                self.neural_net.network.connections = [
                    conn
                    for conn in self.neural_net.network.connections
                    if conn[0] != neuron_id
                    and conn[2] != neuron_id  # source_neuron_id and target_neuron_id
                ]

                return True
            except Exception as e:
                print(f"Error deleting neuron: {e}")
                return False

    def export_neuron(self, neuron_id: int) -> Optional[Dict[str, Any]]:
        """Export single neuron configuration"""
        neuron_data = self.get_neuron(neuron_id)
        if neuron_data:
            return {
                "neuron": neuron_data,
                "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "tick": self.state.current_tick,
            }
        return None

    def add_synapse(self, neuron_id: int, synapse_id: int, distance: int) -> bool:
        """Add synapse to neuron"""
        with self.lock:
            try:
                if (
                    self.neural_net is None
                    or neuron_id not in self.neural_net.network.neurons
                ):
                    return False

                neuron = self.neural_net.network.neurons[neuron_id]
                neuron.add_synapse(synapse_id, distance)
                return True
            except Exception as e:
                print(f"Error adding synapse: {e}")
                return False

    def get_synapse(self, neuron_id: int, synapse_id: int) -> Optional[Dict[str, Any]]:
        """Get synapse information"""
        with self.lock:
            if (
                self.neural_net is None
                or neuron_id not in self.neural_net.network.neurons
            ):
                return None

            neuron = self.neural_net.network.neurons[neuron_id]
            if synapse_id not in neuron.postsynaptic_points:
                return None

            synapse = neuron.postsynaptic_points[synapse_id]
            distance = neuron.distances.get((synapse_id, "hillock"), 0)

            return {
                "neuron_id": neuron_id,
                "synapse_id": synapse_id,
                "distance_to_hillock": distance,
                "potential": synapse.potential,
                "u_i": {
                    "info": synapse.u_i.info,
                    "plast": synapse.u_i.plast,
                    "adapt": synapse.u_i.adapt.tolist(),
                },
            }

    def delete_synapse(self, neuron_id: int, synapse_id: int) -> bool:
        """Delete synapse by ID"""
        with self.lock:
            try:
                if (
                    self.neural_net is None
                    or neuron_id not in self.neural_net.network.neurons
                ):
                    return False

                neuron = self.neural_net.network.neurons[neuron_id]
                if synapse_id in neuron.postsynaptic_points:
                    del neuron.postsynaptic_points[synapse_id]

                # Remove from distances
                if (synapse_id, "hillock") in neuron.distances:
                    del neuron.distances[(synapse_id, "hillock")]

                return True
            except Exception as e:
                print(f"Error deleting synapse: {e}")
                return False

    def send_signal(
        self, neuron_id: int, synapse_id: int, signal_strength: float = 1.5
    ) -> bool:
        """Send signal to specific synapse"""
        with self.lock:
            try:
                if self.neural_net is None:
                    return False

                # Set external input directly (not traveling signal)
                # This ensures the signal reaches the neuron immediately
                self.neural_net.set_external_input(
                    neuron_id=neuron_id,
                    synapse_id=synapse_id,
                    info=signal_strength,
                    mod=None,
                )
                return True
            except Exception as e:
                print(f"Error sending signal: {e}")
                return False

    def send_batch_signals(self, signals: List[Tuple[int, int, float]]) -> int:
        """Send batch of signals, return number of successful sends"""
        success_count = 0
        for neuron_id, synapse_id, strength in signals:
            if self.send_signal(neuron_id, synapse_id, strength):
                success_count += 1
        return success_count

    def get_network_state(self) -> Dict[str, Any]:
        """Get comprehensive network state"""
        with self.lock:
            if self.neural_net is None:
                return {"error": "No network loaded"}

            # Core state information
            core_state = {
                "is_running": self.state.is_running,
                "current_tick": self.state.current_tick,
                "tick_rate": self.state.tick_rate,
                "last_tick_time": self.state.last_tick_time,
            }

            # Network state information
            network = {
                "neurons": {},
                "connections": [],
                "external_inputs": {},
                "traveling_signals": [],
            }

            for neuron_id, neuron in self.neural_net.network.neurons.items():
                network["neurons"][neuron_id] = {
                    "membrane_potential": neuron.S,
                    "firing_rate": neuron.F_avg,
                    "output": neuron.O,
                    "synapses": list(neuron.postsynaptic_points.keys()),
                    "terminals": list(neuron.presynaptic_points.keys()),
                }

            network["connections"] = self.neural_net.network.connections
            network["external_inputs"] = {
                f"{k[0]}_{k[1]}": v
                for k, v in self.neural_net.network.external_inputs.items()
            }

            # Add detailed traveling signal information
            for signal in self.neural_net.traveling_signals:
                event = signal.event
                signal_info = {
                    "travel_time": signal.travel_time,
                    "start_tick": signal.start_tick,
                    "arrival_tick": signal.arrival_tick,
                }

                # Extract event-specific information
                if hasattr(event, "source_neuron_id"):
                    signal_info["source_neuron"] = event.source_neuron_id
                if hasattr(event, "target_neuron_id"):
                    signal_info["target_neuron"] = event.target_neuron_id
                if hasattr(event, "source_terminal_id"):
                    signal_info["source_terminal"] = event.source_terminal_id
                if hasattr(event, "target_terminal_id"):
                    signal_info["target_terminal"] = event.target_terminal_id
                if hasattr(event, "signal_vector"):
                    signal_info["signal_strength"] = event.signal_vector.info
                    signal_info["target_synapse"] = "N/A"  # This varies per connection
                elif hasattr(event, "error_vector"):
                    signal_info["signal_strength"] = f"Error: {event.error_vector}"
                    signal_info["target_synapse"] = "N/A"

                # Add event type for clarity
                signal_info["event_type"] = type(event).__name__

                network["traveling_signals"].append(signal_info)

            # Add density stats
            stats = self.neural_net.network.get_network_statistics()
            network["synaptic_density"] = stats.get("synaptic_density", 0.0)
            network["graph_density"] = stats.get("graph_density", 0.0)

            return {"core_state": core_state, "network": network}

    def add_connection(
        self,
        source_neuron_id: int,
        source_terminal_id: int,
        target_neuron_id: int,
        target_synapse_id: int,
    ) -> bool:
        """Add a synaptic connection between two neurons"""
        with self.lock:
            try:
                if self.neural_net is None:
                    return False

                # Validate source neuron exists
                if source_neuron_id not in self.neural_net.network.neurons:
                    print(f"Source neuron {source_neuron_id} not found")
                    return False

                # Validate target neuron exists
                if target_neuron_id not in self.neural_net.network.neurons:
                    print(f"Target neuron {target_neuron_id} not found")
                    return False

                # Validate source terminal exists
                source_neuron = self.neural_net.network.neurons[source_neuron_id]
                if source_terminal_id not in source_neuron.presynaptic_points:
                    print(
                        f"Terminal {source_terminal_id} not found in source neuron {source_neuron_id}"
                    )
                    return False

                # Validate target synapse exists
                target_neuron = self.neural_net.network.neurons[target_neuron_id]
                if target_synapse_id not in target_neuron.postsynaptic_points:
                    print(
                        f"Synapse {target_synapse_id} not found in target neuron {target_neuron_id}"
                    )
                    return False

                # Check if connection already exists
                connection_tuple = (
                    source_neuron_id,
                    source_terminal_id,
                    target_neuron_id,
                    target_synapse_id,
                )
                if connection_tuple in self.neural_net.network.connections:
                    print(
                        f"Connection already exists: {source_neuron_id}:{source_terminal_id} -> {target_neuron_id}:{target_synapse_id}"
                    )
                    return False

                # Add the connection
                self.neural_net.network.connections.append(connection_tuple)

                # Remove from free synapses if it was there
                free_synapse_key = (target_neuron_id, target_synapse_id)
                if free_synapse_key in self.neural_net.network.free_synapses:
                    self.neural_net.network.free_synapses.remove(free_synapse_key)

                # Remove from external inputs if it was there
                if free_synapse_key in self.neural_net.network.external_inputs:
                    del self.neural_net.network.external_inputs[free_synapse_key]

                return True

            except Exception as e:
                print(f"Error adding connection: {e}")
                return False

    def auto_connect_neurons(
        self,
        min_free_synapses: int = 1,
        min_free_terminals: int = 1,
        connectivity: float = 0.5,
    ) -> bool:
        """Automatically connect neurons randomly while maintaining minimum free synapses and terminals"""
        with self.lock:
            try:
                if self.neural_net is None:
                    return False

                import random

                neurons = self.neural_net.network.neurons
                if len(neurons) < 2:
                    print("Need at least 2 neurons to create connections")
                    return False

                connections_added = 0
                total_attempts = 0
                max_attempts = len(neurons) * 50  # Prevent infinite loops

                neuron_ids = list(neurons.keys())

                # Track outgoing connections per neuron for terminal constraints
                outgoing_connections = {neuron_id: 0 for neuron_id in neuron_ids}
                for conn in self.neural_net.network.connections:
                    source_neuron_id = conn[0]
                    if source_neuron_id in outgoing_connections:
                        outgoing_connections[source_neuron_id] += 1

                for target_id in neuron_ids:
                    target_neuron = neurons[target_id]
                    available_synapses = list(target_neuron.postsynaptic_points.keys())

                    # Get currently connected synapses for this neuron (only neuron-to-neuron connections)
                    connected_synapses = {
                        conn[3]  # target_synapse_id is now at index 3
                        for conn in self.neural_net.network.connections
                        if conn[2] == target_id  # target_neuron_id is now at index 2
                    }

                    # Find synapses that could be used for new connections
                    # This includes both unconnected synapses AND synapses with external inputs
                    free_synapses = []
                    for syn_id in available_synapses:
                        is_connected_to_neuron = syn_id in connected_synapses
                        if not is_connected_to_neuron:
                            free_synapses.append(syn_id)

                    # Determine how many we can connect (leaving min_free_synapses free)
                    connectable_count = max(0, len(free_synapses) - min_free_synapses)
                    target_connections = int(connectable_count * connectivity)

                    # Randomly select synapses to connect
                    if target_connections > 0:
                        synapses_to_connect = random.sample(
                            free_synapses,
                            min(
                                target_connections,
                                len(free_synapses) - min_free_synapses,
                            ),
                        )

                        for synapse_id in synapses_to_connect:
                            total_attempts += 1
                            if total_attempts > max_attempts:
                                break

                            # Find source neurons that still have free terminals
                            eligible_sources = []
                            for nid in neuron_ids:
                                if nid != target_id:  # Can't connect to self
                                    source_neuron = neurons[nid]
                                    total_terminals = len(
                                        source_neuron.presynaptic_points
                                    )
                                    used_terminals = outgoing_connections[nid]

                                    # Check if this neuron has terminals and available capacity
                                    if (
                                        total_terminals > 0
                                        and (total_terminals - used_terminals)
                                        > min_free_terminals
                                    ):
                                        eligible_sources.append(nid)

                            if eligible_sources:
                                source_id = random.choice(eligible_sources)

                                # Choose a random terminal from the source neuron
                                source_neuron = neurons[source_id]
                                available_terminals = list(
                                    source_neuron.presynaptic_points.keys()
                                )
                                if available_terminals:
                                    source_terminal_id = random.choice(
                                        available_terminals
                                    )

                                    # Add connection with 4-element format
                                    connection_tuple = (
                                        source_id,
                                        source_terminal_id,
                                        target_id,
                                        synapse_id,
                                    )
                                    if (
                                        connection_tuple
                                        not in self.neural_net.network.connections
                                    ):
                                        self.neural_net.network.connections.append(
                                            connection_tuple
                                        )
                                        connections_added += 1

                                        # Update outgoing connections count for source neuron
                                        outgoing_connections[source_id] += 1

                                    # Remove from external inputs if it was there
                                    free_synapse_key = (target_id, synapse_id)
                                    if (
                                        free_synapse_key
                                        in self.neural_net.network.external_inputs
                                    ):
                                        del self.neural_net.network.external_inputs[
                                            free_synapse_key
                                        ]

                                    # Remove from free synapses list if it was there
                                    if (
                                        free_synapse_key
                                        in self.neural_net.network.free_synapses
                                    ):
                                        self.neural_net.network.free_synapses.remove(
                                            free_synapse_key
                                        )

                    if total_attempts > max_attempts:
                        break

                print(f"Auto-connected neurons: {connections_added} connections added")

                # Report final statistics
                total_synapses = sum(
                    len(n.postsynaptic_points) for n in neurons.values()
                )
                total_terminals = sum(
                    len(n.presynaptic_points) for n in neurons.values()
                )
                total_connections = len(self.neural_net.network.connections)
                external_inputs_count = len(self.neural_net.network.external_inputs)
                truly_free_synapses = (
                    total_synapses - total_connections - external_inputs_count
                )
                used_terminals = sum(outgoing_connections.values())
                free_terminals = total_terminals - used_terminals

                print(
                    f"Network stats: {total_connections} neuron connections, "
                    f"{external_inputs_count} external inputs, {truly_free_synapses} free synapses"
                )
                print(
                    f"Terminal usage: {used_terminals}/{total_terminals} used, "
                    f"{free_terminals} free terminals available"
                )

                return True

            except Exception as e:
                print(f"Error in auto-connect: {e}")
                return False

    def clear_all_connections(self) -> bool:
        """Clear all neuron-to-neuron connections, converting synapses back to external inputs"""
        with self.lock:
            try:
                if self.neural_net is None:
                    return False

                connections_cleared = len(self.neural_net.network.connections)

                # Convert all connected synapses back to external inputs
                for (
                    source_id,
                    target_id,
                    synapse_id,
                ) in self.neural_net.network.connections:
                    synapse_key = (target_id, synapse_id)

                    # Add as external input with default values
                    self.neural_net.network.external_inputs[synapse_key] = {
                        "info": 0.0,
                        "mod": np.array([0.0, 0.0]),
                    }

                # Clear all connections
                self.neural_net.network.connections.clear()

                print(f"Cleared {connections_cleared} connections")
                print(f"Converted {connections_cleared} synapses to external inputs")

                return True

            except Exception as e:
                print(f"Error clearing connections: {e}")
                return False

    def set_log_level(self, level: str) -> bool:
        """Set log level for all neuron loggers"""
        try:
            from loguru import logger
            import neuron.neuron as neuron_module

            # Validate level
            valid_levels = [
                "TRACE",
                "DEBUG",
                "INFO",
                "SUCCESS",
                "WARNING",
                "ERROR",
                "CRITICAL",
            ]
            if level.upper() not in valid_levels:
                print(
                    f"Invalid log level: {level}. Valid levels: {', '.join(valid_levels)}"
                )
                return False

            # Reconfigure logger
            logger.remove()
            logger.add(
                lambda msg: print(msg, end=""),
                format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                + "<level>{level: <8}</level> | "
                + "<cyan>N:{extra[neuron_int]}({extra[neuron_hex]})</cyan> | "
                + "<level>{message}</level>",
                level=level.upper(),
                colorize=True,
            )

            print(f"Log level set to: {level.upper()}")
            return True

        except Exception as e:
            print(f"Error setting log level: {e}")
            return False
