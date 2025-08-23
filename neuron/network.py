#!/usr/bin/env python3
"""
Multi-Neuron Network Implementation
Core functionality for creating and simulating multi-neuron networks with
signal propagation and network topology management.
"""

import numpy as np
import random
from collections import defaultdict, deque
from typing import Dict, List, Any, Tuple, Optional, Callable

# Handle imports for both direct execution and module execution
try:
    from .neuron import (
        Neuron,
        NeuronParameters,
        NeuronEvent,
        PresynapticReleaseEvent,
        RetrogradeSignalEvent,
        PresynapticOutputVector,
    )
except ImportError:
    from neuron import (
        Neuron,
        NeuronParameters,
        NeuronEvent,
        PresynapticReleaseEvent,
        RetrogradeSignalEvent,
        PresynapticOutputVector,
    )

MIN_CONNECTION_SIGNAL_TRAVEL_TICKS = 1
MAX_CONNECTION_SIGNAL_TRAVEL_TICKS = 1


class NetworkTopology:
    """Manages the network topology and connections between neurons."""

    def __init__(self, num_neurons: int, synapses_per_neuron: int = 5):
        self.num_neurons = num_neurons
        self.synapses_per_neuron = synapses_per_neuron
        self.neurons = {}
        self.connections = (
            []
        )  # (source_neuron_id, source_terminal_id, target_neuron_id, target_synapse_id)
        self.free_synapses = []  # synapses not connected to other neurons
        self.external_inputs = {}  # external input sources

        self._create_network()

    def _create_network(self):
        """Create the network with random connections."""
        # Create neurons with NeuronParameters
        for i in range(self.num_neurons):
            neuron_id = random.randint(0, 2**36 - 1)
            # Add some parameter variation
            params = NeuronParameters(
                r_base=np.random.uniform(0.8, 1.2),
                b_base=np.random.uniform(1.0, 1.4),
                num_neuromodulators=2,
                num_inputs=self.synapses_per_neuron,  # Match actual number of synapses
                c=10,
                lambda_param=20,
                p=1.0,
                gamma=np.array([0.99, 0.995]),
                beta_avg=0.999,
                w_r=np.array([-0.2, 0.05]),
                w_b=np.array([-0.2, 0.05]),
                w_tref=np.array([-20.0, 10.0]),
                delta_decay=0.95,
                eta_post=0.01,
                eta_retro=0.01,
            )

            self.neurons[neuron_id] = Neuron(neuron_id, params, log_level="WARNING")

            # Add axon terminals to this neuron (similar number to synapses)
            for terminal_idx in range(self.synapses_per_neuron):
                terminal_id = terminal_idx  # Use integer terminal ID
                distance = random.randint(2, 8)
                self.neurons[neuron_id].add_axon_terminal(
                    terminal_id, distance_from_hillock=distance
                )

        # Create random connections
        self._create_connections()

    def _create_connections(self):
        """Create random connections between neurons."""
        neuron_ids = list(self.neurons.keys())

        for target_neuron_id in neuron_ids:
            target_neuron = self.neurons[target_neuron_id]

            # Create synapses for this neuron
            for syn_idx in range(self.synapses_per_neuron):
                synapse_id = syn_idx  # Use integer synapse ID
                distance = random.randint(2, 8)
                target_neuron.add_synapse(synapse_id, distance_to_hillock=distance)

                # Randomly decide if this synapse connects to another neuron or is free
                if (
                    random.random() < 0.7 and len(neuron_ids) > 1
                ):  # 70% chance of connection
                    # Connect to a random source neuron (not self)
                    possible_sources = [
                        nid for nid in neuron_ids if nid != target_neuron_id
                    ]
                    source_neuron_id = random.choice(possible_sources)

                    # Choose a random terminal from the source neuron
                    source_terminal_id = random.randint(0, self.synapses_per_neuron - 1)

                    self.connections.append(
                        (
                            source_neuron_id,
                            source_terminal_id,
                            target_neuron_id,
                            synapse_id,
                        )
                    )
                else:
                    # Free synapse - can receive external input
                    self.free_synapses.append((target_neuron_id, synapse_id))
                    self.external_inputs[(target_neuron_id, synapse_id)] = {
                        "info": 0.0,
                        "mod": np.array([0.0, 0.0]),
                        "plast": 0.0,
                    }

    def get_neuron_connections(self, neuron_id: int) -> List[Tuple[int, int, int]]:
        """Get all connections from a given neuron.

        Returns:
            List of (terminal_id, target_neuron_id, target_synapse_id) tuples
            where terminal_id is the terminal on the given neuron_id
        """
        return [
            (terminal_id, target_neuron_id, target_synapse_id)
            for source_neuron_id, terminal_id, target_neuron_id, target_synapse_id in self.connections
            if source_neuron_id == neuron_id
        ]

    def get_incoming_connections(self, neuron_id: int) -> List[Tuple[int, int, int]]:
        """Get all connections to a given neuron.

        Returns:
            List of (source_neuron_id, source_terminal_id, synapse_id) tuples
            where synapse_id is the synapse on the given neuron_id
        """
        return [
            (source_neuron_id, source_terminal_id, synapse_id)
            for source_neuron_id, source_terminal_id, target_neuron_id, synapse_id in self.connections
            if target_neuron_id == neuron_id
        ]

    def get_connection_mapping(
        self, source_neuron_id: int, source_terminal_id: int
    ) -> List[Tuple[int, int]]:
        """Get all target neurons/synapses connected to a specific source terminal.

        Returns:
            List of (target_neuron_id, target_synapse_id) tuples
        """
        return [
            (target_neuron_id, target_synapse_id)
            for src_neuron_id, src_terminal_id, target_neuron_id, target_synapse_id in self.connections
            if src_neuron_id == source_neuron_id
            and src_terminal_id == source_terminal_id
        ]

    def set_external_input(
        self, input_key: Tuple[int, int], info: float, mod: Optional[np.ndarray] = None
    ):
        """Set external input for a specific synapse."""
        # Create external input entry if it doesn't exist
        if input_key not in self.external_inputs:
            self.external_inputs[input_key] = {
                "info": 0.0,
                "mod": np.array([0.0, 0.0]),
                "plast": 0.0,
            }

        # Update the values
        self.external_inputs[input_key]["info"] = info
        if mod is not None:
            self.external_inputs[input_key]["mod"] = mod

    def get_synaptic_density(self) -> float:
        """Return synaptic density: fraction of synaptic slots used (dynamic)."""
        total_synapses = sum(len(n.postsynaptic_points) for n in self.neurons.values())
        return len(self.connections) / total_synapses if total_synapses > 0 else 0.0

    def get_graph_density(self) -> float:
        """Return graph density: fraction of possible directed edges used (dynamic, no self-loops)."""
        num_neurons = len(self.neurons)
        max_possible = num_neurons * (num_neurons - 1) if num_neurons > 1 else 0
        return len(self.connections) / max_possible if max_possible > 0 else 0.0

    def get_network_statistics(self) -> Dict[str, Any]:
        """Get basic network statistics."""
        max_connections = self.num_neurons * self.synapses_per_neuron
        connection_density = (
            len(self.connections) / max_connections if max_connections > 0 else 0.0
        )
        graph_density = self.get_graph_density()
        synaptic_density = self.get_synaptic_density()
        return {
            "num_neurons": self.num_neurons,
            "num_connections": len(self.connections),
            "num_free_synapses": len(self.free_synapses),
            "connection_density": connection_density,  # legacy
            "synaptic_density": synaptic_density,
            "graph_density": graph_density,
            "neurons": list(self.neurons.keys()),
        }


class TravelingSignal:
    """Represents a signal traveling through the network."""

    def __init__(
        self,
        event: NeuronEvent,  # type: ignore
        travel_time: int,
        current_tick: int,
    ):
        self.event = event
        self.travel_time = travel_time
        self.start_tick = current_tick
        self.arrival_tick = current_tick + travel_time

    def has_arrived(self, current_tick: int) -> bool:
        """Check if signal has arrived at its destination."""
        return current_tick >= self.arrival_tick

    def __repr__(self) -> str:
        if isinstance(self.event, PresynapticReleaseEvent):
            return (
                f"TravelingSignal(PresynapticRelease: {self.event.source_neuron_id} -> ?, "
                f"arrival_tick={self.arrival_tick})"
            )
        elif isinstance(self.event, RetrogradeSignalEvent):
            return (
                f"TravelingSignal(Retrograde: {self.event.source_neuron_id} -> {self.event.target_neuron_id}, "
                f"arrival_tick={self.arrival_tick})"
            )
        else:
            return (
                f"TravelingSignal(Unknown event type, arrival_tick={self.arrival_tick})"
            )


class NeuronNetwork:
    """Core multi-neuron network simulation without GUI components."""

    def __init__(self, num_neurons: int = 5, synapses_per_neuron: int = 5):
        # Simulation state
        self.current_tick = 0
        self.max_history = 1000

        # Network setup
        self.network = NetworkTopology(num_neurons, synapses_per_neuron)
        self.traveling_signals = []

        # History tracking
        self.history = {
            "ticks": deque(maxlen=self.max_history),
            "neuron_states": defaultdict(
                lambda: {
                    "membrane_potential": deque(maxlen=self.max_history),
                    "firing": deque(maxlen=self.max_history),
                    "firing_rate": deque(maxlen=self.max_history),
                    "output": deque(maxlen=self.max_history),
                }
            ),
            "network_activity": deque(
                maxlen=self.max_history
            ),  # Total network firing rate
        }

    def set_external_input(
        self,
        neuron_id: int,
        synapse_id: int,
        info: float,
        mod: Optional[np.ndarray] = None,
    ):
        """Set external input for a specific neuron's synapse."""
        input_key = (neuron_id, synapse_id)
        self.network.set_external_input(input_key, info, mod)

    def add_signal(
        self,
        source_neuron: int,
        target_neuron: int,
        target_synapse: int,
        signal_strength: float = 1.5,
        travel_time: int = 3,
    ):
        """Manually add a traveling signal to the network."""
        signal = TravelingSignal(
            event=PresynapticReleaseEvent(
                source_neuron_id=source_neuron,
                source_terminal_id=target_synapse,  # Use synapse as terminal for backward compatibility
                signal_vector=PresynapticOutputVector(info=signal_strength),
                timestamp=self.current_tick,
            ),
            travel_time=travel_time,
            current_tick=self.current_tick,
        )
        self.traveling_signals.append(signal)

    def run_tick(self) -> Dict[str, Any]:
        """Execute one simulation tick and return activity summary."""
        # Update traveling signals and collect arrived ones
        arrived_signals = []
        for signal in self.traveling_signals[:]:
            if signal.has_arrived(self.current_tick):
                arrived_signals.append(signal)
                self.traveling_signals.remove(signal)

        # Prepare inputs for each neuron
        neuron_inputs = defaultdict(dict)

        # Add external inputs
        for input_key, input_data in self.network.external_inputs.items():
            neuron_id, synapse_id = input_key
            if neuron_id in self.network.neurons:
                # Copy external input data but note that it doesn't have source info
                # so retrograde signals won't be generated for external inputs
                neuron_inputs[neuron_id][synapse_id] = input_data.copy()

            # Clear external inputs that were used in this tick
            self.network.external_inputs[input_key] = {
                "info": 0.0,
                "mod": np.array([0.0, 0.0]),
                "plast": 0.0,
            }

        # Process arrived signals based on their event type
        for signal in arrived_signals:
            event = signal.event

            if isinstance(event, PresynapticReleaseEvent):
                # This is a signal from a presynaptic terminal to postsynaptic synapses
                # We need to find which neurons have connections from this source
                source_neuron_id = event.source_neuron_id
                source_terminal_id = event.source_terminal_id

                # Find all target neurons/synapses connected to this specific source terminal
                for (
                    target_neuron_id,
                    target_synapse_id,
                ) in self.network.get_connection_mapping(
                    source_neuron_id, source_terminal_id
                ):
                    if target_neuron_id in self.network.neurons:
                        if target_synapse_id not in neuron_inputs[target_neuron_id]:
                            neuron_inputs[target_neuron_id][target_synapse_id] = {
                                "info": 0.0,
                                "plast": 0.0,
                                "mod": np.array([0.0, 0.0]),
                                "source_neuron_id": source_neuron_id,
                                "source_terminal_id": source_terminal_id,
                            }

                        # Add signal components from the presynaptic release
                        neuron_inputs[target_neuron_id][target_synapse_id][
                            "info"
                        ] += event.signal_vector.info
                        neuron_inputs[target_neuron_id][target_synapse_id][
                            "mod"
                        ] += event.signal_vector.mod

            elif isinstance(event, RetrogradeSignalEvent):
                # This is a retrograde signal going back to a presynaptic terminal
                target_neuron_id = event.target_neuron_id
                if target_neuron_id in self.network.neurons:
                    target_neuron: Neuron = self.network.neurons[target_neuron_id]
                    target_neuron.process_retrograde_signal(event)

        # Run simulation for each neuron and collect events
        all_events = []
        fired_neurons = []
        for neuron_id, neuron in self.network.neurons.items():
            inputs = neuron_inputs[neuron_id]
            events = neuron.tick(inputs, self.current_tick, dt=1.0)
            all_events.extend(events)

            if neuron.O > 0:  # Neuron fired (for history tracking)
                fired_neurons.append(neuron_id)

        # Generate new traveling signals from all events
        new_signals = []
        for event in all_events:
            # Determine travel time based on event type
            travel_time = random.randint(
                MIN_CONNECTION_SIGNAL_TRAVEL_TICKS,
                MAX_CONNECTION_SIGNAL_TRAVEL_TICKS,
            )

            signal = TravelingSignal(
                event,
                travel_time,
                self.current_tick,
            )
            new_signals.append(signal)

        self.traveling_signals.extend(new_signals)

        # Record history
        self.history["ticks"].append(self.current_tick)
        total_activity = 0
        for neuron_id, neuron in self.network.neurons.items():
            self.history["neuron_states"][neuron_id]["membrane_potential"].append(
                neuron.S
            )
            firing = 1 if neuron.O > 0 else 0
            self.history["neuron_states"][neuron_id]["firing"].append(firing)
            self.history["neuron_states"][neuron_id]["firing_rate"].append(neuron.F_avg)
            self.history["neuron_states"][neuron_id]["output"].append(neuron.O)
            total_activity += firing

        self.history["network_activity"].append(total_activity)

        self.current_tick += 1

        # Count event types
        presynaptic_events = sum(
            1 for e in all_events if isinstance(e, PresynapticReleaseEvent)
        )
        retrograde_events = sum(
            1 for e in all_events if isinstance(e, RetrogradeSignalEvent)
        )

        # Return activity summary
        return {
            "tick": self.current_tick - 1,
            "fired_neurons": fired_neurons,
            "arrived_signals": len(arrived_signals),
            "traveling_signals": len(self.traveling_signals),
            "new_signals": len(new_signals),
            "total_activity": total_activity,
            "total_events": len(all_events),
            "presynaptic_events": presynaptic_events,
            "retrograde_events": retrograde_events,
        }

    def run_simulation(
        self, num_ticks: int, progress_callback: Optional[Callable] = None
    ) -> List[Dict[str, Any]]:
        """Run simulation for multiple ticks and return activity log."""
        activity_log = []

        for i in range(num_ticks):
            activity = self.run_tick()
            activity_log.append(activity)

            if progress_callback and i % 10 == 0:
                progress_callback(i, num_ticks, activity)

        return activity_log

    def reset_simulation(self):
        """Reset the simulation to initial state."""
        self.current_tick = 0
        self.traveling_signals.clear()

        # Reset neuron states
        for neuron_id, neuron in self.network.neurons.items():
            neuron.S = 0.0
            neuron.O = 0.0
            neuron.t_last_fire = -np.inf
            neuron.F_avg = 0.0
            neuron.M_vector = np.zeros(neuron.params.num_neuromodulators)
            neuron.r = neuron.params.r_base
            neuron.b = neuron.params.b_base
            neuron.t_ref = neuron.upper_t_ref_bound
            neuron.propagation_queue.clear()

        # Clear history
        self.history["ticks"].clear()
        for neuron_data in self.history["neuron_states"].values():
            for queue in neuron_data.values():
                queue.clear()
        self.history["network_activity"].clear()

        # Reset external inputs
        for input_key in self.network.external_inputs:
            self.network.external_inputs[input_key] = {
                "info": 0.0,
                "mod": np.array([0.0, 0.0]),
                "plast": 0.0,
            }

    def get_network_state(self) -> Dict[str, Any]:
        """Get current state of the entire network."""
        neuron_states = {}
        for neuron_id, neuron in self.network.neurons.items():
            neuron_states[neuron_id] = {
                "membrane_potential": neuron.S,
                "output": neuron.O,
                "firing_rate": neuron.F_avg,
                "threshold": neuron.r,
                "refractory_period": neuron.t_ref,
                "last_fire_tick": neuron.t_last_fire,
            }

        return {
            "current_tick": self.current_tick,
            "neurons": neuron_states,
            "traveling_signals": len(self.traveling_signals),
            "network_stats": self.network.get_network_statistics(),
            "recent_activity": (
                list(self.history["network_activity"])[-10:]
                if self.history["network_activity"]
                else []
            ),
        }

    def get_history(self, neuron_id: Optional[str] = None) -> Dict[str, Any]:
        """Get simulation history for analysis."""
        if neuron_id and neuron_id in self.history["neuron_states"]:
            return {
                "ticks": list(self.history["ticks"]),
                "neuron": {
                    key: list(queue)
                    for key, queue in self.history["neuron_states"][neuron_id].items()
                },
            }

        return {
            "ticks": list(self.history["ticks"]),
            "network_activity": list(self.history["network_activity"]),
            "all_neurons": {
                neuron_id: {key: list(queue) for key, queue in neuron_data.items()}
                for neuron_id, neuron_data in self.history["neuron_states"].items()
            },
        }
