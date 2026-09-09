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
    )
except ImportError:
    from neuron import (
        Neuron,
        NeuronParameters,
        NeuronEvent,
        PresynapticReleaseEvent,
        RetrogradeSignalEvent,
    )

# Synaptic cleft travel time in ticks
# External signal travel, separate from intra-neuron signal travel
MIN_CONNECTION_SIGNAL_TRAVEL_TICKS = 1
MAX_CONNECTION_SIGNAL_TRAVEL_TICKS = 1


class NetworkTopology:
    """Manages the network topology and connections between neurons."""

    def __init__(self, num_neurons: int, synapses_per_neuron: int = 5):
        self.num_neurons = num_neurons
        self.synapses_per_neuron = synapses_per_neuron
        self.neurons = {}
        # OPTIMIZATION: Connection Cache (O(1) lookup)
        # Map: (source_id, terminal_id) -> list of (target_id, synapse_id)
        self.connection_cache = defaultdict(list)
        # OPTIMIZATION: Fast Connection Cache (Direct buffer references)
        # Map: (source_id, terminal_id) -> list of (buffer_ref, synapse_index)
        self.fast_connection_cache = defaultdict(list)
        self.connections = []  # (source_neuron_id, source_terminal_id, target_neuron_id, target_synapse_id)
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

        # OPTIMIZATION: Build fast connection cache with direct references
        self.optimize_runtime_connections()

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
                    # OPTIMIZATION: Populate Cache
                    self.connection_cache[
                        (source_neuron_id, source_terminal_id)
                    ].append((target_neuron_id, synapse_id))
                    # Register source for retrograde signaling
                    self.neurons[target_neuron_id].register_source(
                        synapse_id, source_neuron_id, source_terminal_id
                    )
                else:
                    # Free synapse - can receive external input
                    self.free_synapses.append((target_neuron_id, synapse_id))
                    self.external_inputs[(target_neuron_id, synapse_id)] = {
                        "info": 0.0,
                        "mod": np.array([0.0, 0.0]),
                        "plast": 0.0,
                    }

    def optimize_runtime_connections(self):
        """
        Converts ID-based connections to Direct-Reference connections.
        Call this AFTER creating all neurons.
        """
        # Fast Cache: Map (src_id, src_term) -> List of (Target_Buffer, Target_Syn_Idx)
        for src, src_term, tgt, tgt_syn in self.connections:
            # Get the ACTUAL numpy array object from the target neuron
            target_neuron = self.neurons[tgt]
            target_buffer = target_neuron.input_buffer

            # Store the reference + index tuple
            self.fast_connection_cache[(src, src_term)].append((target_buffer, tgt_syn))

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
        # OPTIMIZATION: Direct Dict Lookup O(1)
        return self.connection_cache.get((source_neuron_id, source_terminal_id), [])

    def set_external_input(
        self, input_key: Tuple[int, int], info: float, mod: Optional[np.ndarray] = None
    ):
        """Set external input for a specific synapse."""
        # Once the vectorized mirror is built it is authoritative for run_tick, so write the
        # value straight into it and skip the dict entirely (the dict's post-tick state is
        # zero either way -- run_tick clears the arrays, not the dict -- which is all any
        # external reader observes between ticks).
        vec = getattr(self, "_ext_vec", None)
        if vec is not None:
            r = vec["row_of"].get(input_key)
            if r is not None:
                vec["info"][r] = info
                if mod is not None:
                    m = np.asarray(mod, dtype=vec["mod"].dtype)
                    vec["mod"][r, : min(2, m.shape[0])] = m[:2]
                return
            self._ext_vec = None  # unknown key: rebuild from the dict next tick

        # Bootstrap / new-key path: write the dict, which is the source _ensure_ext_vectorized
        # reads when it (re)builds the arrays.
        if input_key not in self.external_inputs:
            self.external_inputs[input_key] = {
                "info": 0.0,
                "mod": np.array([0.0, 0.0]),
                "plast": 0.0,
            }
            self._ext_vec = None
        self.external_inputs[input_key]["info"] = info
        if mod is not None:
            self.external_inputs[input_key]["mod"] = mod
        if "plast" not in self.external_inputs[input_key]:
            self.external_inputs[input_key]["plast"] = 0.0

    def _ensure_ext_vectorized(self):
        """Build/refresh a vectorized mirror of external_inputs so run_tick can scatter the
        drive into neuron buffers with per-neuron vectorized writes + a single vectorized
        clear, instead of a Python loop over EVERY external synapse (which dominates tick cost
        under dense sensory/pixel drive). Rebuilt only when the external-input key set changes;
        set_external_input keeps the value arrays in sync between rebuilds.

        The presence of this method is the fast-path opt-in: run_tick uses it only for real
        NetworkTopology instances. Duck-typed topologies (e.g. the C. elegans connectome
        loader) lack it and take the original dict-loop path unchanged."""
        ext = self.external_inputs
        cache = getattr(self, "_ext_vec", None)
        if cache is not None and cache["nkeys"] == len(ext):
            return cache
        keys = list(ext.keys())
        row_of = {k: i for i, k in enumerate(keys)}
        n = len(keys)
        info = np.zeros(n, dtype=np.float64)
        plast = np.zeros(n, dtype=np.float64)
        mod = np.zeros((n, 2), dtype=np.float64)
        for i, k in enumerate(keys):
            d = ext[k]
            info[i] = d.get("info", 0.0)
            plast[i] = d.get("plast", 0.0)
            m = d.get("mod")
            if m is not None:
                m = np.asarray(m)
                mod[i, : min(2, m.shape[0])] = m[:2]
        by_n: dict = {}
        for i, (nid, sid) in enumerate(keys):
            if nid in self.neurons:
                by_n.setdefault(nid, []).append((sid, i))
        groups = []
        for nid, items in by_n.items():
            syn = np.fromiter((s for s, _ in items), dtype=np.intp, count=len(items))
            rows = np.fromiter((r for _, r in items), dtype=np.intp, count=len(items))
            groups.append((self.neurons[nid].input_buffer, syn, rows))
        cache = {"nkeys": n, "keys": keys, "row_of": row_of,
                 "info": info, "plast": plast, "mod": mod, "groups": groups}
        self._ext_vec = cache
        return cache

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
        arrival_tick: int,
    ):
        self.event = event
        self.arrival_tick = arrival_tick

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

    def __init__(
        self,
        num_neurons: int = 5,
        synapses_per_neuron: int = 5,
        record_history: bool = False,
    ):
        # Simulation state
        self.current_tick = 0
        self.max_history = 1000
        self.record_history = record_history

        # Network setup
        self.network = NetworkTopology(num_neurons, synapses_per_neuron)

        # OPTIMIZATION 2: Split Calendar Queues (Wheels)
        # Separate wheels for presynaptic and retrograde events to avoid isinstance checks
        self.max_delay = 10
        self.wheel_size = self.max_delay + 1
        self.presynaptic_wheel = [[] for _ in range(self.wheel_size)]
        self.retrograde_wheel = [[] for _ in range(self.wheel_size)]

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
        # Create lightweight tuple event: (source_id, terminal_id, info_value)
        event_tuple = (source_neuron, target_synapse, signal_strength)
        signal = TravelingSignal(
            event=event_tuple,
            arrival_tick=self.current_tick + travel_time,
        )
        # Schedule using presynaptic wheel (all manual signals are presynaptic)
        slot = (signal.arrival_tick) % self.wheel_size
        self.presynaptic_wheel[slot].append(signal)

    def run_tick(self) -> Dict[str, Any]:
        """Execute one simulation tick and return activity summary."""
        # 1. Pop Events for this Tick O(1) - Split Wheels
        slot = self.current_tick % self.wheel_size
        pre_signals = self.presynaptic_wheel[slot]
        retro_signals = self.retrograde_wheel[slot]
        self.presynaptic_wheel[slot] = []  # Clear for reuse
        self.retrograde_wheel[slot] = []  # Clear for reuse

        # Add external inputs directly to neuron buffers and collect for neuromodulation.
        neuron_external_inputs = {}  # {neuron_id: {synapse_id: {info, mod, plast}}}
        topo = self.network
        if hasattr(topo, "_ensure_ext_vectorized"):
            # ---- VECTORIZED FAST PATH (real NetworkTopology) ----
            # Scatter the drive with per-neuron vectorized buffer writes and a numpy mod mask,
            # instead of a Python loop over every external synapse. This is exactly equivalent
            # to the fallback below: writing a zero into an input_buffer row is a no-op (the
            # buffer self-zeroes each tick and external synapses are disjoint from connection
            # targets), and a zero mod contributes 0 to Neuron.tick's neuromodulatory sum, so
            # only nonzero-mod rows need the mod write + collection.
            vec = topo._ensure_ext_vectorized()
            info_a, plast_a, mod_a = vec["info"], vec["plast"], vec["mod"]
            # Free-running guard: when nothing is driven every value is already zero, so the
            # scatter would only write zeros into already-zero buffers. A couple of vectorized
            # .any() checks skip the whole per-neuron loop in that case (keeps un-driven ticks
            # as cheap as the zero-skip fallback).
            if info_a.any() or plast_a.any() or mod_a.any():
                for buf, syn, rows in vec["groups"]:
                    buf[syn, 0] = info_a[rows]
                    buf[syn, 1] = plast_a[rows]
                nz = np.flatnonzero(np.abs(mod_a).sum(axis=1) > 0.0)
                if nz.size:
                    keys = vec["keys"]
                    neurons = topo.neurons
                    for r in nz:
                        nid, sid = keys[int(r)]
                        if nid in neurons:
                            m = mod_a[r]
                            neurons[nid].input_buffer[sid, 2:] = m
                            neuron_external_inputs.setdefault(nid, {})[sid] = {
                                "info": float(info_a[r]),
                                "mod": m.copy(),
                                "plast": float(plast_a[r]),
                            }
                # Clear the vectorized store (vectorized). The dict is untouched: it is already
                # at its zero post-tick state (set_external_input writes the arrays once the
                # mirror exists), so external readers see the same zeros the legacy path left.
                info_a.fill(0.0)
                plast_a.fill(0.0)
                mod_a.fill(0.0)
        else:
            # ---- FALLBACK dict loop (duck-typed topologies, e.g. C. elegans loader) ----
            for input_key, input_data in topo.external_inputs.items():
                info = input_data.get("info", 0.0)
                plast = input_data.get("plast", 0.0)
                mod = input_data.get("mod")
                mod_nonzero = mod is not None and any(mod)
                if info == 0.0 and plast == 0.0 and not mod_nonzero:
                    continue

                neuron_id, synapse_id = input_key
                if neuron_id in topo.neurons:
                    neuron = topo.neurons[neuron_id]
                    neuron.input_buffer[synapse_id, 0] = info
                    neuron.input_buffer[synapse_id, 1] = plast
                    if mod_nonzero:
                        neuron.input_buffer[synapse_id, 2:] = mod
                        if neuron_id not in neuron_external_inputs:
                            neuron_external_inputs[neuron_id] = {}
                        neuron_external_inputs[neuron_id][synapse_id] = {
                            "info": info,
                            "mod": mod.copy(),
                            "plast": plast,
                        }

                input_data["info"] = 0.0
                input_data["plast"] = 0.0
                _m = input_data.get("mod")
                if _m is not None:
                    _m[...] = 0.0

        # 2. Process Presynaptic Events - Ultra Fast Direct Buffer Writes
        for signal in pre_signals:
            event = signal.event

            # Handle tuple events: (source_id, terminal_id, info_value)
            if isinstance(event, tuple) and len(event) == 3:
                src_id, term_id, sig_info = event

                # ULTRA OPTIMIZATION: Direct buffer references - no ID lookups!
                targets = self.network.fast_connection_cache.get((src_id, term_id), [])

                # Get modulation from source neuron
                sig_mod = (
                    self.network.neurons[src_id].presynaptic_points[term_id].u_o.mod
                )

                # Direct memory writes - zero overhead
                for target_buf, tgt_syn_idx in targets:
                    target_buf[tgt_syn_idx, 0] += sig_info
                    target_buf[tgt_syn_idx, 2:] += sig_mod

        # 3. Process Retrograde Events
        for signal in retro_signals:
            event = signal.event

            if isinstance(event, RetrogradeSignalEvent):
                # Retrograde events go to specific neurons for processing
                target_neuron_id = event.target_neuron_id
                if target_neuron_id in self.network.neurons:
                    target_neuron = self.network.neurons[target_neuron_id]
                    target_neuron.process_retrograde_signal(event)

            elif isinstance(event, RetrogradeSignalEvent):
                # This is a retrograde signal going back to a presynaptic terminal
                target_neuron_id = event.target_neuron_id
                if target_neuron_id in self.network.neurons:
                    target_neuron: Neuron = self.network.neurons[target_neuron_id]
                    target_neuron.process_retrograde_signal(event)

        # 3. Update Neurons (they read from their own input buffers)
        all_events = []
        fired_neurons = []
        for neuron_id, neuron in self.network.neurons.items():
            # Pass external inputs to tick() for neuromodulation processing
            external_inputs_for_neuron = neuron_external_inputs.get(neuron_id, {})
            events = neuron.tick(external_inputs_for_neuron, self.current_tick, dt=1.0)
            all_events.extend(events)

            if neuron.O > 0:  # Neuron fired (for history tracking)
                fired_neurons.append(neuron_id)

        # 4. Schedule New Events to Appropriate Wheels
        for event in all_events:
            # Random delay
            delay = random.randint(
                MIN_CONNECTION_SIGNAL_TRAVEL_TICKS,
                MAX_CONNECTION_SIGNAL_TRAVEL_TICKS,
            )
            target_slot = (self.current_tick + delay) % self.wheel_size
            signal = TravelingSignal(event, arrival_tick=self.current_tick + delay)

            # Route to appropriate wheel based on event type
            if isinstance(event, tuple) or isinstance(event, PresynapticReleaseEvent):
                self.presynaptic_wheel[target_slot].append(signal)
            elif isinstance(event, RetrogradeSignalEvent):
                self.retrograde_wheel[target_slot].append(signal)

        total_activity = len(fired_neurons)
        if self.record_history:
            self.history["ticks"].append(self.current_tick)
            for neuron_id, neuron in self.network.neurons.items():
                self.history["neuron_states"][neuron_id]["membrane_potential"].append(
                    neuron.S
                )
                firing = 1 if neuron.O > 0 else 0
                self.history["neuron_states"][neuron_id]["firing"].append(firing)
                self.history["neuron_states"][neuron_id]["firing_rate"].append(
                    neuron.F_avg
                )
                self.history["neuron_states"][neuron_id]["output"].append(neuron.O)

            self.history["network_activity"].append(total_activity)

        self.current_tick += 1

        # Count traveling signals in both wheels
        traveling_signals_count = sum(
            len(slot) for slot in self.presynaptic_wheel
        ) + sum(len(slot) for slot in self.retrograde_wheel)

        # Return activity summary
        return {
            "tick": self.current_tick - 1,
            "fired_neurons": fired_neurons,
            "arrived_signals": len(pre_signals) + len(retro_signals),
            "traveling_signals": traveling_signals_count,
            "new_signals": len(all_events),  # All events become new signals
            "total_activity": total_activity,
            "total_events": len(all_events),
            "presynaptic_events": len(pre_signals),
            "retrograde_events": len(retro_signals),
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
        # Clear both event wheels
        for slot in self.presynaptic_wheel:
            slot.clear()
        for slot in self.retrograde_wheel:
            slot.clear()

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
            # Opt-in intracellular extensions own state beyond the scalar soma.
            # Ordinary neurons have no hook and keep the existing reset path.
            reset_extra = getattr(neuron, "reset_additional_state", None)
            if reset_extra is not None:
                reset_extra()

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
            "traveling_signals": (
                sum(len(slot) for slot in self.presynaptic_wheel)
                + sum(len(slot) for slot in self.retrograde_wheel)
            ),
            "network_stats": self.network.get_network_statistics(),
            "recent_activity": (
                list(self.history["network_activity"])[-10:]
                if self.history["network_activity"]
                else []
            ),
        }

    def get_history(self, neuron_id: Optional[str] = None) -> Dict[str, Any]:
        """Get simulation history for analysis.

        When ``record_history`` is False, per-tick history is not stored; this
        returns empty series (ticks, activity, neuron traces).
        """
        if not self.record_history:
            if neuron_id:
                return {"ticks": [], "neuron": {}}
            return {
                "ticks": [],
                "network_activity": [],
                "all_neurons": {},
            }

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
