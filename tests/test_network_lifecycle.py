import json
import heapq
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from neuron.network import NeuronNetwork, TravelingSignal
from neuron.network_config import NetworkConfig
from neuron.neuron import Neuron, NeuronParameters
from neuron.nn_core import NNCore


class RecordingNeuron(Neuron):
    """Allow a delivery-only tick hook without modifying the slotted model."""


def recording_network():
    """Exercise delivery without depending on firing or learning equations."""
    simulation = NeuronNetwork(num_neurons=0, synapses_per_neuron=0)
    received = {}
    for neuron_id in (1, 2):
        neuron = RecordingNeuron(
            neuron_id, NeuronParameters(num_inputs=2), log_level="CRITICAL"
        )
        for synapse_id in (0, 1):
            neuron.add_synapse(synapse_id, distance_to_hillock=1)
        neuron.add_axon_terminal(0, distance_from_hillock=1)
        neuron.presynaptic_points[0].u_o.mod[:] = [0.25, 0.5]

        def tick(external_inputs, current_tick, dt=1.0, cell=neuron):
            received[cell.id] = cell.input_buffer.copy()
            cell.input_buffer.fill(0.0)
            return []

        neuron.tick = tick
        simulation.network.neurons[neuron_id] = neuron
    simulation.network.num_neurons = 2
    simulation.network.synapses_per_neuron = 2
    return simulation, received


def queue_release(simulation, source=1, terminal=0, strength=3.0):
    simulation.presynaptic_wheel[simulation.current_tick % simulation.wheel_size].append(
        TravelingSignal((source, terminal, strength), simulation.current_tick)
    )


class ExternalInputLifecycleTests(unittest.TestCase):
    def test_rebuild_preserves_pending_inputs_without_replaying_consumed_inputs(self):
        simulation, received = recording_network()
        simulation.set_external_input(1, 0, 3.0, np.array([0.1, 0.2]))
        simulation.run_tick()
        np.testing.assert_allclose(received[1][0], [3.0, 0.0, 0.1, 0.2])

        # A new key rebuilds the mirror after one pending drive was already set.
        simulation.set_external_input(1, 0, 7.0, np.array([0.3, 0.4]))
        simulation.set_external_input(2, 0, 4.0)
        simulation.run_tick()
        np.testing.assert_allclose(received[1][0], [7.0, 0.0, 0.3, 0.4])
        self.assertEqual(received[2][0, 0], 4.0)

        # Consumed bootstrap inputs must not return on another rebuild.
        simulation.set_external_input(2, 1, 2.0)
        simulation.run_tick()
        np.testing.assert_array_equal(received[1], np.zeros((2, 4)))
        self.assertEqual(received[2][0, 0], 0.0)
        self.assertEqual(received[2][1, 0], 2.0)

    def test_short_modulation_replaces_previous_pending_vector(self):
        simulation, received = recording_network()
        simulation.set_external_input(1, 0, 0.0)
        simulation.run_tick()
        simulation.set_external_input(1, 0, 1.0, np.array([0.2, 0.8]))
        simulation.set_external_input(1, 0, 2.0, np.array([0.5]))
        simulation.run_tick()
        np.testing.assert_allclose(received[1][0], [2.0, 0.0, 0.5, 0.0])

    def test_reset_discards_pending_drive_and_keeps_buffer_aliases(self):
        simulation, received = recording_network()
        simulation.network.connections.append((1, 0, 2, 0))
        simulation.network.optimize_runtime_connections()
        target_buffer = simulation.network.fast_connection_cache[(1, 0)][0][0]
        simulation.set_external_input(1, 0, 3.0)
        simulation.run_tick()
        simulation.set_external_input(1, 0, 9.0, np.array([0.2, 0.7]))
        target_buffer[:] = 4.0
        queue_release(simulation)
        simulation.reset_simulation()

        self.assertIs(target_buffer, simulation.network.neurons[2].input_buffer)
        np.testing.assert_array_equal(target_buffer, np.zeros((2, 4)))
        simulation.run_tick()
        np.testing.assert_array_equal(received[1], np.zeros((2, 4)))
        np.testing.assert_array_equal(received[2], np.zeros((2, 4)))

    def test_export_round_trips_pending_mirror_instead_of_consumed_drive(self):
        simulation, received = recording_network()
        simulation.set_external_input(1, 0, 3.0)
        simulation.run_tick()
        simulation.set_external_input(1, 0, 7.0, np.array([0.3, 0.4]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "network.json"
            NetworkConfig.save_network_config(simulation, path)
            config = json.loads(path.read_text())
            self.assertEqual(config["external_inputs"][0]["info"], 7.0)
            np.testing.assert_allclose(config["external_inputs"][0]["mod"], [0.3, 0.4])
            restored = NetworkConfig.load_network_config(path)
        self.assertIsNone(restored.network._ext_vec)
        self.assertFalse(restored.record_history)
        restored.network._ensure_ext_vectorized()
        self.assertEqual(restored.network.get_external_inputs()[(1, 0)]["info"], 7.0)
        # Exporting must not consume the original pending input.
        simulation.run_tick()
        self.assertEqual(received[1][0, 0], 7.0)


class InteractiveTopologyTests(unittest.TestCase):
    def setUp(self):
        self.simulation, self.received = recording_network()
        self.core = NNCore()
        self.core.neural_net = self.simulation

    def test_connection_rebuild_is_idempotent_and_tracks_retrograde_source(self):
        self.assertTrue(self.core.add_connection(1, 0, 2, 0))
        for _ in range(2):
            self.simulation.network.optimize_runtime_connections()
        self.assertEqual(self.simulation.network.get_connection_mapping(1, 0), [(2, 0)])
        self.assertEqual(self.simulation.network.neurons[2].synapse_sources[0], (1, 0))
        queue_release(self.simulation)
        self.simulation.run_tick()
        np.testing.assert_allclose(self.received[2][0], [3.0, 0.0, 0.25, 0.5])

    def test_clearing_connections_stops_delivery_and_removes_source_map(self):
        self.core.add_connection(1, 0, 2, 0)
        self.assertTrue(self.core.clear_all_connections())
        self.assertEqual(self.simulation.network.neurons[2].synapse_sources, {})
        self.assertIn((2, 0), self.simulation.network.free_synapses)
        queue_release(self.simulation)
        self.simulation.run_tick()
        np.testing.assert_array_equal(self.received[2], np.zeros((2, 4)))

    def test_deleting_source_drops_stale_scheduled_release(self):
        self.core.add_connection(1, 0, 2, 0)
        self.simulation.set_external_input(1, 1, 3.0)
        self.simulation.run_tick()
        queue_release(self.simulation)
        self.assertTrue(self.core.delete_neuron(1))
        self.assertNotIn((1, 1), self.simulation.network.external_inputs)
        self.simulation.run_tick()
        np.testing.assert_array_equal(self.received[2], np.zeros((2, 4)))
        self.assertEqual(self.simulation.network.num_neurons, 1)

    def test_deleting_synapse_removes_delivery_and_integer_distance(self):
        self.core.add_connection(1, 0, 2, 0)
        self.assertEqual(self.core.get_synapse(2, 0)["distance_to_hillock"], 1)
        self.assertTrue(self.core.delete_synapse(2, 0))
        self.assertNotIn(0, self.simulation.network.neurons[2].distances)
        self.assertEqual(self.simulation.network.neurons[2].synapse_sources, {})
        queue_release(self.simulation)
        self.simulation.run_tick()
        np.testing.assert_array_equal(self.received[2], np.zeros((2, 4)))

    def test_deleting_missing_or_negative_synapse_has_no_side_effects(self):
        neuron = self.simulation.network.neurons[2]
        neuron.input_buffer[:] = 4.0
        self.core.add_connection(1, 0, 2, 0)
        before = neuron.input_buffer.copy()
        distances = neuron.distances.copy()
        connections = self.simulation.network.connections.copy()
        for synapse_id in (-1, 2, 4096):
            self.assertFalse(self.core.delete_synapse(2, synapse_id))
            np.testing.assert_array_equal(neuron.input_buffer, before)
            self.assertEqual(neuron.distances, distances)
            self.assertEqual(self.simulation.network.connections, connections)
            self.assertEqual(set(neuron.postsynaptic_points), {0, 1})

    def test_deleting_synapse_discards_its_inflight_intracellular_arrivals(self):
        target = self.simulation.network.neurons[2]
        target.propagation_queue = [
            (2, "hillock", 10.0, 0),
            (4, "hillock", 2.0, 1),
            (3, "hillock", 20.0, 0),
            (6, "hillock", 3.0, 1),
            (5, "hillock", 4.0, 1),
        ]
        heapq.heapify(target.propagation_queue)
        self.assertTrue(self.core.delete_synapse(2, 0))
        # Native arrival processing must never dereference the deleted distance.
        self.assertEqual(target._hillock_current(3, 1.0), 0.0)
        # Keep the model's configured input count for native ticking by replacing
        # the deleted slot; old drive must not transfer to this replacement.
        self.core.add_synapse(2, 0, 1)
        target.tick = Neuron.tick.__get__(target, Neuron)
        target.tick({}, 3)
        self.assertEqual(target.S, 0.0)
        self.assertEqual(target.O, 0.0)
        current = target._hillock_current(6, 1.0)
        self.assertAlmostEqual(current, 9.0 * target.params.delta_decay)
        self.assertEqual(target.propagation_queue, [])


class TickOwnerLifecycleTests(unittest.TestCase):
    def test_start_updates_rate_for_existing_healthy_worker_without_spawning(self):
        core = NNCore()
        core.time_thread = MagicMock()
        core.time_thread.is_alive.return_value = True
        core.state.is_running = True
        core.state.is_paused = True
        with patch("neuron.nn_core.threading.Thread") as thread_factory:
            self.assertFalse(core.start_time_flow(tick_rate=4.0))
        thread_factory.assert_not_called()
        self.assertEqual(core.state.tick_rate, 4.0)
        self.assertFalse(core.state.is_paused)

    def test_import_joins_existing_worker_before_replacement(self):
        core = NNCore()
        old_thread = MagicMock()
        old_thread.is_alive.return_value = True
        core.time_thread = old_thread
        core.state.is_running = True
        replacement = NeuronNetwork(num_neurons=0, synapses_per_neuron=0)

        def finish_join(timeout):
            self.assertEqual(timeout, 1.0)
            self.assertFalse(core.lock.locked())
            self.assertTrue(core._stop_event.is_set())
            old_thread.is_alive.return_value = False

        old_thread.join.side_effect = finish_join
        with patch.object(NetworkConfig, "load_network_config", return_value=replacement):
            self.assertTrue(core.import_network("replacement.json"))
        old_thread.join.assert_called_once_with(timeout=1.0)
        self.assertIs(core.neural_net, replacement)
        with patch("neuron.nn_core.threading.Thread") as thread_factory:
            self.assertTrue(core.start_time_flow())
            thread_factory.return_value.start.assert_called_once()

    def test_import_times_out_without_loading_while_old_worker_is_stuck(self):
        core = NNCore()
        core.time_thread = MagicMock()
        core.time_thread.is_alive.return_value = True
        core.lock.acquire()
        try:
            with patch.object(NetworkConfig, "load_network_config") as load:
                self.assertFalse(core.import_network("replacement.json"))
            load.assert_not_called()
        finally:
            core.lock.release()
        self.assertTrue(core._stop_event.is_set())

    def test_start_does_not_clear_stop_or_spawn_when_old_worker_is_alive(self):
        core = NNCore()
        core.time_thread = MagicMock()
        core.time_thread.is_alive.return_value = True
        core.state.is_running = False
        core._stop_event.set()
        with patch("neuron.nn_core.threading.Thread") as thread_factory:
            self.assertFalse(core.start_time_flow())
        thread_factory.assert_not_called()
        self.assertTrue(core._stop_event.is_set())


if __name__ == "__main__":
    unittest.main()
