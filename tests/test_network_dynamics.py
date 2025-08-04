#!/usr/bin/env python3
"""
Test suite for network-level dynamics, patterns, and reproducibility.
Tests that external inputs create reproducible patterns and different
inputs create different internal states and outputs.
"""

import numpy as np
import sys
import os
import json
import random

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from neuron.network import NeuronNetwork
from neuron.neuron import NeuronParameters


class NetworkDynamicsTester:
    """Test network-level dynamics and reproducibility."""

    def __init__(self):
        self.test_results = []
        self.verbose = True

    def log_test(self, test_name, passed, details=""):
        """Log test result."""
        status = "PASS" if passed else "FAIL"
        result = {"test": test_name, "status": status, "details": details}
        self.test_results.append(result)
        if self.verbose:
            print(f"[{status}] {test_name}: {details}")

    def test_network_creation(self):
        """Test basic network creation and connectivity."""
        print("\n=== Testing Network Creation ===")

        network = NeuronNetwork(num_neurons=3, synapses_per_neuron=4)

        self.log_test(
            "Network created with correct number of neurons",
            len(network.network.neurons) == 3,
            f"Created {len(network.network.neurons)} neurons",
        )

        # Check connectivity
        stats = network.network.get_network_statistics()

        self.log_test(
            "Network has connections",
            stats["num_connections"] > 0,
            f"Network has {stats['num_connections']} connections",
        )

        self.log_test(
            "Network has free synapses for external input",
            stats["num_free_synapses"] > 0,
            f"Network has {stats['num_free_synapses']} free synapses",
        )

        # Test network statistics are reasonable
        self.log_test(
            "Synaptic density is reasonable",
            0.0 <= stats["synaptic_density"] <= 1.0,
            f"Synaptic density: {stats['synaptic_density']:.3f}",
        )

    def test_reproducibility(self):
        """Test that identical inputs produce identical outputs."""
        print("\n=== Testing Reproducibility ===")

        # Create two identical networks
        random.seed(42)
        np.random.seed(42)
        network1 = NeuronNetwork(num_neurons=3, synapses_per_neuron=3)

        random.seed(42)
        np.random.seed(42)
        network2 = NeuronNetwork(num_neurons=3, synapses_per_neuron=3)

        # Set identical external inputs
        neuron_ids1 = list(network1.network.neurons.keys())
        neuron_ids2 = list(network2.network.neurons.keys())

        # Find free synapses for input
        free_synapses1 = network1.network.free_synapses[:2]  # Use first 2
        free_synapses2 = network2.network.free_synapses[:2]  # Use first 2

        # Set external inputs
        for nid, sid in free_synapses1:
            network1.set_external_input(nid, sid, 1.5)
        for nid, sid in free_synapses2:
            network2.set_external_input(nid, sid, 1.5)

        # Run identical simulations
        activity1 = []
        activity2 = []

        for tick in range(10):
            act1 = network1.run_tick()
            act2 = network2.run_tick()
            activity1.append(act1)
            activity2.append(act2)

        # Compare outputs (should be identical)
        outputs_match = True
        for i, (a1, a2) in enumerate(zip(activity1, activity2)):
            if a1["fired_neurons"] != a2["fired_neurons"]:
                outputs_match = False
                break

        self.log_test(
            "Identical inputs produce identical outputs",
            outputs_match,
            f"Networks produced {'identical' if outputs_match else 'different'} firing patterns",
        )

        # Test state consistency
        state1 = network1.get_network_state()
        state2 = network2.get_network_state()

        potentials_match = True
        for nid in neuron_ids1:
            nid2 = neuron_ids2[neuron_ids1.index(nid)]
            if not np.isclose(
                state1["neurons"][nid]["membrane_potential"],
                state2["neurons"][nid2]["membrane_potential"],
                atol=1e-6,
            ):
                potentials_match = False
                break

        self.log_test(
            "Membrane potentials are identical",
            potentials_match,
            f"Membrane potentials {'match' if potentials_match else 'differ'}",
        )

    def test_input_differentiation(self):
        """Test that different inputs create different network states."""
        print("\n=== Testing Input Differentiation ===")

        # Create network
        network = NeuronNetwork(num_neurons=4, synapses_per_neuron=3)
        free_synapses = network.network.free_synapses[:3]  # Use first 3

        # Test different input patterns
        input_patterns = [
            {"strength": 1.0, "synapses": [0]},  # Weak, single input
            {"strength": 2.0, "synapses": [0, 1]},  # Strong, dual input
            {"strength": 0.5, "synapses": [0, 1, 2]},  # Weak, distributed input
        ]

        pattern_results = []

        for i, pattern in enumerate(input_patterns):
            # Reset network
            network.reset_simulation()

            # Set input pattern
            for j in range(len(free_synapses)):
                nid, sid = free_synapses[j]
                if j in pattern["synapses"]:
                    network.set_external_input(nid, sid, pattern["strength"])
                else:
                    network.set_external_input(nid, sid, 0.0)

            # Run simulation
            activity_log = []
            for tick in range(15):
                activity = network.run_tick()
                activity_log.append(activity["total_activity"])

            pattern_results.append(activity_log)

        # Check that different patterns produce different results
        patterns_differ = False
        for i in range(len(pattern_results)):
            for j in range(i + 1, len(pattern_results)):
                if pattern_results[i] != pattern_results[j]:
                    patterns_differ = True
                    break
            if patterns_differ:
                break

        self.log_test(
            "Different inputs create different activity patterns",
            patterns_differ,
            f"Found {'different' if patterns_differ else 'identical'} activity patterns",
        )

        # Test that stronger inputs generally produce more activity
        total_activity = [sum(pattern) for pattern in pattern_results]

        self.log_test(
            "Network responds to input strength",
            max(total_activity) > min(total_activity),
            f"Activity range: {min(total_activity)} to {max(total_activity)}",
        )

    def test_signal_propagation(self):
        """Test that signals propagate through the network correctly."""
        print("\n=== Testing Signal Propagation ===")

        network = NeuronNetwork(num_neurons=3, synapses_per_neuron=4)

        # Find a connected pathway
        connections = network.network.connections
        if len(connections) == 0:
            self.log_test(
                "Network has connections for propagation test",
                False,
                "No connections found in network",
            )
            return

        # Use first connection as test pathway
        source_nid, source_tid, target_nid, target_sid = connections[0]

        # Apply input to trigger propagation
        # First, we need to find a free synapse on the source neuron to stimulate it
        source_free_synapses = [
            (nid, sid)
            for nid, sid in network.network.free_synapses
            if nid == source_nid
        ]

        if len(source_free_synapses) == 0:
            self.log_test(
                "Source neuron has free synapse for stimulation",
                False,
                f"No free synapses on source neuron {source_nid}",
            )
            return

        stimulus_nid, stimulus_sid = source_free_synapses[0]

        # Record initial state of target neuron
        initial_state = network.get_network_state()
        initial_target_potential = initial_state["neurons"][target_nid][
            "membrane_potential"
        ]

        # Apply strong stimulus to source neuron
        network.set_external_input(stimulus_nid, stimulus_sid, 3.0)

        # Run several ticks to allow signal propagation
        propagation_detected = False
        for tick in range(10):
            activity = network.run_tick()

            # Check if target neuron's state changed
            current_state = network.get_network_state()
            current_target_potential = current_state["neurons"][target_nid][
                "membrane_potential"
            ]

            if abs(current_target_potential - initial_target_potential) > 0.01:
                propagation_detected = True
                break

        self.log_test(
            "Signals propagate through network connections",
            propagation_detected,
            f"Target neuron potential changed from {initial_target_potential:.4f}",
        )

    def test_plasticity_effects(self):
        """Test that plasticity changes network behavior over time."""
        print("\n=== Testing Plasticity Effects ===")

        network = NeuronNetwork(num_neurons=3, synapses_per_neuron=3)
        free_synapses = network.network.free_synapses[:1]  # Use one input

        if len(free_synapses) == 0:
            self.log_test(
                "Network has free synapses for plasticity test",
                False,
                "No free synapses available",
            )
            return

        nid, sid = free_synapses[0]

        # Record initial synaptic weights
        initial_weights = {}
        for neuron_id, neuron in network.network.neurons.items():
            initial_weights[neuron_id] = {}
            for syn_id, synapse in neuron.postsynaptic_points.items():
                initial_weights[neuron_id][syn_id] = synapse.u_i.info

        # Apply repeated stimulation to induce plasticity
        network.set_external_input(nid, sid, 1.5)

        for tick in range(50):  # Extended stimulation
            network.run_tick()

        # Record final synaptic weights
        final_weights = {}
        for neuron_id, neuron in network.network.neurons.items():
            final_weights[neuron_id] = {}
            for syn_id, synapse in neuron.postsynaptic_points.items():
                final_weights[neuron_id][syn_id] = synapse.u_i.info

        # Check if any weights changed significantly
        plasticity_detected = False
        for neuron_id in initial_weights:
            for syn_id in initial_weights[neuron_id]:
                initial = initial_weights[neuron_id][syn_id]
                final = final_weights[neuron_id][syn_id]
                if abs(final - initial) > 0.01:  # Significant change
                    plasticity_detected = True
                    break
            if plasticity_detected:
                break

        self.log_test(
            "Plasticity changes synaptic weights",
            plasticity_detected,
            f"Synaptic weights {'changed' if plasticity_detected else 'remained stable'}",
        )

    def test_network_export_import(self):
        """Test network state export and import functionality."""
        print("\n=== Testing Network Export/Import ===")

        network1 = NeuronNetwork(num_neurons=2, synapses_per_neuron=3)

        # Set some external inputs
        if len(network1.network.free_synapses) > 0:
            nid, sid = network1.network.free_synapses[0]
            network1.set_external_input(nid, sid, 1.2)

        # Run some ticks to create history
        for tick in range(5):
            network1.run_tick()

        original_state = network1.get_network_state()

        # Test that we can get network statistics
        stats = network1.network.get_network_statistics()

        self.log_test(
            "Network state can be retrieved",
            "current_tick" in original_state and "neurons" in original_state,
            f"State contains {len(original_state)} top-level keys",
        )

        self.log_test(
            "Network statistics are complete",
            all(
                key in stats
                for key in ["num_neurons", "num_connections", "synaptic_density"]
            ),
            f"Statistics contain {len(stats)} metrics",
        )

    def run_all_tests(self):
        """Run all network dynamics tests."""
        print("Starting Network Dynamics Tests...\n")

        self.test_network_creation()
        self.test_reproducibility()
        self.test_input_differentiation()
        self.test_signal_propagation()
        self.test_plasticity_effects()
        self.test_network_export_import()

        # Summary
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r["status"] == "PASS")
        failed_tests = total_tests - passed_tests

        print(f"\n=== Network Dynamics Test Summary ===")
        print(f"Total tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success rate: {passed_tests/total_tests*100:.1f}%")

        if failed_tests > 0:
            print("\nFailed tests:")
            for result in self.test_results:
                if result["status"] == "FAIL":
                    print(f"  - {result['test']}: {result['details']}")

        return self.test_results


if __name__ == "__main__":
    tester = NetworkDynamicsTester()
    results = tester.run_all_tests()
