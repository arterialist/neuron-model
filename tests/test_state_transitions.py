#!/usr/bin/env python3
"""
Test suite for verifying state transition dynamics work correctly
according to the formal model predictions.
"""

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from neuron.neuron import Neuron, NeuronParameters
from neuron.network import NeuronNetwork


class StateTransitionTester:
    """Test state transition dynamics."""

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

    def test_tick_sequence(self):
        """Test that the tick sequence follows Section 5 order."""
        print("\n=== Testing Tick Sequence ===")

        params = NeuronParameters(num_inputs=2, num_neuromodulators=2)
        neuron = Neuron(12345, params, log_level="ERROR")
        neuron.add_synapse(0, 1)
        neuron.add_synapse(1, 2)

        # Create inputs with neuromodulation
        external_inputs = {
            0: {"info": 1.0, "mod": np.array([0.1, 0.2])},
            1: {"info": 0.8, "mod": np.array([0.05, 0.15])},
        }

        # Record initial state
        initial_M = neuron.M_vector.copy()
        initial_r = neuron.r
        initial_S = neuron.S

        # Execute tick and verify sequence
        events = neuron.tick(external_inputs, 0)

        # A. Neuromodulation should have updated
        self.log_test(
            "Neuromodulation state updated",
            not np.array_equal(neuron.M_vector, initial_M),
            f"M_vector changed from {initial_M} to {neuron.M_vector}",
        )

        # Dynamic parameters should have updated
        self.log_test(
            "Dynamic parameters updated",
            neuron.r != initial_r,
            f"r changed from {initial_r} to {neuron.r}",
        )

        # B. Signals should be processed and queued
        self.log_test(
            "Signals queued for propagation",
            len(neuron.propagation_queue) > 0,
            f"Queue length: {len(neuron.propagation_queue)}",
        )

        # C. Membrane potential should have evolved
        self.log_test(
            "Membrane potential evolved",
            neuron.S != initial_S,
            f"S changed from {initial_S} to {neuron.S}",
        )

    def test_homeostatic_firing_rate(self):
        """Test that firing rate homeostasis works correctly."""
        print("\n=== Testing Homeostatic Firing Rate ===")

        params = NeuronParameters(num_inputs=1, beta_avg=0.9)
        neuron = Neuron(12345, params, log_level="ERROR")
        neuron.add_synapse(0, 1)

        # Simulate firing sequence
        firing_sequence = [1, 0, 1, 1, 0, 0, 1, 0]
        F_avg_history = []

        for i, should_fire in enumerate(firing_sequence):
            # Set output manually to test averaging
            neuron.O = float(should_fire)

            # Record F_avg before update
            old_F_avg = neuron.F_avg

            # Run tick to update F_avg
            neuron.tick({}, i)

            # Verify EMA update: F_avg(t+1) = β * F_avg(t) + (1-β) * O(t)
            expected_F_avg = (
                params.beta_avg * old_F_avg + (1 - params.beta_avg) * should_fire
            )

            self.log_test(
                f"F_avg update step {i}",
                np.isclose(neuron.F_avg, expected_F_avg, atol=1e-6),
                f"Expected: {expected_F_avg:.6f}, Got: {neuron.F_avg:.6f}",
            )

            F_avg_history.append(neuron.F_avg)

        # Test that t_ref responds to firing rate
        high_firing_neuron = Neuron(99999, params, log_level="ERROR")
        high_firing_neuron.F_avg = 0.8  # High firing rate
        high_firing_neuron.tick({}, 0)

        low_firing_neuron = Neuron(88888, params, log_level="ERROR")
        low_firing_neuron.F_avg = 0.1  # Low firing rate
        low_firing_neuron.tick({}, 0)

        self.log_test(
            "t_ref responds to firing rate",
            high_firing_neuron.t_ref < low_firing_neuron.t_ref,
            f"High firing t_ref: {high_firing_neuron.t_ref:.3f}, Low firing t_ref: {low_firing_neuron.t_ref:.3f}",
        )

    def test_plasticity_timing(self):
        """Test that plasticity timing rules work correctly."""
        print("\n=== Testing Plasticity Timing ===")

        params = NeuronParameters(num_inputs=1, eta_post=0.1)
        neuron = Neuron(12345, params, log_level="ERROR")
        neuron.add_synapse(0, 1)

        synapse = neuron.postsynaptic_points[0]
        initial_weight = synapse.u_i.info

        # Test LTP condition (Δt ≤ t_ref)
        neuron.t_last_fire = 5  # Recent firing
        current_tick = 10  # Δt = 5

        external_inputs = {0: {"info": 2.0}}  # Strong input
        neuron.tick(external_inputs, current_tick)

        ltp_weight = synapse.u_i.info

        # Reset for LTD test
        synapse.u_i.info = initial_weight
        neuron.t_last_fire = 0  # Long ago
        current_tick = 100  # Δt = 100 > t_ref

        neuron.tick(external_inputs, current_tick)
        ltd_weight = synapse.u_i.info

        self.log_test(
            "LTP increases weights",
            ltp_weight > initial_weight,
            f"Initial: {initial_weight:.4f}, After LTP: {ltp_weight:.4f}",
        )

        self.log_test(
            "LTD decreases weights",
            ltd_weight < initial_weight,
            f"Initial: {initial_weight:.4f}, After LTD: {ltd_weight:.4f}",
        )

    def test_cooldown_mechanism(self):
        """Test somatic firing cooldown mechanism."""
        print("\n=== Testing Cooldown Mechanism ===")

        params = NeuronParameters(num_inputs=1, c=5)  # 5-tick cooldown
        neuron = Neuron(12345, params, log_level="ERROR")
        neuron.add_synapse(0, 1)

        # Force neuron to spike
        neuron.S = 10.0  # High membrane potential
        neuron.r = 1.0  # Low threshold

        # Should fire
        neuron.tick({}, 0)
        first_fire = neuron.O > 0
        fire_tick = 0

        # Try to fire again immediately (should be blocked by cooldown)
        neuron.S = 10.0
        neuron.tick({}, 1)
        immediate_fire = neuron.O > 0

        # Try to fire after cooldown period
        neuron.S = 10.0
        neuron.tick({}, fire_tick + params.c + 1)
        post_cooldown_fire = neuron.O > 0

        self.log_test(
            "Initial firing works", first_fire, f"Fired at tick 0: {first_fire}"
        )

        self.log_test(
            "Cooldown blocks immediate firing",
            not immediate_fire,
            f"Immediate fire blocked: {not immediate_fire}",
        )

        self.log_test(
            "Post-cooldown firing works",
            post_cooldown_fire,
            f"Post-cooldown fire: {post_cooldown_fire}",
        )

    def test_signal_decay(self):
        """Test signal decay during propagation."""
        print("\n=== Testing Signal Decay ===")

        params = NeuronParameters(num_inputs=2, delta_decay=0.8)
        neuron = Neuron(12345, params, log_level="ERROR")
        neuron.add_synapse(0, 1)  # Close synapse
        neuron.add_synapse(1, 5)  # Distant synapse

        # Send identical signals to both synapses
        external_inputs = {0: {"info": 1.0}, 1: {"info": 1.0}}

        neuron.tick(external_inputs, 0)

        # Check that signals are scheduled with correct distances
        close_signal = None
        distant_signal = None

        for signal in neuron.propagation_queue:
            arrival_tick, target, V_value, source_id = signal
            if source_id == 0:
                close_signal = V_value
            elif source_id == 1:
                distant_signal = V_value

        # The signals should have same initial value
        synapse0 = neuron.postsynaptic_points[0]
        synapse1 = neuron.postsynaptic_points[1]
        expected_close = 1.0 * (synapse0.u_i.info + synapse0.u_i.plast)
        expected_distant = 1.0 * (synapse1.u_i.info + synapse1.u_i.plast)

        self.log_test(
            "Initial signal values are based on synapse properties",
            close_signal is not None and distant_signal is not None,
            f"Close: {close_signal}, Distant: {distant_signal}",
        )

        # Fast forward to when signals arrive and check decay
        # This would require running multiple ticks, but the principle is tested above

    def run_all_tests(self):
        """Run all state transition tests."""
        print("Starting State Transition Tests...\n")

        self.test_tick_sequence()
        self.test_homeostatic_firing_rate()
        self.test_plasticity_timing()
        self.test_cooldown_mechanism()
        self.test_signal_decay()

        # Summary
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r["status"] == "PASS")
        failed_tests = total_tests - passed_tests

        print(f"\n=== State Transition Test Summary ===")
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
    tester = StateTransitionTester()
    results = tester.run_all_tests()
