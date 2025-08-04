#!/usr/bin/env python3
"""
Debug test to understand why neurons aren't firing in network tests.
"""

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from neuron.neuron import Neuron, NeuronParameters
from neuron.network import NeuronNetwork


def test_single_neuron_firing():
    """Test if a single neuron can fire under controlled conditions."""
    print("=== Testing Single Neuron Firing ===")

    # Create neuron with parameters that should make firing easier
    params = NeuronParameters(
        num_inputs=1,
        r_base=0.5,  # Lower threshold
        b_base=0.6,
        lambda_param=5.0,  # Faster response
        c=3,  # Shorter cooldown
    )

    neuron = Neuron(12345, params, log_level="INFO")
    neuron.add_synapse(0, 1)  # Close to hillock

    print(f"Initial state: S={neuron.S}, r={neuron.r}, threshold={neuron.r}")

    # Apply strong input
    strong_input = {0: {"info": 5.0}}  # Very strong

    print("\nApplying strong input...")
    events = neuron.tick(strong_input, 0)
    print(f"After tick 0: S={neuron.S}, O={neuron.O}")

    # Let signal propagate
    events = neuron.tick({}, 1)
    print(f"After tick 1 (signal arrival): S={neuron.S}, O={neuron.O}")

    # Check if neuron fired
    if neuron.O > 0:
        print("✓ Neuron fired successfully!")
        return True
    else:
        print("✗ Neuron did not fire")

        # Debug information
        synapse = neuron.postsynaptic_points[0]
        print(f"Synapse info: u_info={synapse.u_i.info}, u_plast={synapse.u_i.plast}")
        print(f"Local potential: {synapse.potential}")
        print(f"Propagation queue: {neuron.propagation_queue}")

        return False


def test_network_with_manual_setup():
    """Test network with manual neuron parameter adjustment."""
    print("\n=== Testing Network with Manual Setup ===")

    # Create a small network
    network = NeuronNetwork(num_neurons=2, synapses_per_neuron=2)

    # Get neuron IDs
    neuron_ids = list(network.network.neurons.keys())
    print(f"Created neurons: {neuron_ids}")

    # Modify neuron parameters to make firing easier
    for nid, neuron in network.network.neurons.items():
        neuron.params.r_base = 0.3  # Much lower threshold
        neuron.r = 0.3
        neuron.params.lambda_param = 3.0  # Faster response
        print(
            f"Modified neuron {nid}: r={neuron.r}, lambda={neuron.params.lambda_param}"
        )

    # Find free synapses
    free_synapses = network.network.free_synapses
    print(f"Free synapses: {free_synapses}")

    if len(free_synapses) > 0:
        # Apply very strong input
        nid, sid = free_synapses[0]
        network.set_external_input(nid, sid, 10.0)  # Extremely strong
        print(f"Applied strong input to neuron {nid}, synapse {sid}")

        # Run simulation
        print("\nRunning simulation...")
        for tick in range(5):
            activity = network.run_tick()
            print(
                f"Tick {tick}: fired_neurons={activity['fired_neurons']}, total_activity={activity['total_activity']}"
            )

            # Check individual neuron states
            for nid, neuron in network.network.neurons.items():
                print(f"  Neuron {nid}: S={neuron.S:.4f}, O={neuron.O}")

        return (
            len(
                [
                    a
                    for a in [network.run_tick() for _ in range(5)]
                    if a["total_activity"] > 0
                ]
            )
            > 0
        )
    else:
        print("No free synapses available for testing")
        return False


def test_threshold_and_input_scaling():
    """Test different threshold and input combinations."""
    print("\n=== Testing Threshold and Input Scaling ===")

    # Test matrix of thresholds and inputs
    thresholds = [0.1, 0.5, 1.0, 2.0]
    inputs = [0.5, 1.0, 2.0, 5.0, 10.0]

    firing_matrix = []

    for threshold in thresholds:
        row = []
        for input_strength in inputs:
            params = NeuronParameters(
                num_inputs=1, r_base=threshold, lambda_param=1.0  # Fast response
            )

            neuron = Neuron(99999, params, log_level="ERROR")
            neuron.add_synapse(0, 1)

            # Apply input
            strong_input = {0: {"info": input_strength}}
            neuron.tick(strong_input, 0)
            neuron.tick({}, 1)  # Let signal arrive

            fired = neuron.O > 0
            row.append(fired)

        firing_matrix.append(row)

    # Print results
    print("\nFiring Matrix (threshold vs input strength):")
    print("Threshold\\Input", end="")
    for inp in inputs:
        print(f"\t{inp:.1f}", end="")
    print()

    for i, threshold in enumerate(thresholds):
        print(f"{threshold:.1f}", end="")
        for j, fired in enumerate(firing_matrix[i]):
            print(f"\t{'✓' if fired else '✗'}", end="")
        print()

    # Check if any combinations work
    any_firing = any(any(row) for row in firing_matrix)
    print(f"\nAny firing detected: {any_firing}")

    return any_firing


def main():
    """Run all debug tests."""
    print("Starting Firing Debug Tests...\n")

    test1_result = test_single_neuron_firing()
    test2_result = test_network_with_manual_setup()
    test3_result = test_threshold_and_input_scaling()

    print(f"\n=== Debug Test Summary ===")
    print(f"Single neuron firing: {'PASS' if test1_result else 'FAIL'}")
    print(f"Network firing: {'PASS' if test2_result else 'FAIL'}")
    print(f"Threshold scaling: {'PASS' if test3_result else 'FAIL'}")

    if not any([test1_result, test2_result, test3_result]):
        print("\n⚠️  WARNING: No firing detected in any test!")
        print("This suggests a fundamental issue with the firing mechanism.")
    else:
        print(
            "\n✓ Some firing was detected - the mechanism works under some conditions."
        )


if __name__ == "__main__":
    main()
