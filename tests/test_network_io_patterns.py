#!/usr/bin/env python3
"""
Test script for network-level I/O patterns and reproducibility.
Verifies that external inputs create reproducible patterns and different inputs
create different internal states and outputs.
"""

import numpy as np
import sys
import os
import random

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from neuron.network import NeuronNetwork


def test_reproducibility():
    """Test that identical inputs produce identical outputs."""
    print("Testing Network Reproducibility...")

    # Create two identical networks with same seed
    random.seed(42)
    np.random.seed(42)
    network1 = NeuronNetwork(num_neurons=3, synapses_per_neuron=2)

    random.seed(42)
    np.random.seed(42)
    network2 = NeuronNetwork(num_neurons=3, synapses_per_neuron=2)

    # Modify neurons to make firing easier
    for network in [network1, network2]:
        for neuron in network.network.neurons.values():
            neuron.params.r_base = 0.2
            neuron.r = 0.2
            neuron.params.lambda_param = 2.0

    # Set identical external inputs
    external_inputs = list(network1.network.external_inputs.keys())[:2]

    for key in external_inputs:
        network1.set_external_input(key[0], key[1], 2.0)
        network2.set_external_input(key[0], key[1], 2.0)

    # Run identical simulations
    activity1 = []
    activity2 = []

    for tick in range(10):
        act1 = network1.run_tick()
        act2 = network2.run_tick()
        activity1.append(act1["total_activity"])
        activity2.append(act2["total_activity"])

    identical = activity1 == activity2
    print(f"Network 1 activity: {activity1}")
    print(f"Network 2 activity: {activity2}")
    print(f"Identical: {identical}")

    return identical


def test_input_differentiation():  # -> tuple[bool, dict[Any, Any]]:# -> tuple[bool, dict[Any, Any]]:# -> tuple[bool, dict[Any, Any]]:
    """Test that different inputs create different network behaviors."""
    print("\nTesting Input Differentiation...")

    # Create network
    network = NeuronNetwork(num_neurons=3, synapses_per_neuron=2)

    # Modify to make firing easier
    for neuron in network.network.neurons.values():
        neuron.params.r_base = 0.3
        neuron.r = 0.3
        neuron.params.lambda_param = 2.0

    external_inputs = list(network.network.external_inputs.keys())[:3]

    # Test different input patterns
    input_patterns = [
        {"name": "No Input", "values": [0.0, 0.0, 0.0]},
        {"name": "Single Strong", "values": [3.0, 0.0, 0.0]},
        {"name": "Dual Medium", "values": [1.5, 1.5, 0.0]},
        {"name": "Triple Weak", "values": [1.0, 1.0, 1.0]},
    ]

    pattern_results = {}

    for pattern in input_patterns:
        # Reset network
        network.reset_simulation()

        # Reapply neuron modifications after reset
        for neuron in network.network.neurons.values():
            neuron.params.r_base = 0.3
            neuron.r = 0.3
            neuron.params.lambda_param = 2.0

        # Set input pattern
        for i, key in enumerate(external_inputs):
            if i < len(pattern["values"]):
                network.set_external_input(key[0], key[1], pattern["values"][i])

        # Run simulation
        activity_log = []
        for tick in range(15):
            activity = network.run_tick()
            activity_log.append(activity["total_activity"])

        pattern_results[pattern["name"]] = {
            "activity": activity_log,
            "total": sum(activity_log),
            "peak": max(activity_log),
            "active_ticks": sum(1 for a in activity_log if a > 0),
        }

    # Display results
    print("\nInput Pattern Results:")
    for name, results in pattern_results.items():
        print(
            f"{name:.<20} Total: {results['total']:>3}, Peak: {results['peak']:>3}, Active: {results['active_ticks']:>2}"
        )

    # Check that different patterns produce different results
    total_activities = [results["total"] for results in pattern_results.values()]
    unique_totals = len(set(total_activities))

    differentiation = unique_totals > 1
    print(
        f"\nDifferentiation detected: {differentiation} ({unique_totals} unique activity levels)"
    )

    return differentiation, pattern_results


def test_signal_propagation():
    """Test that signals propagate through network connections."""
    print("\nTesting Signal Propagation...")

    # Create network with forced connections
    network = NeuronNetwork(num_neurons=3, synapses_per_neuron=2)

    # Modify to make firing easier
    for neuron in network.network.neurons.values():
        neuron.params.r_base = 0.2
        neuron.r = 0.2
        neuron.params.lambda_param = 1.0

    # Get network connections to understand propagation paths
    connections = network.network.connections
    print(f"Network connections: {len(connections)}")

    if len(connections) == 0:
        print("No connections available for propagation test")
        return False

    # Find external inputs for stimulation
    external_inputs = list(network.network.external_inputs.keys())

    if len(external_inputs) == 0:
        print("No external inputs available")
        return False

    # Record initial neuron states
    initial_states = {}
    for nid, neuron in network.network.neurons.items():
        initial_states[nid] = neuron.S

    # Apply strong stimulus
    stimulus_key = external_inputs[0]
    network.set_external_input(stimulus_key[0], stimulus_key[1], 5.0)

    # Run simulation and track changes
    propagation_detected = False
    neuron_changes = {}

    for tick in range(15):
        activity = network.run_tick()

        # Check for changes in neuron states
        for nid, neuron in network.network.neurons.items():
            if nid not in neuron_changes:
                neuron_changes[nid] = []

            change = abs(neuron.S - initial_states[nid])
            neuron_changes[nid].append(change)

            if change > 0.01:  # Significant change
                propagation_detected = True

    print(f"Propagation detected: {propagation_detected}")

    # Show maximum changes per neuron
    for nid, changes in neuron_changes.items():
        max_change = max(changes)
        print(f"Neuron {nid}: max change = {max_change:.4f}")

    return propagation_detected


def test_plasticity_adaptation():
    """Test that network adapts through plasticity over time."""
    print("\nTesting Plasticity Adaptation...")

    network = NeuronNetwork(num_neurons=2, synapses_per_neuron=2)

    # Modify parameters for easier firing and stronger plasticity
    for neuron in network.network.neurons.values():
        neuron.params.r_base = 0.3
        neuron.r = 0.3
        neuron.params.eta_post = 0.05  # Stronger plasticity
        neuron.params.lambda_param = 2.0

    external_inputs = list(network.network.external_inputs.keys())[:1]

    if not external_inputs:
        print("No external inputs available")
        return False

    # Record initial synaptic weights
    initial_weights = {}
    for nid, neuron in network.network.neurons.items():
        initial_weights[nid] = {}
        for sid, synapse in neuron.postsynaptic_points.items():
            initial_weights[nid][sid] = synapse.u_i.info

    # Apply repeated stimulation
    key = external_inputs[0]
    network.set_external_input(key[0], key[1], 2.0)

    # Run extended simulation to allow plasticity
    for tick in range(50):
        network.run_tick()

    # Record final weights
    final_weights = {}
    for nid, neuron in network.network.neurons.items():
        final_weights[nid] = {}
        for sid, synapse in neuron.postsynaptic_points.items():
            final_weights[nid][sid] = synapse.u_i.info

    # Check for significant weight changes
    significant_changes = 0
    total_synapses = 0

    print("Weight changes:")
    for nid in initial_weights:
        for sid in initial_weights[nid]:
            initial = initial_weights[nid][sid]
            final = final_weights[nid][sid]
            change = abs(final - initial)

            print(
                f"  Neuron {nid}, Synapse {sid}: {initial:.4f} -> {final:.4f} (Δ{change:.4f})"
            )

            if change > 0.01:
                significant_changes += 1
            total_synapses += 1

    adaptation_detected = significant_changes > 0
    print(
        f"\nAdaptation detected: {adaptation_detected} ({significant_changes}/{total_synapses} synapses changed)"
    )

    return adaptation_detected


def run_network_io_tests():
    """Run all network I/O pattern tests."""
    print("=" * 60)
    print("NETWORK I/O PATTERNS TEST SUITE")
    print("=" * 60)

    tests = [
        ("Reproducibility", lambda: test_reproducibility()),
        ("Input Differentiation", lambda: test_input_differentiation()[0]),
        ("Signal Propagation", lambda: test_signal_propagation()),
        ("Plasticity Adaptation", lambda: test_plasticity_adaptation()),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, "PASS" if result else "FAIL"))
        except Exception as e:
            results.append((test_name, f"ERROR: {str(e)}"))

    print("\n" + "=" * 60)
    print("NETWORK I/O PATTERNS TEST RESULTS")
    print("=" * 60)

    passed = 0
    for test_name, result in results:
        print(f"{test_name:.<40} {result}")
        if result == "PASS":
            passed += 1

    print(
        f"\nOverall: {passed}/{len(results)} tests passed ({passed/len(results)*100:.1f}%)"
    )

    return results


if __name__ == "__main__":
    run_network_io_tests()
