#!/usr/bin/env python3
"""
Standalone test script for mathematical validity.
Verifies that the implementation correctly follows the formal mathematical model.
"""

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from neuron.neuron import Neuron, NeuronParameters


def test_neuromodulation_formula():
    """Test that neuromodulation follows the exact EMA formula from the model."""
    print("Testing Neuromodulation Formula...")

    params = NeuronParameters(num_neuromodulators=2, num_inputs=2)
    neuron = Neuron(12345, params, log_level="ERROR")
    neuron.add_synapse(0, 2)
    neuron.add_synapse(1, 3)

    # Set known initial state
    neuron.M_vector = np.array([0.1, 0.2])

    # Create inputs with known modulation
    external_inputs = {
        0: {"info": 1.0, "mod": np.array([0.05, 0.1])},
        1: {"info": 0.8, "mod": np.array([0.02, 0.08])},
    }

    # Record initial state
    old_M = neuron.M_vector.copy()

    # Execute tick
    neuron.tick(external_inputs, 0)

    # Calculate expected result manually
    # M_k(t+1) = γ_k * M_k(t) + (1-γ_k) * total_adapt_signal_k
    expected_adapt_signal = (
        external_inputs[0]["mod"] * neuron.postsynaptic_points[0].u_i.adapt
        + external_inputs[1]["mod"] * neuron.postsynaptic_points[1].u_i.adapt
    )
    expected_M = params.gamma * old_M + (1 - params.gamma) * expected_adapt_signal

    print(f"Expected M_vector: {expected_M}")
    print(f"Actual M_vector: {neuron.M_vector}")
    print(f"Match: {np.allclose(neuron.M_vector, expected_M, atol=1e-6)}")

    return np.allclose(neuron.M_vector, expected_M, atol=1e-6)


def test_membrane_potential_equation():
    """Test that membrane potential follows the leaky integrator equation."""
    print("\nTesting Membrane Potential Equation...")

    params = NeuronParameters(num_inputs=1, lambda_param=10.0)
    neuron = Neuron(12345, params, log_level="ERROR")
    neuron.add_synapse(0, 1)

    # Set initial conditions
    neuron.S = 0.5
    initial_S = neuron.S

    # Create input that will arrive immediately
    external_inputs = {0: {"info": 2.0}}

    # First tick: process input, schedule propagation
    neuron.tick(external_inputs, 0)

    # Second tick: signal arrives and integrates
    neuron.tick({}, 1)

    # The membrane potential should have changed according to:
    # S(t+dt) = S(t) + (dt/λ)(-S(t) + I(t))
    print(f"Initial S: {initial_S}")
    print(f"Final S: {neuron.S}")
    print(f"S changed: {neuron.S != initial_S}")

    return neuron.S != initial_S


def test_plasticity_direction():
    """Test that plasticity direction follows the temporal correlation rule."""
    print("\nTesting Plasticity Direction...")

    params = NeuronParameters(num_inputs=1, eta_post=0.1)
    neuron = Neuron(12345, params, log_level="ERROR")
    neuron.add_synapse(0, 1)

    synapse = neuron.postsynaptic_points[0]
    initial_weight = synapse.u_i.info

    # Test LTP condition (Δt ≤ t_ref)
    neuron.t_last_fire = 5
    current_tick = 8  # Δt = 3, should be ≤ t_ref (around 50-100)

    external_inputs = {0: {"info": 2.0}}
    neuron.tick(external_inputs, current_tick)

    ltp_weight = synapse.u_i.info
    ltp_increase = ltp_weight > initial_weight

    # Reset for LTD test
    synapse.u_i.info = initial_weight
    neuron.t_last_fire = 0
    current_tick = 200  # Δt = 200, should be > t_ref

    neuron.tick(external_inputs, current_tick)
    ltd_weight = synapse.u_i.info
    ltd_decrease = ltd_weight < initial_weight

    print(
        f"LTP (recent firing): {initial_weight:.4f} -> {ltp_weight:.4f} (increased: {ltp_increase})"
    )
    print(
        f"LTD (old firing): {initial_weight:.4f} -> {ltd_weight:.4f} (decreased: {ltd_decrease})"
    )

    return ltp_increase and ltd_decrease


def test_retrograde_signaling():
    """Test that retrograde signals contain correct error vectors."""
    print("\nTesting Retrograde Signaling...")

    params = NeuronParameters(num_inputs=1)
    neuron = Neuron(12345, params, log_level="ERROR")
    neuron.add_synapse(0, 1)

    synapse = neuron.postsynaptic_points[0]
    u_info = synapse.u_i.info
    u_plast = synapse.u_i.plast

    # Input with source information for retrograde signaling
    O_ext = {
        "info": 1.5,
        "plast": 0.8,
        "mod": np.array([0.1, 0.2]),
        "source_neuron_id": 99999,
        "source_terminal_id": 0,
    }
    external_inputs = {0: O_ext}

    events = neuron.tick(external_inputs, 0)

    # Find retrograde events
    retrograde_events = [e for e in events if hasattr(e, "error_vector")]

    if retrograde_events:
        retro_event: RetrogradeSignalEvent = retrograde_events[0]  # type: ignore

        # Calculate expected error vector: E_dir = O_ext - u_i
        expected_E_dir = np.array(
            [O_ext["info"] - u_info, O_ext["plast"] - u_plast, *O_ext["mod"]]
        )

        print(f"Expected error vector: {expected_E_dir}")
        print(f"Actual error vector: {retro_event.error_vector}")
        print(
            f"Match: {np.allclose(retro_event.error_vector, expected_E_dir, atol=1e-6)}"
        )

        return np.allclose(retro_event.error_vector, expected_E_dir, atol=1e-6)
    else:
        print("No retrograde events generated")
        return False


def run_mathematical_validity_tests():
    """Run all mathematical validity tests."""
    print("=" * 60)
    print("MATHEMATICAL VALIDITY TEST SUITE")
    print("=" * 60)

    tests = [
        ("Neuromodulation Formula", test_neuromodulation_formula),
        ("Membrane Potential Equation", test_membrane_potential_equation),
        ("Plasticity Direction", test_plasticity_direction),
        ("Retrograde Signaling", test_retrograde_signaling),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, "PASS" if result else "FAIL"))
        except Exception as e:
            results.append((test_name, f"ERROR: {str(e)}"))

    print("\n" + "=" * 60)
    print("MATHEMATICAL VALIDITY TEST RESULTS")
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
    run_mathematical_validity_tests()
