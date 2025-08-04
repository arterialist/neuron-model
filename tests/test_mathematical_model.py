#!/usr/bin/env python3
"""
Mathematical model test to verify the parameter vector fix resolves the dimension issue.
"""

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from neuron.neuron import (
    Neuron,
    NeuronParameters,
    PostsynapticInputVector,
    PresynapticOutputVector,
    PresynapticPoint,
)


class ModelValidationTester:
    """Test suite for validating implementation against formal model."""

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

    def test_vector_dimensions_fixed(self):
        """Test that vector dimensions are now consistent throughout (FIXED VERSION)."""
        print("\n=== Testing Vector Dimensions (Fixed Version) ===")

        # Test parameter consistency with various neuromodulator counts
        test_cases = [1, 2, 3, 5, 7]

        for num_mods in test_cases:
            params = NeuronParameters(num_neuromodulators=num_mods)

            # All parameter vectors should now match num_neuromodulators
            dims_correct = (
                len(params.gamma) == num_mods
                and len(params.w_r) == num_mods
                and len(params.w_b) == num_mods
                and len(params.w_tref) == num_mods
            )

            self.log_test(
                f"Parameter vector dimensions ({num_mods} mods)",
                dims_correct,
                f"gamma:{len(params.gamma)}, w_r:{len(params.w_r)}, w_b:{len(params.w_b)}, w_tref:{len(params.w_tref)} (expected:{num_mods})",
            )

            # Test neuron initialization with custom dimensions
            neuron = Neuron(12345 + num_mods, params, log_level="ERROR")

            self.log_test(
                f"Neuron M_vector dimension ({num_mods} mods)",
                len(neuron.M_vector) == num_mods,
                f"M_vector length: {len(neuron.M_vector)}, expected: {num_mods}",
            )

            # Test synapse vectors
            neuron.add_synapse(0, 2)
            synapse = neuron.postsynaptic_points[0]
            self.log_test(
                f"Postsynaptic adapt vector dimension ({num_mods} mods)",
                len(synapse.u_i.adapt) == num_mods,
                f"adapt vector length: {len(synapse.u_i.adapt)}, expected: {num_mods}",
            )

            # Test presynaptic vectors
            neuron.add_axon_terminal(0, 2)
            terminal = neuron.presynaptic_points[0]
            self.log_test(
                f"Presynaptic mod vector dimension ({num_mods} mods)",
                len(terminal.u_o.mod) == num_mods,
                f"mod vector length: {len(terminal.u_o.mod)}, expected: {num_mods}",
            )

    def test_neuromodulation_math_with_custom_dimensions(self):
        """Test neuromodulation math with various vector dimensions."""
        print("\n=== Testing Neuromodulation Math with Custom Dimensions ===")

        # Test with 3 neuromodulators instead of default 2
        params = NeuronParameters(num_neuromodulators=3, num_inputs=2)
        neuron = Neuron(12345, params, log_level="ERROR")

        # Add synapses
        neuron.add_synapse(0, 2)
        neuron.add_synapse(1, 3)

        # Test initial state
        self.log_test(
            "Initial M_vector is zero (3D)",
            np.allclose(neuron.M_vector, np.zeros(3)),
            f"M_vector: {neuron.M_vector}",
        )

        # Create external inputs with 3D modulation
        external_inputs = {
            0: {"info": 1.0, "mod": np.array([0.1, 0.2, 0.3])},
            1: {"info": 0.5, "mod": np.array([0.3, 0.1, 0.2])},
        }

        # Record state before tick
        old_M = neuron.M_vector.copy()
        old_r = neuron.r
        old_b = neuron.b

        # Execute tick
        neuron.tick(external_inputs, 0)

        # Verify neuromodulatory state update with 3D vectors
        expected_adapt_signal = (
            external_inputs[0]["mod"] * neuron.postsynaptic_points[0].u_i.adapt
            + external_inputs[1]["mod"] * neuron.postsynaptic_points[1].u_i.adapt
        )
        expected_M = params.gamma * old_M + (1 - params.gamma) * expected_adapt_signal

        self.log_test(
            "Neuromodulatory state update follows EMA rule (3D)",
            np.allclose(neuron.M_vector, expected_M, atol=1e-6),
            f"Expected: {expected_M}, Got: {neuron.M_vector}",
        )

        # Verify dynamic parameter updates with 3D vectors
        expected_r = params.r_base + np.dot(params.w_r, neuron.M_vector)
        expected_b = params.b_base + np.dot(params.w_b, neuron.M_vector)

        self.log_test(
            "Dynamic threshold r follows formula (3D)",
            np.isclose(neuron.r, expected_r, atol=1e-6),
            f"Expected r: {expected_r}, Got: {neuron.r}",
        )

        self.log_test(
            "Dynamic threshold b follows formula (3D)",
            np.isclose(neuron.b, expected_b, atol=1e-6),
            f"Expected b: {expected_b}, Got: {neuron.b}",
        )

    def test_edge_cases(self):
        """Test edge cases like single neuromodulator."""
        print("\n=== Testing Edge Cases ===")

        # Test with 1 neuromodulator
        params_1 = NeuronParameters(num_neuromodulators=1, num_inputs=1)
        neuron_1 = Neuron(99999, params_1, log_level="ERROR")
        neuron_1.add_synapse(0, 1)

        # Test that it works
        external_inputs = {0: {"info": 1.0, "mod": np.array([0.5])}}

        try:
            neuron_1.tick(external_inputs, 0)
            self.log_test(
                "Single neuromodulator case works",
                True,
                f"M_vector: {neuron_1.M_vector}, r: {neuron_1.r:.4f}",
            )
        except Exception as e:
            self.log_test("Single neuromodulator case works", False, f"Error: {str(e)}")

        # Test with large number of neuromodulators
        params_10 = NeuronParameters(num_neuromodulators=10, num_inputs=1)
        neuron_10 = Neuron(88888, params_10, log_level="ERROR")
        neuron_10.add_synapse(0, 1)

        external_inputs_10 = {0: {"info": 1.0, "mod": np.ones(10) * 0.1}}

        try:
            neuron_10.tick(external_inputs_10, 0)
            self.log_test(
                "Large neuromodulator count (10) works",
                True,
                f"Vector lengths correct: M={len(neuron_10.M_vector)}, gamma={len(params_10.gamma)}",
            )
        except Exception as e:
            self.log_test(
                "Large neuromodulator count (10) works", False, f"Error: {str(e)}"
            )

    def run_tests(self):
        """Run the validation tests."""
        print("Starting Mathematical Model Validation Tests...\n")

        self.test_vector_dimensions_fixed()
        self.test_neuromodulation_math_with_custom_dimensions()
        self.test_edge_cases()

        # Summary
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r["status"] == "PASS")
        failed_tests = total_tests - passed_tests

        print(f"\n=== Updated Test Summary ===")
        print(f"Total tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success rate: {passed_tests/total_tests*100:.1f}%")

        if failed_tests > 0:
            print("\nFailed tests:")
            for result in self.test_results:
                if result["status"] == "FAIL":
                    print(f"  - {result['test']}: {result['details']}")
        else:
            print(
                "\n🎉 ALL TESTS PASSED! The parameter vector fix completely resolves the dimension issues."
            )

        return self.test_results


if __name__ == "__main__":
    tester = ModelValidationTester()
    results = tester.run_tests()
