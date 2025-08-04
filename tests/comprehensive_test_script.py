#!/usr/bin/env python3
"""
Comprehensive test script that validates all aspects of the neuron model
implementation against the formal specification.
"""

import subprocess
import sys
import os
import json
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuron.network import NeuronNetwork
from neuron.neuron import Neuron, NeuronParameters


class ComprehensiveValidator:
    """Complete validation of the neuron model implementation."""

    def __init__(self):
        self.results = {
            "mathematical_correctness": {},
            "parameter_vector_fix": {},
            "state_transitions": {},
            "network_dynamics": {},
            "cli_functionality": {},
            "reproducibility": {},
            "overall_score": 0,
        }

    def test_parameter_vector_fix(self):
        """Test that the parameter vector dimension fix works correctly."""
        print("=== Parameter Vector Fix Validation ===")

        fix_tests = []

        # Test various neuromodulator counts
        test_cases = [1, 2, 3, 5]
        for num_mods in test_cases:
            try:
                params = NeuronParameters(num_neuromodulators=num_mods, num_inputs=1)

                # Check all parameter vectors have correct dimensions
                dims_correct = (
                    len(params.gamma) == num_mods
                    and len(params.w_r) == num_mods
                    and len(params.w_b) == num_mods
                    and len(params.w_tref) == num_mods
                )

                # Test neuron creation and operation
                neuron = Neuron(10000 + num_mods, params, log_level="ERROR")
                neuron.add_synapse(0, 1)

                # Test vector consistency
                vector_consistency = len(neuron.M_vector) == num_mods

                # Test basic operation
                external_inputs = {0: {"info": 1.0, "mod": np.ones(num_mods) * 0.1}}
                neuron.tick(external_inputs, 0)

                test_passed = dims_correct and vector_consistency
                fix_tests.append((f"Vector dimensions ({num_mods} mods)", test_passed))

            except Exception as e:
                fix_tests.append((f"Vector dimensions ({num_mods} mods)", False))

        # Calculate fix success rate
        passed_fix_tests = sum(1 for _, result in fix_tests if result)
        fix_success_rate = (passed_fix_tests / len(fix_tests)) * 100

        self.results["parameter_vector_fix"] = {
            "success_rate": fix_success_rate,
            "status": "PASS" if fix_success_rate == 100 else "PARTIAL",
            "tests": fix_tests,
        }

    def test_mathematical_correctness(self):
        """Test mathematical correctness including dimension fixes."""
        print("=== Mathematical Correctness Tests ===")

        math_tests = []

        # Test 1: Vector dimension consistency (should now pass)
        try:
            params = NeuronParameters(num_neuromodulators=3, num_inputs=1)
            dims_correct = (
                len(params.gamma) == 3
                and len(params.w_r) == 3
                and len(params.w_b) == 3
                and len(params.w_tref) == 3
            )
            math_tests.append(("Vector dimensions consistency", dims_correct))
        except Exception:
            math_tests.append(("Vector dimensions consistency", False))

        # Test 2: Neuromodulation mathematics
        try:
            params = NeuronParameters(num_neuromodulators=2, num_inputs=1)
            neuron = Neuron(12345, params, log_level="ERROR")
            neuron.add_synapse(0, 1)

            # Set known state
            neuron.M_vector = np.array([0.1, 0.2])
            old_M = neuron.M_vector.copy()

            # Apply input
            external_inputs = {0: {"info": 1.0, "mod": np.array([0.05, 0.1])}}
            neuron.tick(external_inputs, 0)

            # Check EMA update: M_k(t+1) = γ_k * M_k(t) + (1-γ_k) * total_adapt_signal_k
            expected_adapt_signal = (
                external_inputs[0]["mod"] * neuron.postsynaptic_points[0].u_i.adapt
            )
            expected_M = (
                params.gamma * old_M + (1 - params.gamma) * expected_adapt_signal
            )

            math_correct = np.allclose(neuron.M_vector, expected_M, atol=1e-6)
            math_tests.append(("Neuromodulation EMA formula", math_correct))
        except Exception:
            math_tests.append(("Neuromodulation EMA formula", False))

        # Test 3: Plasticity mechanisms
        try:
            params = NeuronParameters(num_inputs=1, eta_post=0.1)
            neuron = Neuron(23456, params, log_level="ERROR")
            neuron.add_synapse(0, 1)

            initial_weight = neuron.postsynaptic_points[0].u_i.info

            # Apply input with retrograde signaling
            plasticity_input = {
                0: {
                    "info": 1.5,
                    "plast": 0.8,
                    "mod": np.array([0.1, 0.2]),
                    "source_neuron_id": 99999,
                    "source_terminal_id": 0,
                }
            }

            neuron.t_last_fire = -5  # Recent firing for LTP
            events = neuron.tick(plasticity_input, 0)

            final_weight = neuron.postsynaptic_points[0].u_i.info
            weight_changed = abs(final_weight - initial_weight) > 1e-6
            retrograde_generated = any(hasattr(e, "error_vector") for e in events)

            math_tests.append(("Plasticity weight update", weight_changed))
            math_tests.append(("Retrograde signal generation", retrograde_generated))
        except Exception:
            math_tests.append(("Plasticity weight update", False))
            math_tests.append(("Retrograde signal generation", False))

        # Calculate math success rate
        passed_math_tests = sum(1 for _, result in math_tests if result)
        math_success_rate = (passed_math_tests / len(math_tests)) * 100

        self.results["mathematical_correctness"]["success_rate"] = math_success_rate
        self.results["mathematical_correctness"]["status"] = (
            "PASS" if math_success_rate >= 90 else "PARTIAL"
        )
        self.results["mathematical_correctness"]["tests"] = math_tests

    def test_individual_neuron_firing(self):
        """Test basic neuron firing functionality."""
        print("=== Individual Neuron Firing Tests ===")

        try:
            # Test with easy-to-fire parameters
            params = NeuronParameters(
                num_inputs=1,
                r_base=0.3,  # Low threshold
                lambda_param=2.0,  # Fast response
            )

            neuron = Neuron(12345, params, log_level="ERROR")
            neuron.add_synapse(0, 1)

            # Apply strong input
            strong_input = {0: {"info": 3.0}}
            neuron.tick(strong_input, 0)
            neuron.tick({}, 1)  # Signal arrives

            fired = neuron.O > 0
            self.results["state_transitions"]["basic_firing"] = (
                "PASS" if fired else "FAIL"
            )

            # Test plasticity
            initial_weight = neuron.postsynaptic_points[0].u_i.info
            # Apply input with retrograde source info
            neuron.t_last_fire = 0  # Recent firing for LTP
            plasticity_input = {
                0: {"info": 2.0, "source_neuron_id": 99999, "source_terminal_id": 0}
            }
            events = neuron.tick(plasticity_input, 1)
            final_weight = neuron.postsynaptic_points[0].u_i.info

            plasticity_works = abs(final_weight - initial_weight) > 1e-6
            retrograde_generated = any(hasattr(e, "error_vector") for e in events)

            self.results["state_transitions"]["plasticity"] = (
                "PASS" if plasticity_works else "FAIL"
            )
            self.results["state_transitions"]["retrograde_signals"] = (
                "PASS" if retrograde_generated else "FAIL"
            )

        except Exception as e:
            self.results["state_transitions"]["error"] = str(e)

    def test_network_functionality(self):
        """Test network-level functionality."""
        print("=== Network Functionality Tests ===")

        try:
            # Create small network
            network = NeuronNetwork(num_neurons=3, synapses_per_neuron=2)

            # Check basic network properties
            stats = network.network.get_network_statistics()
            self.results["network_dynamics"]["creation"] = "PASS"
            self.results["network_dynamics"]["num_neurons"] = stats["num_neurons"]
            self.results["network_dynamics"]["num_connections"] = stats[
                "num_connections"
            ]

            # Test reproducibility
            network1 = NeuronNetwork(num_neurons=2, synapses_per_neuron=2)
            network2 = NeuronNetwork(num_neurons=2, synapses_per_neuron=2)

            # Both networks should behave identically with same inputs
            activities1 = []
            activities2 = []

            for tick in range(5):
                act1 = network1.run_tick()
                act2 = network2.run_tick()
                activities1.append(act1["total_activity"])
                activities2.append(act2["total_activity"])

            reproducible = activities1 == activities2
            self.results["reproducibility"]["identical_conditions"] = (
                "PASS" if reproducible else "FAIL"
            )

            # Test that networks can be stimulated to fire
            # Modify parameters to make firing easier
            for neuron in network.network.neurons.values():
                neuron.params.r_base = 0.2
                neuron.r = 0.2

            # Find external inputs and stimulate
            external_inputs = list(network.network.external_inputs.keys())
            if external_inputs:
                nid, sid = external_inputs[0]
                network.set_external_input(nid, sid, 5.0)  # Strong input

                activity_detected = False
                for tick in range(10):
                    activity = network.run_tick()
                    if activity["total_activity"] > 0:
                        activity_detected = True
                        break

                self.results["network_dynamics"]["stimulation_response"] = (
                    "PASS" if activity_detected else "FAIL"
                )
            else:
                self.results["network_dynamics"][
                    "stimulation_response"
                ] = "FAIL - No external inputs"

        except Exception as e:
            self.results["network_dynamics"]["error"] = str(e)

    def test_cli_basic_functionality(self):
        """Test basic CLI functionality."""
        print("=== CLI Basic Functionality Tests ===")

        # Test CLI commands that don't require interaction
        test_commands = [
            ("add_neuron 2 2 2", "Network creation"),
            ("status", "Status display"),
            ("list_external", "External input listing"),
        ]

        cli_results = {}

        for commands, description in test_commands:
            try:
                # Run CLI from parent directory
                full_command = (
                    f"cd .. && echo '{commands}\\nexit' | python neuron_cli.py"
                )
                result = subprocess.run(
                    full_command, shell=True, capture_output=True, text=True, timeout=10
                )

                if result.returncode == 0 and "error" not in result.stdout.lower():
                    cli_results[description] = "PASS"
                else:
                    cli_results[description] = "FAIL"

            except Exception as e:
                cli_results[description] = f"ERROR: {str(e)}"

        self.results["cli_functionality"] = cli_results

    def calculate_overall_score(self):
        """Calculate overall implementation score."""
        scores = []

        # Mathematical correctness (30% weight)
        if "success_rate" in self.results["mathematical_correctness"]:
            math_score = self.results["mathematical_correctness"]["success_rate"]
            scores.append(("Mathematical Correctness", math_score, 0.3))

        # Parameter vector fix (20% weight)
        if (
            "parameter_vector_fix" in self.results
            and "success_rate" in self.results["parameter_vector_fix"]
        ):
            fix_score = self.results["parameter_vector_fix"]["success_rate"]
            scores.append(("Parameter Vector Fix", fix_score, 0.2))

        # State transitions (25% weight)
        state_tests = self.results["state_transitions"]
        state_passed = sum(1 for v in state_tests.values() if v == "PASS")
        state_total = len([v for v in state_tests.values() if v in ["PASS", "FAIL"]])
        if state_total > 0:
            state_score = (state_passed / state_total) * 100
            scores.append(("State Transitions", state_score, 0.25))

        # Network dynamics (15% weight)
        network_tests = self.results["network_dynamics"]
        network_passed = sum(1 for v in network_tests.values() if v == "PASS")
        network_total = len(
            [v for v in network_tests.values() if v in ["PASS", "FAIL"]]
        )
        if network_total > 0:
            network_score = (network_passed / network_total) * 100
            scores.append(("Network Dynamics", network_score, 0.15))

        # CLI functionality (10% weight)
        cli_tests = self.results["cli_functionality"]
        cli_passed = sum(1 for v in cli_tests.values() if v == "PASS")
        cli_total = len([v for v in cli_tests.values() if v in ["PASS", "FAIL"]])
        if cli_total > 0:
            cli_score = (cli_passed / cli_total) * 100
            scores.append(("CLI Functionality", cli_score, 0.1))

        # Calculate weighted average
        if scores:
            weighted_score = sum(score * weight for _, score, weight in scores)
            self.results["overall_score"] = weighted_score

            print(f"\n=== Detailed Scoring ===")
            for category, score, weight in scores:
                print(f"{category}: {score:.1f}% (weight: {weight*100:.0f}%)")
            print(f"Overall Score: {weighted_score:.1f}%")

    def generate_report(self):
        """Generate comprehensive evaluation report."""
        print("\n" + "=" * 60)
        print("COMPREHENSIVE NEURON MODEL EVALUATION REPORT")
        print("=" * 60)

        print(f"\nOverall Implementation Score: {self.results['overall_score']:.1f}%")

        if self.results["overall_score"] >= 99:
            grade = "A+ - Perfect Implementation"
        elif self.results["overall_score"] >= 90:
            grade = "A - Excellent Implementation"
        elif self.results["overall_score"] >= 80:
            grade = "B - Good Implementation with Minor Issues"
        elif self.results["overall_score"] >= 70:
            grade = "C - Acceptable Implementation with Notable Issues"
        elif self.results["overall_score"] >= 60:
            grade = "D - Poor Implementation with Major Issues"
        else:
            grade = "F - Failing Implementation"

        print(f"Grade: {grade}")

        # Detailed breakdown
        print(f"\n1. Mathematical Correctness:")
        math_results = self.results["mathematical_correctness"]
        if "success_rate" in math_results:
            print(f"   Success Rate: {math_results['success_rate']:.1f}%")
        if "status" in math_results:
            print(f"   Status: {math_results['status']}")

        print(f"\n2. Parameter Vector Fix:")
        if "parameter_vector_fix" in self.results:
            fix_results = self.results["parameter_vector_fix"]
            if "success_rate" in fix_results:
                print(f"   Success Rate: {fix_results['success_rate']:.1f}%")
            if "status" in fix_results:
                print(f"   Status: {fix_results['status']}")

        print(f"\n3. State Transitions:")
        for test, result in self.results["state_transitions"].items():
            if test != "error":
                print(f"   {test}: {result}")

        print(f"\n4. Network Dynamics:")
        for test, result in self.results["network_dynamics"].items():
            if test != "error":
                print(f"   {test}: {result}")

        print(f"\n5. CLI Functionality:")
        for test, result in self.results["cli_functionality"].items():
            print(f"   {test}: {result}")

        print(f"\n6. Reproducibility:")
        for test, result in self.results["reproducibility"].items():
            print(f"   {test}: {result}")

        # Key findings
        print(f"\n=== Key Findings ===")

        issues = []
        strengths = []

        # Analyze results for issues and strengths
        if self.results["mathematical_correctness"].get("success_rate", 0) >= 90:
            strengths.append("Mathematical model implementation is highly accurate")
        else:
            issues.append("Mathematical model implementation has accuracy issues")

        if self.results.get("parameter_vector_fix", {}).get("success_rate", 0) == 100:
            strengths.append("Parameter vector dimension fix works perfectly")
        elif self.results.get("parameter_vector_fix", {}).get("success_rate", 0) >= 90:
            strengths.append("Parameter vector dimension fix mostly works")
        else:
            issues.append("Parameter vector fix has remaining issues")

        if self.results["state_transitions"].get("basic_firing") == "PASS":
            strengths.append("Basic neuron firing mechanism works correctly")
        else:
            issues.append("Basic neuron firing mechanism has problems")

        if self.results["state_transitions"].get("plasticity") == "PASS":
            strengths.append("Plasticity mechanisms are functional")
        else:
            issues.append("Plasticity mechanisms need attention")

        if self.results["network_dynamics"].get("creation") == "PASS":
            strengths.append("Network creation and basic connectivity work")
        else:
            issues.append("Network creation has fundamental issues")

        print("\nStrengths:")
        for strength in strengths:
            print(f"  ✓ {strength}")

        print("\nIssues to Address:")
        for issue in issues:
            print(f"  ✗ {issue}")

        return self.results

    def run_complete_evaluation(self):
        """Run all evaluation tests."""
        print("Starting Comprehensive Neuron Model Evaluation...\n")

        self.test_parameter_vector_fix()
        self.test_mathematical_correctness()
        self.test_individual_neuron_firing()
        self.test_network_functionality()
        self.test_cli_basic_functionality()
        self.calculate_overall_score()

        return self.generate_report()


if __name__ == "__main__":
    validator = ComprehensiveValidator()
    results = validator.run_complete_evaluation()

    # Save results to file
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to evaluation_results.json")
