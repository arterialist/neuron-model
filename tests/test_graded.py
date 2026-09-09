"""Cell-level contracts; no claim about embodied behavioural acceptance."""
import copy
import unittest

import numpy as np

from neuron.neuron import Neuron, NeuronParameters, PresynapticPoint, PresynapticOutputVector
from neuron.extensions.graded import GradedNeuron


def cell(cls=GradedNeuron, gain=0.5, **kwargs):
    n = cls(1, NeuronParameters(num_inputs=1, lambda_param=4, r_base=0.1,
                               b_base=0.2, eta_post=0, eta_retro=0),
            metadata={"graded_gain": gain}, log_level="CRITICAL", **kwargs)
    n.add_synapse(0, 0)
    n.postsynaptic_points[0].u_i.info = 1
    n.postsynaptic_points[0].u_i.plast = 0
    n.presynaptic_points[1] = PresynapticPoint(PresynapticOutputVector(info=2, mod=np.zeros(2)))
    return n


class GradedContracts(unittest.TestCase):
    def test_ordinary_threshold_does_not_reset_graded_membrane(self):
        n = cell()
        for t in range(1, 25):
            n.input_buffer[0, 0] = 1
            events = n.tick({}, t)
            expected = 1 - 0.75**t
            # The inherited input buffer/current accumulation uses float32.
            self.assertAlmostEqual(n.S, expected, places=6)
            self.assertAlmostEqual(n.O, 0.5 * expected, places=6)
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0][:2], (1, 1))
            self.assertAlmostEqual(events[0][2], expected, places=6)
            self.assertEqual(n.t_last_fire, -np.inf)
        self.assertEqual(n.params.r_base, 0.1)
        self.assertEqual(n._ablation, set())

    def test_zero_gain_is_base_neuron(self):
        n = cell(gain=0)
        base = cell(cls=Neuron, gain=0)
        # Remove constructor randomness from the comparison.
        base.postsynaptic_points = copy.deepcopy(n.postsynaptic_points)
        base.presynaptic_points = copy.deepcopy(n.presynaptic_points)
        for t in range(40):
            n.input_buffer[0, 0] = base.input_buffer[0, 0] = 1
            self.assertEqual(n.tick({}, t), base.tick({}, t))
            self.assertEqual((n.S, n.O, n.t_ref, n.t_last_fire),
                             (base.S, base.O, base.t_ref, base.t_last_fire))

    def test_preserves_existing_ablation(self):
        n = cell(ablation="thresholds_frozen,tref_frozen")
        n.input_buffer[0, 0] = 1
        n.tick({}, 1)
        self.assertEqual(n._ablation, {"thresholds_frozen", "tref_frozen"})


if __name__ == "__main__":
    unittest.main()
