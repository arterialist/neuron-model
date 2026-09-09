import copy
import math
import unittest

import numpy as np

from neuron.neuron import NeuronParameters, RetrogradeSignalEvent
from neuron.extensions.experimental.bounded_plasticity import BoundedPlasticityNeuron
from neuron.extensions.experimental.eligibility_trace import EligibilityTraceNeuron, eligibility_step


def cell(cls=EligibilityTraceNeuron, enabled=True):
    np.random.seed(11)
    params = NeuronParameters(num_inputs=3, c=3, lambda_param=1., r_base=.65, b_base=.9,
                              eta_post=1e-5, eta_retro=1e-7, w_r=np.zeros(2), w_b=np.zeros(2), w_tref=np.zeros(2))
    n = cls(1, params, log_level="CRITICAL", metadata={"bounded_plasticity": True,
                "eligibility_ports": [1, 2] if enabled else [], "plasticity_rate_boost": 499.})
    for i, w in enumerate((2., .1, .1)):
        n.add_synapse(i, 1)
        n.postsynaptic_points[i].u_i.info = w
        n.postsynaptic_points[i].u_i.plast = 0.
        n.postsynaptic_points[i].u_i.adapt[:] = 0.
        n.register_source(i, 2, 900)
    return n


class EligibilityTraceTests(unittest.TestCase):
    def test_disabled_matches_inherited_state_events_and_weights(self):
        a, b = cell(enabled=False), cell(cls=BoundedPlasticityNeuron, enabled=False)
        for t in range(64):
            for n in (a, b):
                n.input_buffer[:, 0] = (1. if t % 9 == 0 else 0., .5, .2)
            ea, eb = a.tick({}, t), b.tick({}, t)
            self.assertEqual((a.S, a.O, a.t_ref), (b.S, b.O, b.t_ref))
            self.assertEqual(a.propagation_queue, b.propagation_queue)
            np.testing.assert_array_equal(a.M_vector, b.M_vector)
            np.testing.assert_array_equal([p.u_i.info for p in a.postsynaptic_points.values()], [p.u_i.info for p in b.postsynaptic_points.values()])
            self.assertEqual(len(ea), len(eb))
            for x, y in zip(ea, eb):
                if isinstance(x, RetrogradeSignalEvent):
                    np.testing.assert_array_equal(x.error_vector, y.error_vector)
                else:
                    self.assertEqual(x, y)

    def test_delayed_somatic_spike_credits_earlier_arrival(self):
        n = cell()
        n.input_buffer[1, 0] = 1.
        n.tick({}, 0)
        self.assertEqual(n.postsynaptic_points[1].u_i.info, .1)
        n.input_buffer[0, 0] = 1.
        n.tick({}, 1)
        n.tick({}, 2)
        self.assertGreater(n.O, 0)
        expected = 1.-.9*math.exp(-1e-5*math.exp(-2/4))
        self.assertAlmostEqual(n.postsynaptic_points[1].u_i.info, expected, places=15)
        self.assertEqual(n.postsynaptic_points[2].u_i.info, .1)
        n.input_buffer[2, 0] = 1.
        n.tick({}, 3)
        self.assertLess(n.postsynaptic_points[2].u_i.info, .1)
        self.assertGreater(n.postsynaptic_points[1].u_i.info, .1)

    def test_flow_is_bounded_and_composes(self):
        q = np.array([0., .1, .9, 1.])
        for plus, minus in ((np.zeros(4), np.zeros(4)), (np.ones(4), np.zeros(4)), (np.ones(4), np.ones(4)*2)):
            a = eligibility_step(q, plus, minus, .002, 1.)
            np.testing.assert_allclose(eligibility_step(a, plus, minus, .003, 1.), eligibility_step(q, plus, minus, .005, 1.), rtol=1e-14)
            z = eligibility_step(q, plus, minus, 100., 1.)
            self.assertTrue(((z >= 0)&(z <= 1)).all())

    def test_complete_clone_keeps_eligibility_state(self):
        a = cell()
        a.input_buffer[1, 0] = 1.
        a.tick({}, 0)
        b = copy.deepcopy(a)
        for n in (a, b):
            n.input_buffer[0, 0] = 1.
            n.tick({}, 1)
            n.tick({}, 2)
        np.testing.assert_array_equal(a.eligibility_pre, b.eligibility_pre)
        self.assertEqual(a.postsynaptic_points[1].u_i.info, b.postsynaptic_points[1].u_i.info)
        self.assertIsNot(a.eligibility_pre, b.eligibility_pre)

    def test_local_rate_boost_preserves_positive_baseline(self):
        a, b = cell(), cell()
        b.M_vector[0] = .1
        for t in range(3):
            for n in (a, b):
                if t == 0: n.input_buffer[1, 0] = 1.
                if t == 1: n.input_buffer[0, 0] = 1.
                n.tick({}, t)
        self.assertGreater(b.postsynaptic_points[1].u_i.info, a.postsynaptic_points[1].u_i.info)
        self.assertEqual(b.params.eta_post, 1e-5)
        self.assertGreater(b.params.eta_retro, 0)


if __name__ == '__main__':
    unittest.main()
