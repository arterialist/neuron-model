"""Cell-level tests for the experimental rate receptor; no behavioural claims."""
import unittest
import numpy as np

from neuron.neuron import Neuron, NeuronParameters, RetrogradeSignalEvent
from neuron.extensions.experimental.plasticity_rate import PlasticityRateNeuron


def cell(cls=PlasticityRateNeuron, boost=0., params=None):
    np.random.seed(11)
    params = params or NeuronParameters(num_inputs=3, c=2, eta_post=1e-5, eta_retro=1e-5,
        gamma=np.array([.9, .9]), w_r=np.zeros(2), w_b=np.zeros(2), w_tref=np.zeros(2))
    n = cls(1, params, metadata={"plasticity_rate_boost": boost}, log_level="CRITICAL")
    for i in range(3):
        n.add_synapse(i, 1)
        n.postsynaptic_points[i].u_i.info = 2.
        n.postsynaptic_points[i].u_i.plast = 0.
        n.postsynaptic_points[i].u_i.adapt = np.array([1., 0.])
    n.add_axon_terminal(900, 1)
    n.t_last_fire = 0
    return n


class PlasticityRateTests(unittest.TestCase):
    def test_disabled_extension_and_enabled_without_modulator_match_base(self):
        for boost in (0., 999.):
            n, base = cell(boost=boost), cell(cls=Neuron)
            for t in range(1, 50):
                n.input_buffer[0, 0] = base.input_buffer[0, 0] = 1.
                self.assertEqual(n.tick({}, t), base.tick({}, t))
                self.assertEqual((n.S, n.O, n.t_ref), (base.S, base.O, base.t_ref))
                self.assertEqual(n.postsynaptic_points[0].u_i.info, base.postsynaptic_points[0].u_i.info)

    def test_postsynaptic_rate_uses_previous_local_concentration(self):
        n = cell(boost=9.)
        n.M_vector[0] = .1
        n.input_buffer[0, 0] = 1.
        n.tick({}, 1)
        self.assertAlmostEqual(n.last_tick_rate_multiplier, 5.5)
        self.assertAlmostEqual(n.postsynaptic_points[0].u_i.info, 2.+1.56*1e-5*5.5)
        self.assertEqual(n.params.eta_post, 1e-5)
        self.assertAlmostEqual(n.M_vector[0], .09)

    def test_retrograde_rate_is_gated_at_the_sender(self):
        n = cell(boost=9.)
        n.M_vector[0] = .1
        before = n.presynaptic_points[900].u_o.info
        n.process_retrograde_signal(RetrogradeSignalEvent(2, 0, 1, 900, np.array([.2, 0., 0., 0.]), 1))
        self.assertAlmostEqual(n.presynaptic_points[900].u_o.info-before, 1e-5*5.5*.2)
        self.assertEqual(n.params.eta_retro, 1e-5)

    def test_modulator_arrival_has_one_tick_latency_and_decays(self):
        n = cell(boost=9.)
        n.input_buffer[1, 2] = .5
        n.tick({}, 1)
        self.assertEqual(n.last_tick_rate_multiplier, 1.)
        self.assertGreater(n.M_vector[0], 0.)
        n.tick({}, 2)
        self.assertGreater(n.last_tick_rate_multiplier, 1.)
        for t in range(3, 300):
            n.tick({}, t)
        self.assertGreaterEqual(n.rate_multiplier(), 1.)
        self.assertLess(n.rate_multiplier(), 1.+1e-10)
        self.assertGreater(n.params.eta_post, 0.)

    def test_shared_parameters_and_exception_restoration(self):
        params = NeuronParameters(num_inputs=3, eta_post=1e-5, eta_retro=1e-5)
        n = cell(boost=9., params=params)
        self.assertIsNot(n.params, params)
        n.M_vector[0] = .1
        n.params.num_inputs = 4
        with self.assertRaises(AssertionError):
            n.tick({}, 1)
        self.assertEqual(n.params.eta_post, 1e-5)
        self.assertEqual(params.num_inputs, 3)
        self.assertEqual(params.eta_post, 1e-5)

    def test_enabled_gate_rejects_frozen_baseline(self):
        for field in ("eta_post", "eta_retro"):
            p = NeuronParameters(num_inputs=3)
            setattr(p, field, 0.)
            with self.assertRaises(ValueError):
                cell(boost=9., params=p)


if __name__ == "__main__":
    unittest.main()
