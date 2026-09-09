"""Mechanism tests, not evidence of associative learning or consciousness."""
import math
import unittest

import numpy as np

from neuron.neuron import NeuronParameters, RetrogradeSignalEvent
from neuron.extensions.experimental.plasticity_rate import PlasticityRateNeuron
from neuron.extensions.experimental.bounded_plasticity import BoundedPlasticityNeuron, magnitude_step


def cell(cls=BoundedPlasticityNeuron, enabled=True, boost=499., eta=1e-5):
    np.random.seed(11)
    p = NeuronParameters(num_inputs=3, c=2, eta_post=eta, eta_retro=1e-7,
        gamma=np.array([.9, .9]), w_r=np.zeros(2), w_b=np.zeros(2), w_tref=np.zeros(2))
    n = cls(1, p, metadata={"bounded_plasticity": enabled, "plasticity_rate_boost": boost}, log_level="CRITICAL")
    for sid, weight in enumerate((2., -.8, 0.)):
        n.add_synapse(sid, 1)
        n.postsynaptic_points[sid].u_i.info = weight
        n.postsynaptic_points[sid].u_i.plast = 0.
        n.postsynaptic_points[sid].u_i.adapt = np.array([1., 0.])
        n.register_source(sid, 2, 900)
    n.add_axon_terminal(900, 1)
    n.t_last_fire = 0
    return n


class BoundedPlasticityTests(unittest.TestCase):
    def test_default_preserves_rate_neuron_tick_and_events(self):
        n, old = cell(enabled=False), cell(cls=PlasticityRateNeuron, enabled=False)
        for t in range(1, 64):
            for x in (n, old):
                x.input_buffer[:, 0] = (1., .5, 0.)
                x.input_buffer[2, 2] = .2 if t < 20 else 0.
            a, b = n.tick({}, t), old.tick({}, t)
            self.assertEqual((n.S, n.O, n.t_ref, n.last_tick_rate_multiplier),
                             (old.S, old.O, old.t_ref, old.last_tick_rate_multiplier))
            np.testing.assert_array_equal(n.M_vector, old.M_vector)
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                if isinstance(x, RetrogradeSignalEvent):
                    np.testing.assert_array_equal(x.error_vector, y.error_vector)
                else:
                    self.assertEqual(x, y)
            self.assertEqual([s.u_i.info for s in n.postsynaptic_points.values()],
                             [s.u_i.info for s in old.postsynaptic_points.values()])

    def test_flow_stays_in_interval_with_both_credit_signs(self):
        for q in (1e-8, .8, 9., 10.):
            for error in (0., .019999999, .02, .020000001, 1., 101., 1e5):
                for eta in (1e-7, .005, 1., 100.):
                    for direction in (-1, 1):
                        new = magnitude_step(q, error, direction, eta)
                        self.assertTrue(math.isfinite(new))
                        self.assertGreaterEqual(new, 0.)
                        self.assertLessEqual(new, 10.+1e-12)
        self.assertEqual(magnitude_step(0., 10., 1, .1), 0.)

    def test_small_step_matches_declared_local_ode(self):
        for direction in (-1, 1):
            q, error, eta = .8, 1.8, 1e-7
            derivative = q*(error*(1-q/10)-.02) if direction > 0 else -q*(error+.02)
            self.assertAlmostEqual((magnitude_step(q, error, direction, eta)-q)/eta, derivative, places=5)
        # Autonomous held-coefficient flow composes in adaptation time.
        first = magnitude_step(.8, 2., 1, .003)
        self.assertAlmostEqual(magnitude_step(first, 2., 1, .002), magnitude_step(.8, 2., 1, .005), places=14)

    def test_rates_remain_positive_effective_and_neurally_modulated(self):
        basal, boosted = cell(boost=0.), cell()
        before = basal.postsynaptic_points[1].u_i.info
        for x in (basal, boosted):
            x.M_vector[0] = .1
            x.input_buffer[1, 0] = 1.
            x.tick({}, 1)
            self.assertLess(x.postsynaptic_points[1].u_i.info, before)
            self.assertEqual(x.params.eta_post, 1e-5)
            self.assertGreater(x.params.eta_retro, 0.)
        self.assertLess(boosted.postsynaptic_points[1].u_i.info, basal.postsynaptic_points[1].u_i.info)
        self.assertAlmostEqual(boosted.last_tick_rate_multiplier, 250.5)

    def test_current_propagation_and_retrograde_use_preupdate_state(self):
        n, old = cell(), cell(cls=PlasticityRateNeuron)
        for x in (n, old):
            x.M_vector[0] = .1
            x.input_buffer[:, 0] = (1., 1., 0.)
        events, references = n.tick({}, 1), old.tick({}, 1)
        self.assertEqual(n.propagation_queue, old.propagation_queue)
        self.assertEqual((n.S, n.O, n.t_ref), (old.S, old.O, old.t_ref))
        for event, reference in zip(events, references):
            if isinstance(event, RetrogradeSignalEvent):
                np.testing.assert_array_equal(event.error_vector, reference.error_vector)
        self.assertNotEqual(n.postsynaptic_points[1].u_i.info, old.postsynaptic_points[1].u_i.info)

    def test_inhibition_survives_long_alternating_credit(self):
        n = cell(eta=.001)
        values = []
        for t in range(1, 2001):
            n.M_vector[0] = .1
            n.t_last_fire = t if t % 100 < 60 else -10000
            n.input_buffer[1, 0] = 1.
            n.tick({}, t)
            values.append(n.postsynaptic_points[1].u_i.info)
        self.assertTrue(all(-10. <= x < 0 for x in values))
        self.assertGreater(n.bounded_updates, 1000)
        self.assertEqual(n.bounded_underflows, 0)

    def test_rejects_incompatible_modes_and_out_of_range_active_weights(self):
        n = cell()
        n.postsynaptic_points[1].u_i.info = -100.
        n.input_buffer[1, 0] = 1.
        with self.assertRaises(ValueError):
            n.tick({}, 1)
        for value in (-1., float("nan")):
            with self.assertRaises(ValueError):
                magnitude_step(.8, 1., 1, .1, cap=value)


if __name__ == "__main__":
    unittest.main()
