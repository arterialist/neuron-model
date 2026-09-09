"""Input-current dynamics, release identity and native adaptation invariants."""
import copy

import numpy as np
import pytest

from neuron.neuron import (Neuron, NeuronParameters, PostsynapticPoint,
                           PostsynapticInputVector, PresynapticPoint,
                           PresynapticOutputVector, RetrogradeSignalEvent)
from neuron.extensions.experimental.input_current import InputCurrentNeuron


def cell(mode="peak", tau=(3., 15.), fractions=(.8, .2), filtered=True):
    cls = InputCurrentNeuron if filtered else Neuron
    c = cls(100, NeuronParameters(num_inputs=3, lambda_param=20, c=3,
                                eta_post=1e-6, eta_retro=1e-6), log_level="CRITICAL")
    for p, weight in enumerate((2., -1., 1.)):
        c.postsynaptic_points[p] = PostsynapticPoint(
            PostsynapticInputVector(weight, 0, np.zeros(2)))
        c.distances[p] = (2, 4, 0)[p]
        c.register_source(p, p+1, 0)
    c.presynaptic_points[0] = PresynapticPoint(PresynapticOutputVector(1, np.zeros(2)), 1)
    if filtered:
        c.configure_current_kernel([0, 1], tau, fractions, mode)
    return c


def test_single_release_has_tail_but_no_extra_plasticity_or_retrograde_events():
    c = cell()
    c.input_buffer[0, 0] = 1
    first = c.tick({}, 0)
    weight = c.postsynaptic_points[0].u_i.info
    assert weight != 2
    assert len([e for e in first if isinstance(e, RetrogradeSignalEvent)]) == 1
    assert c.total_current == 0
    for t in range(1, 40):
        events = c.tick({}, t)
        assert not any(isinstance(e, RetrogradeSignalEvent) for e in events)
        assert c.postsynaptic_points[0].u_i.info == weight
        amplitude = float(np.float32(2) * .95**2)
        expected = 0 if t < 2 else amplitude*(.8*np.exp(-(t-2)/3)+.2*np.exp(-(t-2)/15))
        assert c.total_current == pytest.approx(expected)
        assert c.arrived_port_impulse[0] == (amplitude if t == 2 else 0)
    assert not c.propagation_queue and c.S > 0


@pytest.mark.parametrize("dt", [1., .25])
def test_area_conservation_preserves_shape_with_overlapping_signed_events(dt):
    peak, area = cell(), cell("area")
    total_impulse = np.zeros(3)
    total_current = np.zeros(3)
    gain = sum(f/(1-np.exp(-dt/t)) for f, t in zip((.8, .2), (3, 15)))
    for t in range(2000):
        for c in (peak, area):
            if t in (0, 7):
                c.input_buffer[0, 0] = 1
            if t == 3:
                c.input_buffer[1, 0] = 1
            if t == 10:
                c.input_buffer[2, 0] = .2  # unfiltered current-injection control
            c.tick({}, t, dt)
        total_impulse += area.arrived_port_impulse*dt
        total_current += area.last_port_current*dt
        np.testing.assert_allclose(peak.last_port_current[:2], area.last_port_current[:2]*gain,
                                   atol=1e-12, rtol=1e-5)
        assert peak.last_port_current[2] == area.last_port_current[2]
    np.testing.assert_allclose(total_impulse, total_current, atol=1e-12)
    assert total_current[0] > 0 and total_current[1] < 0


def test_queued_potential_is_not_recomputed_from_later_weights():
    c = cell()
    c.input_buffer[0, 0] = 1
    c.tick({}, 0)
    c.postsynaptic_points[0].u_i.info = 99
    c.tick({}, 1)
    c.tick({}, 2)
    assert c.total_current == pytest.approx(float(np.float32(2) * .95**2))


def test_unfiltered_current_has_identical_spikes_and_learning_to_native():
    a, b = cell(), cell(filtered=False)
    for t in range(180):
        for c in (a, b):
            c.input_buffer[2, 0] = 4 if t < 100 else 0
        ea, eb = a.tick({}, t), b.tick({}, t)
        assert a.O == b.O and a.t_last_fire == b.t_last_fire
        assert a.S == pytest.approx(b.S, abs=2e-7)
        assert a.postsynaptic_points[2].u_i.info == b.postsynaptic_points[2].u_i.info
        assert [e for e in ea if isinstance(e, tuple)] == [e for e in eb if isinstance(e, tuple)]
        assert len(ea) == len(eb)
    assert a.t_last_fire > 0


def test_tails_survive_spikes_and_deepcopy_but_reset_preserves_learning():
    from neuron.network import NeuronNetwork
    c = cell()
    c.input_buffer[0, 0] = 50
    for t in range(5):
        c.tick({}, t)
    assert c.t_last_fire > 0 and c.total_current > 0
    clone = copy.deepcopy(c)
    for t in range(5, 15):
        c.tick({}, t)
        clone.tick({}, t)
        np.testing.assert_array_equal(c.current_state, clone.current_state)
        assert c.S == clone.S and c.O == clone.O
    net = NeuronNetwork(num_neurons=0)
    net.network.neurons[c.id] = c
    learned = c.postsynaptic_points[0].u_i.info
    net.reset_simulation()
    assert not c.current_state.any() and c.total_current == 0
    assert not c.arrived_port_impulse.any() and not c.last_port_current.any()
    assert c.postsynaptic_points[0].u_i.info == learned != 2
    c.tick({}, 0)
    assert c.S == c.total_current == 0


@pytest.mark.parametrize("kwargs", [
    {"ports": [0, 0]}, {"ports": [-1]}, {"ports": [99]}, {"ports": [.1]},
    {"decay_ticks": [0, 2]}, {"decay_ticks": [np.inf, 2]},
    {"peak_fractions": [-.1, 1.1]}, {"peak_fractions": [.1, .2]},
    {"normalization": "silent_gain"},
])
def test_invalid_configuration_fails(kwargs):
    c = InputCurrentNeuron(1, NeuronParameters(num_inputs=3), log_level="CRITICAL")
    c.postsynaptic_points = cell().postsynaptic_points
    args = dict(ports=[0, 1], decay_ticks=[3, 15], peak_fractions=[.8, .2])
    args.update(kwargs)
    with pytest.raises(ValueError):
        c.configure_current_kernel(**args)


def test_missing_config_and_time_changes_fail_before_native_mutation():
    c = InputCurrentNeuron(1, NeuronParameters(num_inputs=1), log_level="CRITICAL")
    with pytest.raises(RuntimeError, match="not configured"):
        c.tick({}, 0)
    c = cell()
    c.tick({}, 0)
    for tick, dt in ((2, 1), (0, 1), (1, .5), (1, 0), (1, np.nan)):
        c.input_buffer[0, 0] = 1
        with pytest.raises(ValueError):
            c.tick({}, tick, dt)
        assert not c.propagation_queue and c.S == 0
