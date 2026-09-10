import copy

import numpy as np
import pytest

from neuron.neuron import Neuron, NeuronParameters, PostsynapticPoint, PostsynapticInputVector, PresynapticPoint, PresynapticOutputVector
from neuron.extensions.experimental.release_depression import DepressingReleaseNeuron


def cell(depletion=.22, recovery=893., native=False):
    cls = Neuron if native else DepressingReleaseNeuron
    c = cls(7, NeuronParameters(num_inputs=1, eta_post=1e-6, eta_retro=1e-6), log_level="CRITICAL")
    c.postsynaptic_points[0] = PostsynapticPoint(PostsynapticInputVector(1., 0., np.zeros(2)))
    c.distances[0] = 0
    c.register_source(0, 8, 0)
    for t in (0, 1):
        c.presynaptic_points[t] = PresynapticPoint(PresynapticOutputVector(1., np.zeros(2)), 1.)
    if not native:
        c.configure_release_depression([0], depletion, recovery)
    return c


def test_zero_depletion_is_bit_exact_native_with_learning_and_return_events():
    a, b = cell(0.), cell(native=True)
    for tick in range(160):
        for c in (a,b):
            c.input_buffer[0,0] = 50. if tick%7 == 0 else 0.
        x, y = a.tick({},tick), b.tick({},tick)
        assert repr(x) == repr(y)
        assert (a.S,a.O,a.t_ref) == (b.S,b.O,b.t_ref)
        assert a.postsynaptic_points[0].u_i.info == b.postsynaptic_points[0].u_i.info
    assert a.postsynaptic_points[0].u_i.info != 1.
    np.testing.assert_array_equal(a.release_available, 1.)


@pytest.mark.parametrize("dt", [1., .5])
def test_event_recurrence_matches_independent_analytic_solution(dt):
    c = cell()
    R, last = 1., None
    observed = []
    for tick in range(1200):
        c.input_buffer[0,0] = 50. if tick in [0,100,200,300,400,500,1100] else 0.
        events = c.tick({},tick,dt)
        forward = [e for e in events if isinstance(e,tuple)]
        if forward:
            if last is not None:
                R = 1-(1-R)*np.exp(-(tick-last)*dt/893.)
            assert forward[0][2] == pytest.approx(forward[1][2]*R, abs=1e-13)
            assert c.release_used_fraction[0] == pytest.approx(R,abs=1e-13)
            observed.append((tick,R))
            R *= .78
            last = tick
    assert len(observed) == 7 and observed[-1][1] > observed[-2][1]
    assert c.params.eta_post > 0 and c.params.eta_retro > 0 and not c._ablation


def test_deepcopy_and_reset_preserve_weights_but_reset_resources():
    from neuron.network import NeuronNetwork
    c = cell()
    for t in range(10):
        c.input_buffer[0,0] = 50.
        c.tick({},t)
    twin = copy.deepcopy(c)
    for t in range(10,30):
        assert repr(c.tick({},t)) == repr(twin.tick({},t))
        np.testing.assert_array_equal(c.release_available,twin.release_available)
    learned = c.postsynaptic_points[0].u_i.info
    net = NeuronNetwork(num_neurons=0)
    net.network.neurons[c.id] = c
    net.reset_simulation()
    np.testing.assert_array_equal(c.release_available,1.)
    assert c.postsynaptic_points[0].u_i.info == learned != 1.
    c.tick({},0)


def test_bad_configuration_and_time_fail_before_mutation():
    for depletion,recovery in ((-.1,1),(1.1,1),(.2,0),(.2,np.inf)):
        with pytest.raises(ValueError):
            cell(depletion,recovery)
    c = cell()
    c.tick({},0)
    for tick,dt in ((0,1),(2,1),(1,.5)):
        state = c.release_available.copy()
        with pytest.raises(ValueError):
            c.tick({},tick,dt)
        np.testing.assert_array_equal(state,c.release_available)
