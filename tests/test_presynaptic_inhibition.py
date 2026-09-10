import numpy as np
import pytest

from neuron.neuron import Neuron, NeuronParameters, PostsynapticPoint, PostsynapticInputVector, PresynapticPoint, PresynapticOutputVector
from neuron.extensions.experimental.presynaptic_inhibition import PresynapticInhibitionNeuron


def cell(cls=PresynapticInhibitionNeuron, gain=1.):
    c = cls(1, NeuronParameters(num_inputs=2, lambda_param=1., c=3, eta_post=1e-8, eta_retro=1e-6), log_level="CRITICAL")
    for port, strength in ((0, -1.), (1, 3.)):
        c.postsynaptic_points[port] = PostsynapticPoint(PostsynapticInputVector(info=strength, plast=0., adapt=np.zeros(2)))
        c.distances[port] = 2 if port == 0 else 0
    for port in (0, 1):
        c.presynaptic_points[port] = PresynapticPoint(PresynapticOutputVector(info=1., mod=np.zeros(2)))
    c.synapse_sources[0] = (8, 0)
    if cls is PresynapticInhibitionNeuron:
        c.configure_presynaptic_inhibition([0], [0], gain, 10.)
    return c


def test_zero_gain_preserves_native_soma_learning_and_all_events():
    a, b = cell(Neuron), cell(gain=0.)
    for t in range(30):
        for c in (a,b):
            c.input_buffer[0,0] = 1. if t % 7 == 0 else 0.
            c.input_buffer[1,0] = 1. if t % 4 == 0 else 0.
        ea, eb = a.tick({}, t), b.tick({}, t)
        assert len(ea) == len(eb)
        for x,y in zip(ea, eb):
            if isinstance(x,tuple): assert x == y
            else: np.testing.assert_array_equal(x.error_vector, y.error_vector)
        assert (a.S,a.O,a.F_avg,a.t_ref) == (b.S,b.O,b.F_avg,b.t_ref)
        for port in (0,1): assert a.postsynaptic_points[port].u_i.info == b.postsynaptic_points[port].u_i.info


def test_delay_selective_release_and_recovery_without_changing_spikes():
    a,b = cell(Neuron),cell()
    for t in range(30):
        for c in (a,b):
            c.input_buffer[0,0] = 1. if t == 0 else 0.
            c.input_buffer[1,0] = 1. if t % 4 == 0 else 0.
        ea,eb = a.tick({},t),b.tick({},t)
        assert (a.S,a.O,a.F_avg) == (b.S,b.O,b.F_avg)
        if t < 2: assert b.inhibition_state == 0
        else: assert b.inhibition_state == pytest.approx(.95**2*np.exp(-(t-2)/10))
        for x,y in zip(ea,eb):
            if isinstance(x,tuple):
                assert y[:2] == x[:2]
                assert y[2] == (x[2]*b.inhibition_fraction if x[1] == 0 else x[2])
            else: np.testing.assert_array_equal(x.error_vector, y.error_vector)
    assert .9 < b.inhibition_fraction < 1
    weight = b.postsynaptic_points[0].u_i.info
    b.reset_additional_state()
    assert b.inhibition_state == 0 and b.inhibition_fraction == 1 and not b.inhibition_queue
    assert b.postsynaptic_points[0].u_i.info == weight


def test_rejects_invalid_or_silently_changed_configuration():
    c = cell()
    with pytest.raises(RuntimeError): c.configure_presynaptic_inhibition([0],[0],1.,10.)
    c.tick({},0)
    with pytest.raises(ValueError): c.tick({},2)
    c.postsynaptic_points[0].u_i.info = 1.
    with pytest.raises(ValueError, match="polarity"): c.tick({},1)
    c = cell(); c.input_buffer[0,0] = np.nan
    with pytest.raises(ValueError): c.tick({},0)
