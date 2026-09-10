import copy
from unittest.mock import patch

import numpy as np
import pytest

from neuron.neuron import Neuron, NeuronParameters, PostsynapticPoint, PostsynapticInputVector
from neuron.network import NeuronNetwork
from neuron.extensions.experimental.electrical import ElectricalNeuron, ElectricalCoupling


def preparation(g=.1, reverse=False):
    net = NeuronNetwork(num_neurons=0)
    cells = {}
    for i in (0, 1):
        c = ElectricalNeuron(i, NeuronParameters(num_inputs=1, lambda_param=20., eta_post=1e-6), log_level="CRITICAL")
        c.postsynaptic_points[0] = PostsynapticPoint(PostsynapticInputVector(1., 0., np.zeros(2)))
        c.distances[0] = 0
        c.configure_electrical_input()
        cells[i] = c
    net.network.neurons = dict(reversed(list(cells.items()))) if reverse else cells
    return net, ElectricalCoupling(net, [(0, 1, g)])


def test_exchange_is_conservative_dissipative_and_order_independent():
    net, coupling = preparation()
    other, second = preparation(reverse=True)
    for n in (net, other):
        n.network.neurons[0].S = -.8
        n.network.neurons[1].S = -.2
    for _ in range(100):
        before = [net.network.neurons[i].S for i in (0, 1)]
        coupling.run_tick(); second.run_tick()
        currents = [net.network.neurons[i].electrical_current for i in (0, 1)]
        assert sum(currents) == 0
        assert np.dot(before, currents) <= 0
        assert [net.network.neurons[i].S for i in (0, 1)] == [other.network.neurons[i].S for i in (0, 1)]


def test_no_coupling_matches_ordinary_with_learning():
    from neuron.neuron import Neuron
    net, coupled = preparation(g=0.)
    native = Neuron(0, copy.deepcopy(net.network.neurons[0].params), log_level="CRITICAL")
    native.postsynaptic_points = copy.deepcopy(net.network.neurons[0].postsynaptic_points)
    native.distances = {0: 0}
    for t in range(80):
        info = 50. if t % 7 == 0 else 0.
        native.input_buffer[0, 0] = info
        net.network.neurons[0].input_buffer[0, 0] = info
        native.tick({}, t)
        coupled.run_tick()
        c = net.network.neurons[0]
        assert (c.S, c.O, c.t_ref) == (native.S, native.O, native.t_ref)
        assert c.postsynaptic_points[0].u_i.info == native.postsynaptic_points[0].u_i.info
    assert native.postsynaptic_points[0].u_i.info != 1.


def test_passive_step_approaches_analytic_coupling_ratio_and_resets():
    net, coupling = preparation(g=.03)
    native = Neuron._hillock_current

    def electrode(cell, tick, dt):
        # Inject current during integration, not voltage before the junction
        # snapshot. The latter imposes a different discrete-time experiment.
        current = native(cell, tick, dt)
        return current-.5 if cell.id == 0 else current

    with patch.object(Neuron, "_hillock_current", electrode):
        for _ in range(700):
            coupling.run_tick()
    a, b = (net.network.neurons[i] for i in (0, 1))
    assert b.S/a.S == pytest.approx(.03/1.03, abs=1e-12)
    net.reset_simulation(); coupling.run_tick()
    assert a.S == b.S == 0.


def test_unequal_leaks_conserve_physical_flux_not_normalized_current():
    net, coupling = preparation(g=.3)
    a, b = (net.network.neurons[i] for i in (0, 1))
    a.electrical_leak_conductance = 2.
    b.electrical_leak_conductance = 5.
    a.S, b.S = -.8, -.2
    for _ in range(100):
        before = np.array([a.S, b.S])
        coupling.run_tick()
        flux = np.array([a.electrical_current*2, b.electrical_current*5])
        assert flux.sum() == pytest.approx(0., abs=1e-16)
        assert np.dot(before, flux) <= 0


def test_invalid_wiring_and_missing_snapshot_fail():
    net, coupling = preparation()
    for edges in ([(0, 1, .1), (1, 0, .1)], [(0, 2, .1)], [(0, 0, .1)], [(0, 1, -1.)], [(0, 1, 100.)]):
        with pytest.raises(ValueError):
            ElectricalCoupling(net, edges)
    with pytest.raises(ValueError, match="snapshot"):
        net.network.neurons[0].tick({}, 0)


@pytest.mark.parametrize("filtered", [False, True])
def test_zero_gap_preserves_native_events_return_learning_and_current_tails(filtered):
    from neuron.neuron import PresynapticPoint, PresynapticOutputVector, RetrogradeSignalEvent
    from neuron.extensions.experimental.input_current import InputCurrentNeuron
    from neuron.extensions.experimental.electrical import ElectricalCurrentNeuron
    cls = ElectricalCurrentNeuron if filtered else ElectricalNeuron
    base = InputCurrentNeuron if filtered else Neuron
    pair = []
    for typ in (cls, base):
        c = typ(0, NeuronParameters(num_inputs=1, eta_post=1e-6, eta_retro=1e-6), log_level="CRITICAL")
        c.postsynaptic_points[0] = PostsynapticPoint(PostsynapticInputVector(1., 0., np.zeros(2)))
        c.presynaptic_points[0] = PresynapticPoint(PresynapticOutputVector(1., np.zeros(2)), 1.)
        c.distances[0] = 2
        c.register_source(0, 0, 0)
        if filtered:
            c.configure_current_kernel([0], [3., 15.], [.8, .2], "peak")
        pair.append(c)
    a, b = pair
    a.configure_electrical_input()
    return_count = forward_count = 0
    for tick in range(160):
        for c in pair:
            c.input_buffer[0, 0] = 50. if tick % 7 == 0 else 0.
        a.electrical_tick, a.electrical_voltage = tick, float(a.S)
        ea, eb = a.tick({}, tick), b.tick({}, tick)
        assert repr(ea) == repr(eb)
        assert (a.S, a.O, a.t_ref) == (b.S, b.O, b.t_ref)
        for c, events in zip(pair, (ea, eb)):
            for e in events:
                if isinstance(e, RetrogradeSignalEvent):
                    c.process_retrograde_signal(e)
        return_count += sum(isinstance(e, RetrogradeSignalEvent) for e in ea)
        forward_count += sum(isinstance(e, tuple) for e in ea)
        assert a.postsynaptic_points[0].u_i.info == b.postsynaptic_points[0].u_i.info
        assert a.presynaptic_points[0].u_o.info == b.presynaptic_points[0].u_o.info
        if filtered:
            np.testing.assert_array_equal(a.current_state, b.current_state)
    assert forward_count > 0 and return_count > 0
    assert a.postsynaptic_points[0].u_i.info != 1.
    assert a.presynaptic_points[0].u_o.info != 1.


def test_combined_extension_deepcopy_and_reset_clear_both_dynamic_states():
    from neuron.extensions.experimental.electrical import ElectricalCurrentNeuron
    net, _ = preparation()
    old = net.network.neurons[0]
    c = ElectricalCurrentNeuron(0, old.params, log_level="CRITICAL")
    c.postsynaptic_points = old.postsynaptic_points
    c.distances = old.distances
    c.configure_electrical_input()
    c.configure_current_kernel([0], [3., 15.], [.8, .2])
    net.network.neurons[0] = c
    coupled = ElectricalCoupling(net, [(0, 1, .1)])
    c.input_buffer[0, 0] = 50.
    for _ in range(10):
        coupled.run_tick()
    clone = copy.deepcopy(coupled)
    for _ in range(30):
        coupled.run_tick(); clone.run_tick()
        for i in (0, 1):
            a, b = net.network.neurons[i], clone.network.network.neurons[i]
            assert (a.S, a.O, a.electrical_current) == (b.S, b.O, b.electrical_current)
    learned = c.postsynaptic_points[0].u_i.info
    net.reset_simulation()
    assert c.postsynaptic_points[0].u_i.info == learned != 1.
    assert not c.current_state.any() and c.electrical_current == 0
    coupled.run_tick()
    assert c.S == c.total_current == c.electrical_current == 0
