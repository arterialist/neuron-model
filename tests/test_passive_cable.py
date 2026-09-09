"""Equation and neural-interface checks, not physiological acceptance."""
import copy

import numpy as np
import pytest

from neuron.neuron import (NeuronParameters, PostsynapticPoint, PostsynapticInputVector,
                           PresynapticPoint, PresynapticOutputVector, RetrogradeSignalEvent)
from neuron.extensions.graded import GradedNeuron
from neuron.extensions.experimental.passive_cable import PassiveCable, LocalCableGradedNeuron


def cable(ratio=25):
    return PassiveCable(np.array([-1, 0, 1]), [[0, 0, 0], [10, 0, 0], [20, 0, 0]],
                        [.2, .2, .2], ratio)


def cell(local=True, delay=2):
    cls = LocalCableGradedNeuron if local else GradedNeuron
    c = cls(100, NeuronParameters(num_inputs=3, lambda_param=4, eta_post=1e-6, eta_retro=1e-6),
            metadata={"graded_gain": .1, "graded_max": 1}, log_level="CRITICAL")
    for sid, weight in enumerate((2.0, -1.0, 1.0)):
        c.postsynaptic_points[sid] = PostsynapticPoint(PostsynapticInputVector(weight, 0, np.zeros(2)))
        c.distances[sid] = delay if sid < 2 else 0
        c.register_source(sid, sid + 1, 0)
    for tid in (4, 7):
        c.presynaptic_points[tid] = PresynapticPoint(PresynapticOutputVector(1, np.zeros(2)), 1)
    if local:
        c.configure_cable(cable(), [0, 1], [0, 2], [4, 7], [0, 2], [0, 2], uniform_input_port=2)
    return c


def test_dense_equation_and_conservation_with_signed_current():
    c = cable()
    current = np.array([2.0, 0, -1.0])
    for _ in range(12):
        old = c.voltage.copy()
        matrix = np.diag(c.capacity) + .25 * c.laplacian.toarray()
        expected = np.linalg.solve(matrix, .75 * c.capacity * old + .25 * current)
        expected_mass = .75 * (c.capacity @ old) + .25 * current.sum()
        c.step(current, .25)
        np.testing.assert_allclose(c.voltage, expected, atol=1e-13, rtol=1e-13)
        assert c.capacity @ c.voltage == pytest.approx(expected_mass, abs=1e-13)


def test_spatial_spread_and_washout_are_dynamic_not_a_static_distance_lookup():
    c = cable()
    c.step([1, 0, 0], .25)
    assert c.voltage[0] > c.voltage[1] > c.voltage[2] > 0
    near, far = c.voltage[0], c.voltage[2]
    c.step([0, 0, 0], .25)
    assert c.voltage[0] < near and c.voltage[2] > far
    for _ in range(150):
        c.step([0, 0, 0], .25)
    assert np.max(abs(c.voltage)) < 1e-17


def test_uniform_input_matches_native_graded_while_learning_stays_active():
    local, global_cell = cell(), cell(False)
    for tick in range(30):
        for c in (local, global_cell):
            c.input_buffer[2, 0] = 1.0 if tick < 15 else 0
        local_events = local.tick({}, tick)
        global_events = global_cell.tick({}, tick)
        # The unextended membrane iterates in float32 for these native inputs;
        # the conservative cable iterates in float64. This checks the uniform
        # physical limit within the former's explicit numerical precision.
        assert local.S == pytest.approx(global_cell.S, abs=1e-7)
        assert local.O == pytest.approx(global_cell.O, abs=1e-8)
        np.testing.assert_allclose(local.cable.voltage, global_cell.S, atol=1e-7, rtol=0)
        assert [e.error_vector.tolist() for e in local_events if isinstance(e, RetrogradeSignalEvent)] == [
            e.error_vector.tolist() for e in global_events if isinstance(e, RetrogradeSignalEvent)]
        assert local.postsynaptic_points[2].u_i.info == global_cell.postsynaptic_points[2].u_i.info
    assert local.postsynaptic_points[2].u_i.info != 1
    assert local.params.eta_post > 0 and local.params.eta_retro > 0
    assert local.t_last_fire == -np.inf and not local._ablation


def test_delayed_native_prelearning_potential_and_actual_local_terminal_events():
    c = cell(delay=2)
    c.input_buffer[0, 0] = 1
    first = c.tick({}, 0)
    assert any(isinstance(e, RetrogradeSignalEvent) for e in first)
    assert c.postsynaptic_points[0].u_i.info != 2
    assert not c.cable.voltage.any()
    c.postsynaptic_points[0].u_i.info = 99  # queued potential must remain the original 2
    c.tick({}, 1)
    assert not c.cable.voltage.any()
    events = c.tick({}, 2)
    arriving = float(np.float32(2) * .95**2)
    assert c.arrived_port_current.tolist() == [arriving, 0, 0]
    assert c.cable.last_current.sum() == pytest.approx(arriving)
    forward = {e[1]: e[2] for e in events if isinstance(e, tuple)}
    assert forward[4] > forward[7] > 0
    assert list(forward) == c.terminal_ids.tolist()
    assert not c.propagation_queue


def test_each_input_splits_aggregate_current_instead_of_multiplying_contact_count():
    c = cell(False)
    c.__class__ = LocalCableGradedNeuron
    c.configure_cable(cable(), [0, 0, 1], [0, 1, 2], [4, 4, 7], [0, 1, 2], [0, 1, 2], uniform_input_port=2)
    np.testing.assert_array_equal(c.input_projection @ [2, 1, 0], [1, 1, 1])
    np.testing.assert_array_equal(c.terminal_projection @ [1, 3, 5], [2, 5])


def test_deepcopy_rebuilds_factor_without_losing_queue_or_cable_state():
    c = cell()
    for t in range(5):
        c.input_buffer[0, 0] = 1
        c.tick({}, t)
    cloned = copy.deepcopy(c)
    assert cloned.cable._factor is None
    assert cloned.propagation_queue == c.propagation_queue
    for t in range(5, 15):
        a, b = c.tick({}, t), cloned.tick({}, t)
        np.testing.assert_array_equal(c.cable.voltage, cloned.cable.voltage)
        np.testing.assert_array_equal(c.terminal_release, cloned.terminal_release)
        assert [e for e in a if isinstance(e, tuple)] == [e for e in b if isinstance(e, tuple)]


@pytest.mark.parametrize("alpha", [0, -1, 1.1, float('nan')])
def test_unstable_timestep_is_rejected(alpha):
    with pytest.raises(ValueError, match="dt/lambda"):
        cable().step([1, 0, 0], alpha)


def test_zero_endpoint_radius_is_explicit_segment_mean_not_zero_conductance():
    c = PassiveCable(np.array([-1, 0]), [[0, 0, 0], [10, 0, 0]], [.2, 0], 25)
    c.step([1, 0], .25)
    assert c.voltage[1] > 0
    with pytest.raises(ValueError, match="zero-mean-radius"):
        PassiveCable(np.array([-1, 0]), [[0, 0, 0], [10, 0, 0]], [0, 0], 25)


@pytest.mark.parametrize("parents", [[-1, -1, 1], [2, 0, 1], [-1, 1, 1]])
def test_disconnected_and_cyclic_trees_fail(parents):
    with pytest.raises(ValueError):
        PassiveCable(np.array(parents), [[0, 0, 0], [10, 0, 0], [20, 0, 0]], [.2] * 3, 25)


def test_unconfigured_or_incomplete_contact_mapping_cannot_run():
    c = cell(False)
    c.__class__ = LocalCableGradedNeuron
    with pytest.raises(RuntimeError, match="not configured"):
        c.tick({}, 0)
    with pytest.raises(ValueError, match="Every anatomical input"):
        c.configure_cable(cable(), [0], [0], [4, 7], [0, 2], [0, 2], uniform_input_port=2)


def test_network_reset_clears_compartments_without_erasing_learning():
    from neuron.network import NeuronNetwork
    net = NeuronNetwork(num_neurons=0)
    c = cell()
    net.network.neurons[c.id] = c
    for tick in range(4):
        c.input_buffer[2, 0] = 1
        c.tick({}, tick)
    learned = c.postsynaptic_points[2].u_i.info
    factor = c.cable._factor
    assert c.cable.voltage.any()
    net.reset_simulation()
    assert not c.cable.voltage.any() and not c.cable.last_current.any()
    assert not c.terminal_release.any() and not c.local_release.any()
    assert c.S == c.O == c.F_avg == 0
    assert c.postsynaptic_points[2].u_i.info == learned != 1
    assert c.cable._factor is factor
    c.tick({}, 0)
    assert c.S == c.O == 0
