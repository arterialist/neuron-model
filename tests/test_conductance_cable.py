from copy import deepcopy

import numpy as np
import pytest

from neuron.extensions.experimental.conductance_cable import ConductanceCable
from neuron.extensions.experimental.passive_cable import PassiveCable


def make(cls=ConductanceCable):
    return cls(np.array([-1, 0, 1]), np.array([[0., 0., 0.], [2., 0., 0.], [2., 3., 0.]]),
               np.array([.2, .3, .25]), 20.)


def test_zero_conductance_preserves_native_trajectory_exactly():
    a, b = make(), make(PassiveCable)
    for tick in range(35):
        current = np.array([.1, -.03, 0.])*(tick % 4)
        a.step_conductance(current, .05, np.zeros(3))
        b.step(current, .05)
        np.testing.assert_array_equal(a.voltage, b.voltage)
        np.testing.assert_array_equal(a.last_current, b.last_current)


def test_changing_ligand_and_native_steps_match_dense_tick_equation():
    a = make()
    expected = a.voltage.copy()
    for tick in range(40):
        alpha = (.05, .1)[tick % 2]
        current = np.array([.01, .0, -.001])
        g = a.capacity*np.array([tick % 3, 0., .2])
        if tick % 5 == 0:
            g = np.zeros(3)
            a.step(current, alpha)
        else:
            a.step_conductance(current, alpha, g, .8)
        matrix = np.diag(a.capacity+alpha*g) + alpha*a.laplacian.toarray()
        expected = np.linalg.solve(matrix, (1-alpha)*a.capacity*expected+alpha*(current+g*.8))
        np.testing.assert_allclose(a.voltage, expected, atol=2e-14)
        np.testing.assert_allclose(a.last_current, current+g*(.8-expected), atol=2e-14)


def test_uniform_conductance_has_known_full_tick_trajectory():
    for gain in (.1, 1., 10., 100.):
        a = make()
        scalar = 0.
        for tick in range(30):
            a.step_conductance(np.zeros(3), .05, a.capacity*gain)
            scalar = (.95*scalar+.05*gain)/(1+.05*gain)
            np.testing.assert_allclose(a.voltage, scalar, atol=2e-14)
            assert np.all(a.voltage >= 0) and np.all(a.voltage <= 1)


def test_clone_rebuilds_factor_without_resetting_state():
    a = make()
    g = a.capacity*np.array([3., 0., .5])
    a.step_conductance(np.zeros(3), .05, g)
    b = deepcopy(a)
    assert b._ligand_factor is None
    for tick in range(8):
        a.step_conductance(np.zeros(3), .05, g)
        b.step_conductance(np.zeros(3), .05, g)
        np.testing.assert_array_equal(a.voltage, b.voltage)


@pytest.mark.parametrize("g", [[-1., 0., 0.], [np.nan, 0., 0.], [1., 2.]])
def test_invalid_conductance_rejected(g):
    with pytest.raises(ValueError):
        make().step_conductance(np.zeros(3), .05, g)
