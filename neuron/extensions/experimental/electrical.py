"""Opt-in passive electrical exchange alongside native PAULA chemical synapses.

Junction endpoints and conductances must be supplied explicitly. This module
never infers electrical contacts from a chemical connectome. Voltages are the
native somatic state, not reconstructed action-potential waveforms or dendritic
junction voltages. Conductance is in a shared, abstract voltage/current system.
"""
from __future__ import annotations

import math

from ...neuron import Neuron
from .input_current import InputCurrentNeuron


class ElectricalInput:
    """Receive a synchronous junction snapshot, compute current inside the cell."""

    def configure_electrical_input(self, leak_conductance=1.):
        if hasattr(self, "electrical_leak_conductance"):
            raise RuntimeError("Electrical input already configured")
        if not math.isfinite(leak_conductance) or leak_conductance <= 0:
            raise ValueError("Need a finite positive leak conductance")
        self.electrical_leak_conductance = float(leak_conductance)
        self.electrical_tick = None
        self.electrical_neighbors = ()
        self.electrical_current = 0.
        self.electrical_native_current = 0.

    def _hillock_current(self, current_tick, dt):
        self._validate_electrical_step(current_tick, dt)
        native = super()._hillock_current(current_tick, dt)
        flux = sum(g*(v-self.electrical_voltage) for v, g in self.electrical_neighbors)
        self.electrical_current = flux/self.electrical_leak_conductance
        self.electrical_native_current = native
        # Exact native numeric types and events when electrical current is zero.
        return native if self.electrical_current == 0 else float(native)+self.electrical_current

    def _validate_electrical_step(self, current_tick, dt):
        if self.electrical_tick != current_tick:
            raise ValueError("Missing synchronous electrical snapshot")
        if float(self.S) != self.electrical_voltage:
            raise ValueError("Soma changed after junction snapshot")
        gsum = sum(g for _, g in self.electrical_neighbors)
        if (not math.isfinite(dt) or dt <= 0 or not math.isfinite(self.params.lambda_param)
                or dt*(1+gsum/self.electrical_leak_conductance) > self.params.lambda_param):
            raise ValueError("Unstable explicit electrical/membrane time step")

    def tick(self, external_inputs, current_tick, dt=1.):
        self._validate_electrical_step(current_tick, dt)
        return super().tick(external_inputs, current_tick, dt)

    def reset_additional_state(self):
        reset = getattr(super(), "reset_additional_state", None)
        if reset is not None:
            reset()
        self.electrical_tick = None
        self.electrical_neighbors = ()
        self.electrical_current = self.electrical_native_current = 0.


class ElectricalNeuron(ElectricalInput, Neuron):
    pass


class ElectricalCurrentNeuron(ElectricalInput, InputCurrentNeuron):
    pass


class ElectricalCoupling:
    """Transport endpoint voltages before all native cell updates.

    Each undirected junction is declared once as (cell_id, cell_id, g). The
    two local currents are equal/opposite after multiplying by each cell's
    leak conductance. No events, synapses, firing decisions or learning rules
    are replaced. The caller owns the native network and its normal reset.
    """

    def __init__(self, network, junctions):
        self.network = network
        self.junctions = []
        self.cells = {i: c for i, c in network.network.neurons.items() if isinstance(c, ElectricalInput)}
        if not self.cells or any(not hasattr(c, "electrical_leak_conductance") for c in self.cells.values()):
            raise ValueError("Need configured PAULA electrical cells")
        seen = set()
        for a, b, g in junctions:
            if a == b or a not in self.cells or b not in self.cells or not math.isfinite(g) or g < 0:
                raise ValueError("Invalid electrical endpoints or conductance")
            pair = tuple(sorted((a, b)))
            if pair in seen:
                raise ValueError("Duplicate electrical junction")
            seen.add(pair)
            self.junctions.append((a, b, float(g)))
        # Reject unstable wiring before any native state can advance.
        for i, c in self.cells.items():
            gsum = sum(g for a, b, g in self.junctions if i in (a, b))
            if 1+gsum/c.electrical_leak_conductance > c.params.lambda_param:
                raise ValueError("Unstable electrical coupling at native dt=1")

    def run_tick(self):
        tick = self.network.current_tick
        voltage = {i: float(c.S) for i, c in self.cells.items()}
        if not all(math.isfinite(v) for v in voltage.values()):
            raise ValueError("Nonfinite electrical voltage")
        neighbors = {i: [] for i in self.cells}
        for a, b, g in self.junctions:
            neighbors[a].append((voltage[b], g))
            neighbors[b].append((voltage[a], g))
        for i, c in self.cells.items():
            c.electrical_tick = tick
            c.electrical_voltage = voltage[i]
            c.electrical_neighbors = tuple(neighbors[i])
        for c in self.cells.values():
            c._validate_electrical_step(tick, 1.)
        return self.network.run_tick()
