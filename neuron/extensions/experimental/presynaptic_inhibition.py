"""Local, opt-in inhibition of selected native PAULA release terminals.

Declared inhibitory receptors drive a decaying intracellular state x. Native
release is multiplied by 1/(1+gain*x). The receptor still contributes its
ordinary somatic input and receives native plasticity/retrograde processing.
This effective receptor-to-release coupling is not a GABA receptor kinetic fit.
No stimulus labels, firing-rate decoder or population controller enters it.
"""
from __future__ import annotations

import heapq
import numpy as np

from ...neuron import Neuron


class PresynapticInhibitionNeuron(Neuron):
    def configure_presynaptic_inhibition(self, receptor_ports, terminals, gain, decay_ticks):
        if hasattr(self, "inhibition_state"):
            raise RuntimeError("Presynaptic inhibition already configured")
        ports = np.asarray(receptor_ports); outputs = np.asarray(terminals)
        for values, available in ((ports, self.postsynaptic_points), (outputs, self.presynaptic_points)):
            if (values.ndim != 1 or values.dtype.kind not in "iu" or not len(values)
                    or len(np.unique(values)) != len(values) or any(int(v) not in available for v in values)):
                raise ValueError("Need distinct existing receptors and terminals")
        if not np.isfinite([gain, decay_ticks]).all() or gain < 0 or decay_ticks <= 0:
            raise ValueError("Need nonnegative gain and positive decay time")
        for port in ports:
            p = self.postsynaptic_points[int(port)]
            if p.u_i.info+p.u_i.plast >= 0:
                raise ValueError("Declared inhibitory receptor must have a negative native coefficient")
            if self.distances[int(port)] < 0 or int(self.distances[int(port)]) != self.distances[int(port)]:
                raise ValueError("Need nonnegative integer propagation delays")
        self.inhibition_ports = ports.astype(np.int64, copy=True)
        self.inhibition_terminals = frozenset(map(int, outputs))
        self.inhibition_gain = float(gain)
        self.inhibition_decay_ticks = float(decay_ticks)
        self.reset_additional_state()

    def reset_additional_state(self):
        self.inhibition_state = 0.
        self.inhibition_fraction = 1.
        self.inhibition_arriving_drive = 0.
        self.inhibition_queue = []
        self.inhibition_last_native = {}
        self.inhibition_last_effective = {}
        self._inhibition_tick = self._inhibition_dt = None

    def tick(self, external_inputs, current_tick, dt=1.):
        if not hasattr(self, "inhibition_ports"):
            raise RuntimeError("Presynaptic inhibition is not configured")
        if (not np.isfinite(dt) or dt <= 0 or dt > self.params.lambda_param
                or (self._inhibition_dt is not None and dt != self._inhibition_dt)
                or (self._inhibition_tick is not None and current_tick != self._inhibition_tick+1)):
            raise ValueError("Need consecutive ticks and constant stable dt")
        for port in self.inhibition_ports:
            amplitude = float(self.input_buffer[port, 0])
            receptor = self.postsynaptic_points[int(port)]
            strength = float(-(receptor.u_i.info+receptor.u_i.plast))
            if not np.isfinite([amplitude, strength]).all() or amplitude < 0 or strength <= 0:
                raise ValueError("Invalid inhibitory signal or changed receptor polarity")
            if amplitude > 0:
                distance = self.distances[int(port)]
                drive = amplitude*strength*self.params.delta_decay**distance
                heapq.heappush(self.inhibition_queue, (current_tick+distance, int(port), drive))
        self.inhibition_state *= np.exp(-dt/self.inhibition_decay_ticks)
        self.inhibition_arriving_drive = 0.
        while self.inhibition_queue and self.inhibition_queue[0][0] <= current_tick:
            self.inhibition_arriving_drive += heapq.heappop(self.inhibition_queue)[2]
        self.inhibition_state += self.inhibition_arriving_drive
        self.inhibition_fraction = 1./(1.+self.inhibition_gain*self.inhibition_state)
        if not np.isfinite(self.inhibition_state) or not 0 < self.inhibition_fraction <= 1:
            raise FloatingPointError("Invalid inhibition state; no clipping or silent reset")
        events = super().tick(external_inputs, current_tick, dt)
        result = []
        self.inhibition_last_native = {}; self.inhibition_last_effective = {}
        for event in events:
            if not isinstance(event, tuple) or event[1] not in self.inhibition_terminals:
                result.append(event)
                continue
            self.inhibition_last_native[event[1]] = event[2]
            amplitude = event[2] if self.inhibition_fraction == 1. else event[2]*self.inhibition_fraction
            self.inhibition_last_effective[event[1]] = amplitude
            result.append(event if self.inhibition_fraction == 1. else (event[0], event[1], amplitude))
        self._inhibition_tick, self._inhibition_dt = current_tick, dt
        return result
