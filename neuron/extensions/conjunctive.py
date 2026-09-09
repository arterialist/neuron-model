"""Strictly inherited graded neuron with dendritic coincidence-gated release.

``GradedNeuron`` is appropriate for an analog afferent, but it is not by
itself an AND gate: a continuous angular-velocity input depolarises every
P-EN cell, including cells outside the active heading bump.  This subclass
keeps PAULA's ordinary membrane, delay, modulation, plasticity, and event
scheduling semantics, but replaces release for a narrowly opted-in cell with
the local dendritic coincidence:

    release = gain * min(max(ring_dendrite, 0), max(velocity_dendrite - S0, 0))

That implements the heading × angular-velocity conjunction measured for fly
P-EN cells (Turner-Evans et al., 2017; Green et al., 2017).  A cell without
``conjunctive_graded_gain`` delegates exactly to :class:`GradedNeuron`.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .graded import GradedNeuron
from ..neuron import NeuronEvent


class ConjunctiveGradedNeuron(GradedNeuron):
    """Release only when the designated bump and velocity dendrites coincide."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        md = self.metadata or {}
        self._cg_gain = float(md.get("conjunctive_graded_gain", 0.0) or 0.0)
        self._cg_ring = int(md.get("conjunctive_ring_synapse", 0))
        self._cg_velocity = int(md.get("conjunctive_velocity_synapse", 1))
        velocity_synapses = md.get("conjunctive_velocity_synapses")
        self._cg_velocity_synapses = (
            (self._cg_velocity,) if velocity_synapses is None
            else tuple(int(synapse) for synapse in velocity_synapses)
        )
        # Some PAULA vestibular projections arrive as an excitatory and an
        # inhibitory dendrite on the same P-EN.  Keeping both locally lets
        # the inherited coincidence gate see their signed opponent residual
        # instead of rectifying the positive half-wave before subtraction.
        velocity_negative = md.get("conjunctive_velocity_negative_synapse")
        self._cg_velocity_negative = (
            None if velocity_negative is None else int(velocity_negative)
        )
        self._cg_s0 = float(md.get("conjunctive_graded_S0", 0.0) or 0.0)
        self._cg_ring_tau = max(1.0, float(md.get("conjunctive_ring_tau", 1.0) or 1.0))
        self._cg_velocity_tau = max(1.0, float(md.get("conjunctive_velocity_tau", 1.0) or 1.0))
        # Optional local phase lead on the signed velocity dendrite.  It is
        # the discrete first-order lead ``v + k*(v-v_previous)``; no body
        # coordinate, heading estimate, or host-side filter participates.
        self._cg_velocity_lead_gain = float(
            md.get("conjunctive_velocity_lead_gain", 0.0) or 0.0
        )
        self._cg_ring_state = 0.0
        self._cg_velocity_state = 0.0
        self._cg_velocity_previous = 0.0
        self._on_gain = float(md.get("opponent_normalized_gain", 0.0) or 0.0)
        self._on_positive = int(md.get("opponent_positive_synapse", 0))
        self._on_negative = int(md.get("opponent_negative_synapse", 1))
        self._on_epsilon = float(md.get("opponent_normalized_epsilon", 1e-6) or 1e-6)
        self._ol_gain = float(md.get("opponent_lead_gain", 0.0) or 0.0)
        self._ol_previous_release = 0.0

    def _dendritic_drive(self, synapse_id: int) -> float:
        if synapse_id not in self.postsynaptic_points or synapse_id >= self.input_buffer.shape[0]:
            return 0.0
        input_info = float(self.input_buffer[synapse_id, 0])
        synapse = self.postsynaptic_points[synapse_id]
        return input_info * float(synapse.u_i.info + synapse.u_i.plast)

    def tick(self, external_inputs: Dict[int, Dict[str, Any]], current_tick: int,
             dt: float = 1.0) -> List[NeuronEvent]:
        if self._on_gain > 0.0:
            positive = max(0.0, self._dendritic_drive(self._on_positive))
            # The opponent dendrite is an inhibitory synapse.  Its local
            # potential is therefore negative; recover its magnitude here.
            negative = max(0.0, -self._dendritic_drive(self._on_negative))
            inherited = super().tick(external_inputs, current_tick, dt)
            events = [event for event in inherited if not (
                isinstance(event, tuple) and len(event) == 3 and event[0] == self.id
            )]
            # Divisive opponent normalization: release represents the
            # fractional left/right imbalance, not raw gait-stroke power.
            # This is the local rate-code counterpart of P-EN populations'
            # left/right trade-off around an approximately constant total.
            release = self._on_gain * max(0.0, positive - negative) / (
                self._on_epsilon + positive + negative
            )
            if self._ol_gain:
                release = max(0.0, release + self._ol_gain * (release - self._ol_previous_release))
            self._ol_previous_release = release
            self.O = release
            if release > 0.0:
                for terminal_id, terminal in self.presynaptic_points.items():
                    events.append((self.id, terminal_id, float(terminal.u_o.info) * release))
            return events
        if self._ol_gain:
            inherited = super().tick(external_inputs, current_tick, dt)
            raw_release = float(self.O)
            events = [event for event in inherited if not (
                isinstance(event, tuple) and len(event) == 3 and event[0] == self.id
            )]
            release = max(0.0, raw_release + self._ol_gain * (raw_release - self._ol_previous_release))
            self._ol_previous_release = raw_release
            self.O = release
            if release > 0.0:
                for terminal_id, terminal in self.presynaptic_points.items():
                    events.append((self.id, terminal_id, float(terminal.u_o.info) * release))
            return events
        if self._cg_gain <= 0.0:
            return super().tick(external_inputs, current_tick, dt)

        # Capture local potentials before GradedNeuron performs its inherited
        # propagation and clears ``input_buffer`` for the next PAULA tick.
        bump = max(0.0, self._dendritic_drive(self._cg_ring))
        velocity_drive = sum(
            self._dendritic_drive(synapse) for synapse in self._cg_velocity_synapses
        )
        if self._cg_velocity_negative is not None:
            velocity_drive += self._dendritic_drive(self._cg_velocity_negative)
        raw_velocity_drive = velocity_drive
        if self._cg_velocity_lead_gain:
            velocity_drive += self._cg_velocity_lead_gain * (
                velocity_drive - self._cg_velocity_previous
            )
        self._cg_velocity_previous = raw_velocity_drive
        velocity = max(0.0, velocity_drive - self._cg_s0)
        # A P-EN's local dendrites retain a short analog trace of their two
        # inputs.  With the default tau=1 this is the strict instantaneous
        # coincidence rule above.  A declared tau>1 makes the same PAULA
        # neuron tolerant to the one-tick transport offset between a graded
        # vestibular event and a sparse ring release; it is not a host filter.
        self._cg_ring_state += (bump - self._cg_ring_state) / self._cg_ring_tau
        self._cg_velocity_state += (velocity - self._cg_velocity_state) / self._cg_velocity_tau
        inherited = super().tick(external_inputs, current_tick, dt)
        # Remove only the parent's unrestricted graded release from this
        # source. Retrograde/plasticity events remain inherited unchanged.
        events = [event for event in inherited if not (
            isinstance(event, tuple) and len(event) == 3 and event[0] == self.id
        )]
        release = self._cg_gain * min(self._cg_ring_state, self._cg_velocity_state)
        self.O = release
        if release > 0.0:
            for terminal_id, terminal in self.presynaptic_points.items():
                events.append((self.id, terminal_id, float(terminal.u_o.info) * release))
        return events
