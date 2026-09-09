"""Experimental phenomenological phase sampler for PAULA diagnostics.

This optional subclass accumulates a *signed local dendritic difference* until
a local clock dendrite fires, then stores it for a configurable number of
ticks.  It has no body coordinate, heading, motor command, world object, or
host-language state: the clock and signal are ordinary PAULA inputs.

That containment is an implementation fact, not a biological validation.  A
discrete accumulator plus sample/reset/hold counter is an engineered
phenomenological convenience; it is **not** a faithful cellular model of a
P-EN, central-complex circuit, or gait-phase integration.  Keep it restricted
to explicitly named diagnostic experiments.  A biological candidate must use
ordinary PAULA synapses, explicit phase circuitry, and inhibitory/reset
interneurons instead of this subclass.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..conjunctive import ConjunctiveGradedNeuron
from ...neuron import NeuronEvent


class PhaseLockedGradedNeuron(ConjunctiveGradedNeuron):
    """Experimental only: local signed integrate/sample/hold release.

    This class is deliberately not used by default agent configurations and
    must never be reported as biologically faithful behavior.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        md = self.metadata or {}
        self._pl_gain = float(md.get("phase_locked_gain", 0.0) or 0.0)
        self._pl_positive = int(md.get("phase_locked_positive_synapse", 0))
        self._pl_negative = int(md.get("phase_locked_negative_synapse", 1))
        self._pl_clock = int(md.get("phase_locked_clock_synapse", 2))
        self._pl_clock_threshold = float(md.get("phase_locked_clock_threshold", 0.0) or 0.0)
        # Zero preserves sample-and-hold.  A positive duration turns the
        # stored value into a finite transmitter pulse.  Both modes remain
        # experimental modeling conveniences, not a cellular claim.
        self._pl_hold_ticks = max(0, int(md.get("phase_locked_hold_ticks", 0) or 0))
        self._pl_accumulator = 0.0
        self._pl_held = 0.0
        self._pl_remaining = 0

    def tick(self, external_inputs: Dict[int, Dict[str, Any]], current_tick: int,
             dt: float = 1.0) -> List[NeuronEvent]:
        if self._pl_gain <= 0.0:
            return super().tick(external_inputs, current_tick, dt)

        signed_drive = (
            self._dendritic_drive(self._pl_positive)
            + self._dendritic_drive(self._pl_negative)
        )
        clock = self._dendritic_drive(self._pl_clock)
        # Call the inherited implementation to preserve all PAULA membrane,
        # delay, modulation, plasticity, and buffer-clearing semantics, then
        # replace only this cell's unrestricted graded release.
        inherited = super().tick(external_inputs, current_tick, dt)
        events = [event for event in inherited if not (
            isinstance(event, tuple) and len(event) == 3 and event[0] == self.id
        )]

        self._pl_accumulator += signed_drive
        if clock > self._pl_clock_threshold:
            self._pl_held = max(0.0, self._pl_accumulator)
            self._pl_accumulator = 0.0
            self._pl_remaining = self._pl_hold_ticks
        active = self._pl_hold_ticks == 0 or self._pl_remaining > 0
        release = self._pl_gain * self._pl_held if active else 0.0
        if self._pl_hold_ticks and self._pl_remaining > 0:
            self._pl_remaining -= 1
        self.O = release
        if release > 0.0:
            for terminal_id, terminal in self.presynaptic_points.items():
                events.append((self.id, terminal_id, float(terminal.u_o.info) * release))
        return events
