"""Phenomenological non-spiking PAULA extension, not a cellular reconstruction.

Opted-in cells retain the base membrane integration and delayed dendritic
inputs but suppress spike/reset dynamics. Release is
``graded_gain * max(0, S - graded_S0)``, optionally clipped to ``graded_max``.
This separates membrane integration time from discrete spike frequency.
It does not implement ion channels, calcium-dependent release or vesicle
depletion, and should not be presented as a faithful model of a named cell.

The inherited adaptation and plasticity code still runs. Its interpretation
needs care: ``F_avg`` measures graded output, not a spike rate; ``t_last_fire``
does not advance. Spike-timing-dependent learning is therefore NOT supplied
by this extension. Threshold neuromodulation is intentionally suppressed;
membrane, synaptic and learning-window effects remain as in the base model.

With ``graded_gain <= 0`` the exact base tick is used. No behavioural policy
or task-specific timing mechanism is implemented here.
"""
from typing import Any, Dict, List

from ..neuron import Neuron, NeuronEvent

# Sentinel used to suppress spiking. Large enough that S can never reach it given the model's own
# MAX_MEMBRANE_POTENTIAL clamp (1000.0), so the firing branch in the base tick() is unreachable.
_NEVER = 1e12


class GradedNeuron(Neuron):
    """Non-spiking neuron with tonic, depolarisation-proportional release.

    Opt-in per neuron via METADATA (NeuronParameters is a dataclass and rejects unknown keys, so the
    graded settings ride in `metadata`, which NetworkConfig already plumbs through and ckit exposes as
    `neuron(..., meta={...})`):
        graded_gain : float  release per unit depolarisation above `graded_S0`. <=0 -> base behaviour.
        graded_S0   : float  release threshold (resting offset). Default 0.0.
        graded_max  : float  clip on the emitted value. Default 0 (no clip).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        md = self.metadata or {}
        self._gg = float(md.get("graded_gain", 0.0) or 0.0)
        self._gs0 = float(md.get("graded_S0", 0.0) or 0.0)
        self._gmax = float(md.get("graded_max", 0.0) or 0.0)
        if self._gg > 0:
            # Never spike: the base tick() then leaves S untouched, so it integrates freely.
            self.r = _NEVER
            self.b = _NEVER

    def tick(self, external_inputs: Dict[int, Dict[str, Any]], current_tick: int,
             dt: float = 1.0) -> List[NeuronEvent]:
        if self._gg <= 0:
            return super().tick(external_inputs, current_tick, dt)

        # Suppress threshold recalculation BEFORE the inherited firing test.
        # Reasserting thresholds afterwards is too late: an ordinary r_base
        # has already triggered a spike and reset the integrating membrane.
        # Keep the caller's ablations and shared parameter object unchanged.
        self.r = _NEVER
        self.b = _NEVER
        frozen = "thresholds_frozen" in self._ablation
        self._ablation.add("thresholds_frozen")
        try:
            events = super().tick(external_inputs, current_tick, dt)
        finally:
            if not frozen:
                self._ablation.remove("thresholds_frozen")

        rel = self._gg * (float(self.S) - self._gs0)
        if rel <= 0.0:
            self.O = 0.0
            return events
        if self._gmax > 0.0:
            rel = min(rel, self._gmax)
        self.O = rel

        # Tonic release on every terminal, every tick -- the graded analogue of the base class's
        # spike-triggered release. Same lightweight tuple format the network already consumes.
        for terminal_id, terminal in self.presynaptic_points.items():
            events.append((self.id, terminal_id, float(terminal.u_o.info) * rel))
        return events
