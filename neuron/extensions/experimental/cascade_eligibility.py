"""Local cascaded eligibility for delayed neural teaching signals.

Only the selected predictive information update changes. A cascade of leaky
nonnegative local states replaces the single exponential eligibility:
  z[0]' = d*z[0] + (1-d)*arrival
  z[k]' = d*z[k] + (1-d)*z[k-1]'
  q' = clip(q + eta(previous_error)*previous_error*z[-1]', 0, cap).
Stages update sequentially within a unit tick. The impulse kernel has unit
total mass and mean age stages*d/(1-d). It has no hold counter or task clock.

This is a phenomenological intracellular cascade, not a reconstruction of a
named signaling pathway. Suvrathan, Payne & Raymond (2016), Neuron 92:959-967,
doi:10.1016/j.neuron.2016.10.022 motivates delay-sensitive plasticity, not these
equations. Timing is specified anatomy here, not learned delay estimation.

The inherited exponential context remains observable. Native forward release,
all unselected plasticity and all retrograde events retain their pre-update
order. Only final selected weights and their bound counter are replaced.
No host loss, behavioral rule, pretrained value or training/recall switch.

Default stages=1 with no explicit mean follows the parent exactly. Positive
basal rates remain mandatory. The cascade is additional checkpoint state.
"""
import math

import numpy as np

from .magnitude_retrograde import MagnitudeRetrogradeNeuron


def cascade_step(state, arrivals, decay):
    state, arrivals = np.asarray(state, dtype=float), np.asarray(arrivals, dtype=float)
    if (state.ndim != 2 or not 1 <= state.shape[0] <= 32 or
            arrivals.shape != state.shape[1:] or not 0 < decay < 1 or
            not np.isfinite(state).all() or not np.isfinite(arrivals).all() or
            np.any(state < 0) or np.any(arrivals < 0)):
        raise ValueError('Invalid local eligibility cascade')
    result = state.copy(); incoming = arrivals
    for i in range(len(result)):
        result[i] = decay*state[i] + (1-decay)*incoming
        incoming = result[i]
    return result


class CascadeEligibilityNeuron(MagnitudeRetrogradeNeuron):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        stages = self.metadata.get('prediction_credit_stages', 1)
        mean = self.metadata.get('prediction_credit_mean')
        if type(stages) is not int or not 1 <= stages <= 32:
            raise ValueError('Need 1..32 local eligibility stages')
        self.credit_enabled = stages != 1 or mean is not None
        if self.credit_enabled and not self.prediction_ports:
            raise ValueError('A credit cascade requires selected predictive ports')
        legacy_d = math.exp(-1/self.prediction_tau_context)
        self.credit_mean = legacy_d/(1-legacy_d) if mean is None else float(mean)
        if not math.isfinite(self.credit_mean) or self.credit_mean <= 0:
            raise ValueError('Credit mean age must be finite and positive')
        self.credit_decay = self.credit_mean/(self.credit_mean+stages) if self.credit_enabled else legacy_d
        self.credit_states = np.zeros((stages,len(self.prediction_ports)))

    def tick(self, external_inputs, current_tick, dt=1.):
        if not self.credit_enabled:
            events = super().tick(external_inputs,current_tick,dt)
            if self.prediction_ports:
                self.credit_states[0] = self.prediction_context
            return events
        points = [self.postsynaptic_points[s] for s in self.prediction_ports]
        before = np.array([p.u_i.info for p in points]); hits = self.prediction_bound_hits
        arrivals = self.input_buffer[list(self.prediction_ports),0].astype(float)
        state = cascade_step(self.credit_states,arrivals,self.credit_decay)
        events = super().tick(external_inputs,current_tick,dt)
        raw = before+self.prediction_eta*self.prediction_error_used*state[-1]
        for point,value in zip(points,np.clip(raw,0.,self.prediction_cap)):
            point.u_i.info = float(value)
        self.prediction_bound_hits = hits+int(np.count_nonzero((raw<0)|(raw>self.prediction_cap)))
        self.credit_states = state
        return events
