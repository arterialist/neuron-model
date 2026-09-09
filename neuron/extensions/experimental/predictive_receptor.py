"""Local signed-error plasticity for explicitly declared contextual inputs.

Experimental PAULA extension, NOT a reproduction of Urbanczik & Senn's
two-compartment conductance model. Their distinction between contextual
prediction and independently arriving evidence motivates the circuit design;
this class uses a phenomenological three-factor update instead.

Positive/negative neural error channels arrive as ordinary information signals
on zero-throughput receptor ports. This preserves analog release amplitude;
PAULA's terminal modulation is not multiplied by graded release. Receptor
polarity is an anatomical setting, never a stimulus label or host loss.

At each unit tick: use the PREVIOUS completed receptor trace e, update each
context trace x from its actual arriving amplitude, and update selected weights
by q <- clip(q + eta(e)*e*x, 0, cap). eta(e) has a strictly positive basal
rate and bounded, symmetric error-dependent amplification. Then update the
receptor trace from this tick's arriving opponent signals. There is no
training/recall switch. Silence is evidence of absence, not missing-data magic.

Native membrane integration, delayed propagation, all other incoming learning,
outgoing terminal adaptation, and retrograde messages remain active. Only the
selected contextual information weights replace the inherited provisional
updates. With no prediction_ports, the inherited path is unchanged exactly.
The receptor and contextual traces are additional checkpoint state.
"""
import math

import numpy as np

from .graded_eligibility import GradedEligibilityNeuron


def predictive_step(q, x, error, eta, cap):
    q, x = np.asarray(q, dtype=float), np.asarray(x, dtype=float)
    if (q.shape != x.shape or not np.isfinite(q).all() or not np.isfinite(x).all()
            or not all(math.isfinite(v) for v in (error, eta, cap))
            or eta <= 0 or cap <= 0 or np.any(x < 0)
            or np.any(q < 0) or np.any(q > cap)):
        raise ValueError('Invalid predictive receptor update')
    return np.clip(q + eta * error * x, 0., cap)


class PredictiveReceptorNeuron(GradedEligibilityNeuron):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        md = self.metadata
        ports = md.get('prediction_ports', [])
        if (len(set(ports)) != len(ports) or any(type(i) is not int or
                not 0 <= i < self.params.num_inputs for i in ports)):
            raise ValueError('Prediction ports must be unique input indices')
        self.prediction_ports = tuple(ports)
        self.prediction_error_ports = tuple(tuple(p) for p in md.get('prediction_error_ports', []))
        self.prediction_tau_context = float(md.get('prediction_tau_context', 8.))
        self.prediction_tau_error = float(md.get('prediction_tau_error', 4.))
        self.prediction_cap = float(md.get('prediction_cap', 1.))
        self.prediction_boost = float(md.get('prediction_boost', 499.))
        self.prediction_half = float(md.get('prediction_half', .01))
        self.prediction_context = np.zeros(len(ports))
        self.prediction_arrivals = np.zeros(len(ports))
        self.prediction_error = 0.
        self.prediction_error_arrival = 0.
        self.prediction_error_used = 0.
        self.prediction_eta = self.params.eta_post
        self.prediction_last_tick = None
        self.prediction_bound_hits = 0
        if not ports:
            if self.prediction_error_ports:
                raise ValueError('Error receptors without prediction ports')
            return
        if not self._bounded_enabled or self._gg <= 0 or self.eligibility_ports:
            raise ValueError('Predictor requires bounded graded release without spike eligibility')
        if self._rate_boost != 0:
            raise ValueError('Predictor uses its own declared local rate receptor')
        positive = (self.prediction_tau_context, self.prediction_tau_error,
                    self.prediction_cap, self.prediction_half, self.params.eta_post,
                    self.params.eta_retro)
        if (any(not math.isfinite(v) or v <= 0 for v in positive)
                or not math.isfinite(self.prediction_boost) or self.prediction_boost < 0
                or self.prediction_cap > self._magnitude_cap):
            raise ValueError('Invalid predictive receptor constants')
        receptors = self.prediction_error_ports
        if (len(receptors) != 2 or any(len(p) != 2 for p in receptors)
                or {p[1] for p in receptors} != {-1, 1}
                or len({p[0] for p in receptors}) != 2
                or any(type(s) is not int or not 0 <= s < self.params.num_inputs
                       or s in ports for s, _ in receptors)):
            raise ValueError('Need separate positive and negative error receptor ports')

    def tick(self, external_inputs, current_tick, dt=1.):
        if not self.prediction_ports:
            return super().tick(external_inputs, current_tick, dt)
        if dt != 1. or (self.prediction_last_tick is not None and
                       current_tick != self.prediction_last_tick + 1):
            raise ValueError('Predictive receptor requires consecutive unit ticks')
        if any(s in external_inputs for s, _ in self.prediction_error_ports):
            raise ValueError('Predictive teaching receptor must be driven by neurons')
        points = [self.postsynaptic_points[s] for s in self.prediction_ports]
        if any(p.u_i.plast != 0 for p in points):
            raise ValueError('Context requires zero plastic throughput')
        if any(self.postsynaptic_points[s].u_i.info != 0 or
               self.postsynaptic_points[s].u_i.plast != 0
               for s, _ in self.prediction_error_ports):
            raise ValueError('Error receptor must not directly drive the membrane')
        before = np.array([p.u_i.info for p in points])
        arrivals = self.input_buffer[list(self.prediction_ports), 0].astype(float).copy()
        if not np.isfinite(arrivals).all() or np.any(arrivals < 0):
            raise ValueError('Context amplitudes must be finite and nonnegative')
        error_arrivals = np.array([self.input_buffer[s, 0] for s, _ in self.prediction_error_ports])
        if not np.isfinite(error_arrivals).all() or np.any(error_arrivals < 0):
            raise ValueError('Opponent receptor amplitudes must be finite and nonnegative')
        d = math.exp(-1. / self.prediction_tau_context)
        x = d * self.prediction_context + (1.-d) * arrivals
        e = self.prediction_error
        eta = self.params.eta_post * (1. + self.prediction_boost *
                                     abs(e) / (self.prediction_half + abs(e)))
        after = predictive_step(before, x, e, eta, self.prediction_cap)
        events = super().tick(external_inputs, current_tick, dt)
        for point, value in zip(points, after):
            point.u_i.info = float(value)
        self.prediction_bound_hits += int(np.count_nonzero(
            (before + eta * e * x < 0) | (before + eta * e * x > self.prediction_cap)))
        self.prediction_context = x
        self.prediction_arrivals = arrivals
        self.prediction_error_used = e
        self.prediction_error_arrival = float(sum(a * polarity for a, (_, polarity)
                                                  in zip(error_arrivals, self.prediction_error_ports)))
        d = math.exp(-1. / self.prediction_tau_error)
        self.prediction_error = d * e + (1.-d) * self.prediction_error_arrival
        self.prediction_eta = eta
        self.prediction_last_tick = current_tick
        return events
