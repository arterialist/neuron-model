"""Experimental neural-reference contrast in predictive learning eligibility.

Changes ONE equation on explicitly selected prediction ports:
  x_i <- d*x_i + (1-d)*a_i                 (inherited contextual trace)
  z_r <- d*z_r + (1-d)*b_r                 (new neural reference trace)
  q_i <- clip(q_i + eta(e)*e*(x_i-g*z_r), 0, cap)
Here b_r is an actual nonnegative arrival on a zero-throughput reference port,
not an average computed by the host. A circuit must generate the reference.
The inherited previous-tick signed error and strictly positive rate are used.
Signed eligibility is allowed: relative inactivity can reverse an update.

Forward potentials, native returns and all other learning use pre-update
weights in the inherited order. Reference ports have zero somatic throughput;
their native retrograde messages remain active. Added circuit wiring can
therefore still affect expression through return paths and later learning.
No equality of whole-network trajectories is implied by zero direct drive.

This is a phenomenological extension, not a demonstrated OLM cellular rule,
nor an estimator of statistical covariance over stimuli. Sejnowski (1977),
doi:10.1007/BF00275079, motivates distinguishing correlation from raw activity;
its model is not implemented here. No label, surprise flag, fitted weight or
training/recall switch enters this rule. The default absent map is exact.
"""
import math

import numpy as np

from .magnitude_retrograde import MagnitudeRetrogradeNeuron


class ContrastEligibilityNeuron(MagnitudeRetrogradeNeuron):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mapping = self.metadata.get('prediction_reference_map', [])
        self.prediction_reference_strength = float(self.metadata.get('prediction_reference_strength', 0.))
        g = self.prediction_reference_strength
        if not math.isfinite(g) or not 0 <= g <= 1:
            raise ValueError('Reference strength must be finite in 0..1')
        if (any(not isinstance(p, (list, tuple)) or len(p) != 2 or
                any(type(s) is not int for s in p) for p in mapping) or
                len({p[0] for p in mapping}) != len(mapping)):
            raise ValueError('Reference map requires unique contextual port mappings')
        lookup = dict(mapping)
        if mapping and set(lookup) != set(self.prediction_ports):
            raise ValueError('Map must cover every selected prediction port')
        if g and not mapping:
            raise ValueError('Active contrast requires neural reference ports')
        self.prediction_reference_ports = tuple(sorted(set(lookup.values())))
        excluded = set(self.prediction_ports) | {s for s,_ in self.prediction_error_ports}
        if any(s in excluded or not 0 <= s < self.params.num_inputs for s in self.prediction_reference_ports):
            raise ValueError('Reference ports must be separate valid inputs')
        self.prediction_reference_indices = np.array([
            self.prediction_reference_ports.index(lookup[s]) for s in self.prediction_ports
        ], dtype=int) if mapping else np.empty(0,dtype=int)
        self.prediction_reference = np.zeros(len(self.prediction_reference_ports))
        self.prediction_reference_arrivals = self.prediction_reference.copy()
        self.prediction_eligibility = self.prediction_context.copy()

    def tick(self, external_inputs, current_tick, dt=1.):
        ports = self.prediction_reference_ports
        if not ports:
            return super().tick(external_inputs, current_tick, dt)
        if any(s in external_inputs or s not in self.synapse_sources for s in ports):
            raise ValueError('Reference receptors must be driven by neurons')
        if any(self.postsynaptic_points[s].u_i.info != 0 or
               self.postsynaptic_points[s].u_i.plast != 0 for s in ports):
            raise ValueError('Reference must have zero somatic throughput')
        arrivals = self.input_buffer[list(ports),0].astype(float).copy()
        if not np.isfinite(arrivals).all() or np.any(arrivals < 0):
            raise ValueError('Reference arrivals must be finite and nonnegative')
        d = math.exp(-1/self.prediction_tau_context)
        reference = d*self.prediction_reference+(1-d)*arrivals
        points = [self.postsynaptic_points[s] for s in self.prediction_ports]
        before = np.array([p.u_i.info for p in points]); hits = self.prediction_bound_hits
        events = super().tick(external_inputs,current_tick,dt)
        effective = self.prediction_context-self.prediction_reference_strength*reference[self.prediction_reference_indices]
        if self.prediction_reference_strength:
            raw = before+self.prediction_eta*self.prediction_error_used*effective
            after = np.clip(raw,0.,self.prediction_cap)
            for point,value in zip(points,after): point.u_i.info = float(value)
            self.prediction_bound_hits = hits+int(np.count_nonzero((raw<0)|(raw>self.prediction_cap)))
        self.prediction_reference = reference
        self.prediction_reference_arrivals = arrivals
        self.prediction_eligibility = effective
        return events
