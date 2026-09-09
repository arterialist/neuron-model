"""Opt-in local pre/post timing traces on declared excitatory input ports.

Motivation: Guetig et al. 2003, doi:10.1523/JNEUROSCI.23-09-03697.2003,
methods equations 1-2 and all-pair accumulation. This uses the linear
weight-dependence case, not their nonlinear competition mechanism. It is a
PAULA extension experiment, not a reproduction of their conductance neuron.

At each tick, exponentially decay each port's presynaptic trace x and the
cell's postsynaptic trace y. Let a_i mark a positive arriving info signal,
and o mark an actual PAULA spike. Using traces before adding current events:
  Lplus_i = o*x_i; Lminus_i = alpha*a_i*y
  dq_i/ds = Lplus_i*(cap-q_i) - Lminus_i*q_i
Integrate this held-coefficient flow over the existing effective eta_post.
Then add a_i to x_i and o to y. Zero-lag pairs are excluded explicitly.
Traces measure receptor arrival vs somatic firing, not dendritic arrival.
There is no stimulus identity, firing target, host error, or finite hold timer.

Default eligibility_ports=[] takes the exact inherited tick path. Selected
ports require nonnegative info weights and zero plastic throughput; inhibitory
ports and all other mechanisms retain the bounded PAULA rule. Selected weights
can change on a later somatic spike without a new presynaptic arrival. Existing
weak positive rates and local neuromodulatory scaling remain active.

The inherited tick produces pre-update propagation and native retrograde
errors. We retain that outgoing-terminal adaptation deliberately, as a separate
native learning process, and replace only selected incoming info updates.
Inherited bounded counters/logs include provisional selected-port updates;
use eligibility_updates for this extension's effective selected updates.
The additional traces are dynamical state and must be checkpointed explicitly.
"""
import math

import numpy as np

from .bounded_plasticity import BoundedPlasticityNeuron


def eligibility_step(q, plus, minus, eta, cap):
    q, plus, minus = np.asarray(q), np.asarray(plus), np.asarray(minus)
    if (not math.isfinite(eta) or eta <= 0 or not math.isfinite(cap) or cap <= 0 or
            any(not np.isfinite(a).all() for a in (q, plus, minus)) or
            np.any(q < 0) or np.any(q > cap) or np.any(plus < 0) or np.any(minus < 0)):
        raise ValueError("Invalid eligibility flow")
    total = plus+minus
    equilibrium = np.divide(cap*plus, total, out=np.zeros_like(q, dtype=float), where=total > 0)
    amount = -np.expm1(-eta*total)
    return q + (equilibrium-q)*amount


class EligibilityTraceNeuron(BoundedPlasticityNeuron):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ports = self.metadata.get("eligibility_ports", [])
        if any(type(i) is not int or i < 0 or i >= self.params.num_inputs for i in ports) or len(set(ports)) != len(ports):
            raise ValueError("Eligibility ports must be unique configured input indices")
        self.eligibility_ports = tuple(ports)
        self.eligibility_pre = np.zeros(len(ports), dtype=float)
        self.eligibility_post = 0.
        self.eligibility_last_tick = None
        self.eligibility_updates = 0
        self.eligibility_tau_pre = float(self.metadata.get("eligibility_tau_pre", 4.))
        self.eligibility_tau_post = float(self.metadata.get("eligibility_tau_post", 4.))
        self.eligibility_alpha = float(self.metadata.get("eligibility_alpha", 1.))
        self.eligibility_cap = float(self.metadata.get("eligibility_cap", 1.))
        if ports:
            if not self._bounded_enabled:
                raise ValueError("Eligibility experiment requires bounded native control")
            if any(not math.isfinite(x) or x <= 0 for x in (self.eligibility_tau_pre, self.eligibility_tau_post,
                                                           self.eligibility_alpha, self.eligibility_cap)):
                raise ValueError("Eligibility constants must be finite and positive")
            if self.eligibility_cap > self._magnitude_cap:
                raise ValueError("Eligibility cap exceeds the inherited bound")

    def tick(self, external_inputs, current_tick, dt=1.):
        if not self.eligibility_ports:
            return super().tick(external_inputs, current_tick, dt)
        if dt != 1. or (self.eligibility_last_tick is not None and current_tick != self.eligibility_last_tick+1):
            raise ValueError("Eligibility traces currently require consecutive unit ticks")
        points = [self.postsynaptic_points[i] for i in self.eligibility_ports]
        if any(p.u_i.plast != 0 or p.u_i.info < 0 or p.u_i.info > self.eligibility_cap for p in points):
            raise ValueError("Selected inputs must have nonnegative info and zero plastic throughput")
        before = np.array([p.u_i.info for p in points])
        arrivals = (self.input_buffer[list(self.eligibility_ports), 0] > 0).astype(float)
        eta = self.params.eta_post*self.rate_multiplier()
        pre = self.eligibility_pre*math.exp(-1./self.eligibility_tau_pre)
        post = self.eligibility_post*math.exp(-1./self.eligibility_tau_post)
        events = super().tick(external_inputs, current_tick, dt)
        spike = float(self.O > 0)
        after = eligibility_step(before, spike*pre, self.eligibility_alpha*arrivals*post, eta, self.eligibility_cap)
        for point, weight in zip(points, after):
            point.u_i.info = float(weight)
        self.eligibility_updates += int(np.count_nonzero(before != after))
        self.eligibility_pre = pre+arrivals
        self.eligibility_post = post+spike
        self.eligibility_last_tick = current_tick
        return events
