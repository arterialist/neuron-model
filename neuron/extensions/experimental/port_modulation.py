"""Port-specific susceptibility to an existing local plasticity modulator.

An opt-in extension of EligibilityTraceNeuron. For declared native-rule input
ports, eta_i = eta_basal * (1 + sensitivity_i * (cell_gain - 1)).
Sensitivity zero removes modulation, not basal adaptation. Empty declarations
or sensitivity one preserve the inherited path exactly. Eligibility ports may
not be overridden. Forward propagation, somatic dynamics, eligibility learning
and outgoing retrograde adaptation retain their inherited rules.

This is a phenomenological receptor-sensitivity hypothesis, not a molecular
model. Motivation: Mitsushima et al. 2013, doi:10.1038/ncomms3760, distinct
cholinergic receptor contributions to excitatory/inhibitory plasticity. Their
results do not prescribe this equation or universally slower inhibition.

Metadata native_port_modulation is a list of {port: int, sensitivity: float}.
There is no stimulus label, host reset, task error or frozen weight. As in the
bounded extension, the inherited provisional update is replaced before another
cell can observe it. Only the selected incoming info values are replaced.
"""
import math
import numpy as np

from .eligibility_trace import EligibilityTraceNeuron
from .bounded_plasticity import magnitude_step


class PortModulationNeuron(EligibilityTraceNeuron):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        entries=self.metadata.get('native_port_modulation', [])
        self.native_port_modulation={}
        for entry in entries:
            if set(entry)!={'port','sensitivity'}:raise ValueError('Expected port and sensitivity')
            port=entry['port'];sensitivity=float(entry['sensitivity'])
            if type(port) is not int or not 0<=port<self.params.num_inputs or port in self.native_port_modulation:
                raise ValueError('Invalid or duplicate native input port')
            if port in self.eligibility_ports:raise ValueError('Cannot override eligibility ports')
            if not math.isfinite(sensitivity) or sensitivity<0:raise ValueError('Invalid modulation sensitivity')
            self.native_port_modulation[port]=sensitivity
        if entries and not self._bounded_enabled:raise ValueError('Requires bounded native learning')

    def tick(self, external_inputs, current_tick, dt=1.):
        selected=[(sid,s) for sid,s in self.native_port_modulation.items()
                  if s!=1. and self.input_buffer[sid,0]>0]
        if not selected:return super().tick(external_inputs,current_tick,dt)
        gain=self.rate_multiplier();basal=self.params.eta_post;pending=[]
        for sid,sensitivity in selected:
            p=self.postsynaptic_points[sid];row=self.input_buffer[sid]
            before=float(p.u_i.info)
            error=float(np.linalg.norm(np.array([row[0]-p.u_i.info,row[1]-p.u_i.plast,*row[2:]])))
            eta=basal*(1.+sensitivity*(gain-1.))
            if not math.isfinite(eta) or eta<=0:raise ValueError('Port rate must remain finite and positive')
            pending.append((sid,before,error,eta))
        events=super().tick(external_inputs,current_tick,dt)
        direction=1 if current_tick-self.t_last_fire<=self.t_ref else -1
        for sid,before,error,eta in pending:
            q=magnitude_step(abs(before),error,direction,eta,self._magnitude_cap,self._magnitude_decay)
            self.postsynaptic_points[sid].u_i.info=math.copysign(q,before)
        return events
