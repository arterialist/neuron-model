"""Opt-in magnitude-coordinate feedback for signed inputs to graded PAULA cells.

The native information return compares arrival a with signed throughput w.
This experiment compares a with |w| when w < 0, preserving w's inhibitory sign
in forward integration. All other return components, native timing direction,
postsynaptic learning, delays and presynaptic update equations are unchanged.
It changes a local error coordinate, not an action, target label or host loss.

For a non-spiking target, native direction d=-1. Holding q=|w| and source
release activity h fixed, the approximate terminal update is
u_next = u + eta*(q-h*u). Its positive fixed point q/h differs from the
negative target induced by comparing arrival with -q. This is a local
held-coefficient observation, NOT a positivity or stability proof for a
multi-target, delayed, continuously adapting network.

The equation is a PAULA consistency hypothesis, not a molecular reconstruction
of endocannabinoid signaling. Incoming signed weights still follow the existing
bounded rule; that rule's error magnitude is intentionally not changed here.
No permanent terminal reset, floor, rate freeze or task-dependent gate is added.

Metadata retrograde_magnitude_error=True opts in. Missing/False follows the
inherited path exactly. Initially restricted to bounded graded cells, where
the failure motivating this experiment was measured. Old classes and ordinary
call sites are not changed. The mixin permits explicit alternative compositions.
"""
import numpy as np

from ...neuron import RetrogradeSignalEvent
from .predictive_receptor import PredictiveReceptorNeuron


class MagnitudeRetrogradeMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        enabled=self.metadata.get('retrograde_magnitude_error',False)
        if type(enabled) is not bool:
            raise ValueError('retrograde_magnitude_error must be boolean')
        self.retrograde_magnitude_error=enabled
        self.magnitude_retrograde_events=0
        if enabled and (getattr(self,'_gg',0)<=0 or not getattr(self,'_bounded_enabled',False)):
            raise ValueError('Magnitude feedback currently requires bounded graded cells')

    def tick(self, external_inputs, current_tick, dt=1.):
        if not self.retrograde_magnitude_error:
            return super().tick(external_inputs,current_tick,dt)
        magnitude_errors={}
        for sid in np.flatnonzero(self.input_buffer[:,0]>0):
            point=self.postsynaptic_points.get(sid)
            if point is not None and point.u_i.info<0:
                if point.u_i.plast!=0:
                    raise ValueError('Magnitude feedback requires zero plastic throughput on inhibitory ports')
                # Match the native scalar dtypes and use the PRE-update weight.
                magnitude_errors[int(sid)]=self.input_buffer[sid,0]-abs(point.u_i.info)
        events=super().tick(external_inputs,current_tick,dt)
        direction=1. if current_tick-self.t_last_fire<=self.t_ref else -1.
        for event in events:
            if isinstance(event,RetrogradeSignalEvent) and event.source_synapse_id in magnitude_errors:
                event.error_vector[0]=magnitude_errors[event.source_synapse_id]*direction
                self.magnitude_retrograde_events+=1
        return events


class MagnitudeRetrogradeNeuron(MagnitudeRetrogradeMixin,PredictiveReceptorNeuron):
    """Explicit composition with the existing local predictive-receptor extension."""
