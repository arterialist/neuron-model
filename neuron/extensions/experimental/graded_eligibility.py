"""Compose existing graded release with the existing plasticity extensions.

No new dynamical equation is introduced. GradedNeuron owns the release rule;
EligibilityTraceNeuron supplies bounded native learning and local rate control.
With graded_gain<=0 this follows the ordinary EligibilityTraceNeuron path.

Graded cells may not select spike-based eligibility ports. Their lack of a
somatic spike would leave the postsynaptic eligibility trace without a defined
event. Their other incoming ports retain weak native bounded adaptation,
including the negative native timing direction for a cell that never spikes.
This is an experimental composition, not a cone or synaptic-release model.
"""
from ..graded import GradedNeuron
from .eligibility_trace import EligibilityTraceNeuron


class GradedEligibilityNeuron(GradedNeuron, EligibilityTraceNeuron):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._gg > 0 and self.eligibility_ports:
            raise ValueError('Graded release does not define a postsynaptic spike for eligibility ports')
