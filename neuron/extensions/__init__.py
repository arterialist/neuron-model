"""Opt-in PAULA neuron extensions.

The base :mod:`neuron.neuron` model remains unchanged.  Extensions live here
so callers can state explicitly when they require non-spiking graded release
or a dendritic conjunction.  Experimental mechanisms are deliberately kept in
the ``experimental`` subpackage and are not part of the normal extension API.
"""

from .graded import GradedNeuron
from .conjunctive import ConjunctiveGradedNeuron

__all__ = ["GradedNeuron", "ConjunctiveGradedNeuron"]
