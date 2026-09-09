"""
Neuron Model Package
Provides single neuron and multi-neuron network simulation capabilities.
"""

# Import core neuron functionality
from .neuron import Neuron, setup_neuron_logger, NeuronParameters
from .neuron import PostsynapticInputVector, PresynapticOutputVector
from .neuron import PostsynapticPoint, PresynapticPoint
from .neuron import (
    NeuronEvent,
    PresynapticReleaseEvent,
    RetrogradeSignalEvent,
)

# Import network functionality
from .network import NetworkTopology, TravelingSignal, NeuronNetwork

# Import configuration functionality
from .network_config import NetworkConfig

# Import core functionality
from .nn_core import NNCore, NNCoreState

# Explicit, opt-in extensions.  The experimental namespace is intentionally
# not imported here: a caller must opt in by naming it.
from .extensions import GradedNeuron, ConjunctiveGradedNeuron

__all__ = [
    # Single neuron components
    "Neuron",
    "NeuronParameters",
    "setup_neuron_logger",
    "PostsynapticInputVector",
    "PresynapticOutputVector",
    "PostsynapticPoint",
    "PresynapticPoint",
    "NeuronEvent",
    "PresynapticReleaseEvent",
    "RetrogradeSignalEvent",
    # Multi-neuron network components
    "NetworkTopology",
    "TravelingSignal",
    "NeuronNetwork",
    # Configuration components
    "NetworkConfig",
    # Core components
    "NNCore",
    "NNCoreState",
    # Stable opt-in extensions
    "GradedNeuron",
    "ConjunctiveGradedNeuron",
]

__version__ = "0.1.0"
