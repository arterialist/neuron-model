"""Foveated perception for the PAULA/ALERM classification pipeline.

A fixed-resolution retina (k x k) is resampled from a larger image at a
movable position (fy, fx). A supervisor above the perception network reads
its free-energy proxy and injects m0/m1 (ALERM dual-channel neuromodulation)
to destabilize (search/saccade) or crystallize (fixate) the fovea's motion.

This package is additive; it does not change existing entrypoints.
"""

from snn_classification_realtime.foveation.fovea import Fovea
from snn_classification_realtime.foveation.perception import PerceptionNetwork
from snn_classification_realtime.foveation.signals import (
    FreeEnergyProbe,
    StabilityReward,
)

__all__ = [
    "Fovea",
    "PerceptionNetwork",
    "FreeEnergyProbe",
    "StabilityReward",
]
