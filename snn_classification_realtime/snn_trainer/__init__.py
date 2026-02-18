"""SNN classifier training package."""

import sys
from pathlib import Path

_parent = Path(__file__).resolve().parent.parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

from snn_classification_realtime.snn_trainer.dataset import (
    ActivityDataset,
    collate_fn,
)
from snn_classification_realtime.snn_trainer.model import SNNClassifier

__all__ = ["ActivityDataset", "collate_fn", "SNNClassifier"]
