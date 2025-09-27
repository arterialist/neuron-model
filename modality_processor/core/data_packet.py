from dataclasses import dataclass
from typing import Any, Optional
import numpy as np

@dataclass
class DataPacket:
    """
    A standardized data structure for passing information through the processing pipeline.
    """
    data: np.ndarray
    modality: str
    timestamp: float  # Can be seconds, frame number, or other sequential metric
    metadata: Optional[dict[str, Any]] = None
