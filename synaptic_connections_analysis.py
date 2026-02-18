"""Entry point for synaptic connectivity analysis. Delegates to viz package."""

import os
import sys

_root = os.path.dirname(os.path.abspath(__file__))
if _root not in sys.path:
    sys.path.insert(0, _root)

from snn_classification_realtime.viz.synaptic_analysis.run import main

if __name__ == "__main__":
    main()
