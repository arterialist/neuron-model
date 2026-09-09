"""Cold-start ignition test for the tonic-driven reservoir.

Question: with a constant background (tonic) current + calibration, does the pool
ignite from a FRESH reset() and hold a sparse code (~target participation) under
both continuous input (A, period=1) and intermittent input (B, period=6)?

Sweeps tonic_drive to see its effect. Measures per-tick pool participation
(fraction of pool units firing) and its stability over a cold-start window.
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                "../../../../Projects/agi-research/neuron-model")))

from snn_classification_realtime.activity_dataset_builder.vision_datasets import (
    load_dataset_by_name,
)
from snn_classification_realtime.foveation.minibrain.core import MiniBrainConfig, MiniBrain

ds_cfg = load_dataset_by_name("cifar10_grayscale", train=True)
ds = ds_cfg.dataset


def pool_participation(brain, sig, n_ticks, per):
    """Stream `sig` every `per` ticks from cold; return per-tick pool spike fraction."""
    brain.sub.reset()
    m = brain._readout_mask   # pool = readout layer
    frac = []
    for t in range(n_ticks):
        drive = (sig if t % per == 0 else []) + brain._tonic
        st = brain.sub.step(drive)
        frac.append(float(st.O[m].mean()))
    return np.array(frac)


for tonic in [0.0, 0.15, 0.30]:
    cfg = MiniBrainConfig(dataset_name="cifar10_grayscale", tonic_drive=tonic,
                          target_participation=0.05, seed=0)
    print(f"\n{'='*66}\n=== tonic_drive = {tonic} ===")
    brain = MiniBrain(cfg, ds_cfg)
    sig = brain.sub.patch_to_signals(brain.retina.render(ds[0][0]))
    n_pool = int(brain._readout_mask.sum())
    print(f"  pool units (readout) = {n_pool} | tonic targets = {len(brain._tonic)}")
    for label, per in [("A continuous per=1", 1), ("B intermittent per=6", 6)]:
        f = pool_participation(brain, sig, 200, per)
        # split into early / late to see if it ignites and holds (not dies, not saturates)
        early, late = f[:50].mean(), f[-100:].mean()
        alive = (f[-100:] > 0).mean()   # fraction of late ticks with ANY pool activity
        print(f"    {label:22s}: early {early:.3f}  late {late:.3f}  "
              f"late-alive-ticks {alive:.2f}  peak {f.max():.3f}")
