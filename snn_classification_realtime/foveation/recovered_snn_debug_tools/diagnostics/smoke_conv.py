"""Smoke: retinal-conv MiniBrain builds, ignites, runs present() for gray + color."""
import numpy as np
from snn_classification_realtime.activity_dataset_builder.vision_datasets import (
    load_dataset_by_name,
)
from snn_classification_realtime.foveation.minibrain.core import MiniBrainConfig, MiniBrain

for dn in ["cifar10_grayscale", "cifar10_color"]:
    ds_cfg = load_dataset_by_name(dn, train=True)
    ds = ds_cfg.dataset
    cfg = MiniBrainConfig(dataset_name=dn, retinal_conv=True, tonic_drive=0.15,
                          target_participation=0.05, dwell=40, seed=0)
    brain = MiniBrain(cfg, ds_cfg)
    m = brain._readout_mask
    print(f"\n=== {dn} ===")
    print(f"  input feature dim {brain.retconv.feature_dim(2*brain.C, cfg.grid)} | "
          f"n_neurons {brain.n_neurons} | pool {int(m.sum())} | conv scale {brain.retconv.scale:.3f}")
    # aliveness with image vs tonic-only
    sig = brain.sub.patch_to_signals(brain._encode(ds[0][0]))
    brain.sub.reset()
    fim = np.mean([brain.sub.step(sig + brain._tonic).O[m].mean() for _ in range(60)][-40:])
    brain.sub.reset()
    fto = np.mean([brain.sub.step(brain._tonic).O[m].mean() for _ in range(60)][-40:])
    print(f"  pool participation: image {fim:.3f}  tonic-only {fto:.3f}  (want image>>tonic)")
    # a few present() calls
    rng = np.random.RandomState(0)
    ok = 0
    for idx in rng.randint(0, len(ds), size=20):
        img, y = ds[idx]
        x, pred = brain.present(img, int(y), learn=True, teach=True)
        ok += int(np.isfinite(x).all())
    print(f"  present() ran 20x, finite reps {ok}/20, dim {x.size}, last m0/m1 {brain._last_nm}")
