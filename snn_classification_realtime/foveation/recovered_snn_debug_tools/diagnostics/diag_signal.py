"""Where does the image signal die? Compare tonic-only vs tonic+image, per layer.

If pool participation is identical with and without the image, the reservoir is
running on tonic alone and the image is invisible -> tonic is too strong / the
input->buffer->pool path is too weak. We want: tonic alone = sub-threshold simmer
(near 0 pool firing), image = the thing that ignites it to target participation.
"""
import numpy as np
from snn_classification_realtime.activity_dataset_builder.vision_datasets import (
    load_dataset_by_name,
)
from snn_classification_realtime.foveation.minibrain.core import MiniBrainConfig, MiniBrain

ds_cfg = load_dataset_by_name("cifar10_grayscale", train=True)
ds = ds_cfg.dataset

cfg = MiniBrainConfig(dataset_name="cifar10_grayscale", tonic_drive=0.15,
                      target_participation=0.05, seed=0)
brain = MiniBrain(cfg, ds_cfg)
lay = brain.sub.layer_of_pos
layers = sorted(set(lay.tolist()))
sig = brain.sub.patch_to_signals(brain.retina.render(ds[0][0]))

def run(drive_image, n=200, per=1):
    brain.sub.reset()
    accum = {L: [] for L in layers}
    for t in range(n):
        d = []
        if drive_image and t % per == 0:
            d = sig
        d = d + brain._tonic
        st = brain.sub.step(d)
        for L in layers:
            accum[L].append(float(st.O[lay == L].mean()))
    return {L: np.array(v) for L, v in accum.items()}

print("layer sizes:", {int(L): int((lay == L).sum()) for L in layers})
print("\nper-layer LATE participation (mean of last 100 ticks):")
print(f"{'condition':28s} " + " ".join(f"L{int(L)}" for L in layers))
for name, di, per in [("tonic only (NO image)", False, 1),
                      ("tonic + image per=1", True, 1),
                      ("tonic + image per=6", True, 6)]:
    r = run(di, per=per)
    print(f"{name:28s} " + " ".join(f"{r[L][-100:].mean():.3f}" for L in layers))

# also: does the image change the pool STATE (S), even if firing fraction matches?
brain.sub.reset()
for t in range(120): brain.sub.step(brain._tonic)         # tonic-only baseline
S_tonic = np.array([n.S for n in brain.sub.sim.network.neurons.values()])
brain.sub.reset()
for t in range(120): brain.sub.step(sig + brain._tonic)   # with image
S_img = np.array([n.S for n in brain.sub.sim.network.neurons.values()])
print(f"\nmean |S(image) - S(tonic)| across all neurons: {np.abs(S_img - S_tonic).mean():.4f}")
print(f"mean S tonic-only {S_tonic.mean():.4f} | mean S with image {S_img.mean():.4f}")
