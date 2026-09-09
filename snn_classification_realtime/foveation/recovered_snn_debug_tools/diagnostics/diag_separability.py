"""Is the substrate readout linearly separable AT ALL (frozen, no teacher)?

Chicken-and-egg in the loop hides whether the bottleneck is the SUBSTRATE or the
LEARNING. Freeze the substrate, present N labeled images (NO plasticity, NO teacher),
collect the [S|F|O] rep of EACH layer, and train a proper offline linear classifier
with a held-out split. If even that is at chance, the reservoir readout carries no
class info and the substrate must change; if it separates, the loop dynamics are the
problem.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from snn_classification_realtime.activity_dataset_builder.vision_datasets import (
    load_dataset_by_name,
)
from snn_classification_realtime.foveation.minibrain.core import MiniBrainConfig, MiniBrain

ds_cfg = load_dataset_by_name("cifar10_grayscale", train=True)
ds = ds_cfg.dataset
cfg = MiniBrainConfig(dataset_name="cifar10_grayscale", tonic_drive=0.15,
                      target_participation=0.05, dwell=40, seed=0)
brain = MiniBrain(cfg, ds_cfg)
lay = brain.sub.layer_of_pos
layers = sorted(set(lay.tolist()))

def rep_all_layers(states):
    """Return dict layer-> mean [S|F|O] over the dwell for that layer's neurons."""
    S = np.mean([s.S for s in states], 0)
    F = np.mean([s.F_avg for s in states], 0)
    O = np.mean([s.O for s in states], 0)
    out = {}
    for L in layers:
        mk = lay == L
        out[L] = np.concatenate([S[mk], F[mk], O[mk]]).astype(np.float64)
    out["all"] = np.concatenate([S, F, O]).astype(np.float64)
    return out

brain.sub.set_learning(False)                 # FROZEN substrate, no plasticity
rng = np.random.RandomState(1)
N = 800
order = rng.randint(0, len(ds), size=N)
per = max(1, cfg.input_period)
feats = {L: [] for L in layers}; feats["all"] = []
ys = []
for idx in order:
    img, y = ds[idx]; ys.append(int(y))
    sig = brain.sub.patch_to_signals(brain.retina.render(img))
    states = []
    for t in range(cfg.dwell):
        d = (sig if t % per == 0 else []) + brain._tonic
        states.append(brain.sub.step(d))
    r = rep_all_layers(states)
    for k in feats: feats[k].append(r[k])
ys = np.array(ys)

ntr = int(N * 0.7)
def evalL(X):
    X = np.asarray(X)
    mu, sd = X[:ntr].mean(0), X[:ntr].std(0) + 1e-6
    Xn = (X - mu) / sd
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xn[:ntr], ys[:ntr])
    return clf.score(Xn[ntr:], ys[ntr:])

print(f"frozen-substrate OFFLINE linear separability (test acc, chance=0.10), N={N}:")
name = {0: "input", 1: "buffer", 2: "pool"}
for L in layers:
    print(f"  layer {L} ({name.get(L,L):6s}, {(lay==L).sum():3d} units): {evalL(feats[L]):.3f}")
print(f"  all layers concatenated              : {evalL(feats['all']):.3f}")

# also: raw retina input (skip substrate) as an upper-ish reference
Xr = []
for idx in order:
    img, y = ds[idx]
    Xr.append(brain.retina.render(img).flatten().numpy())
print(f"\n  raw retina pixels (reference)        : {evalL(np.array(Xr)):.3f}")
