"""Frozen-substrate separability, done RIGHT (strong regularization for high-dim,
few-sample reps). For grayscale AND color: how much of the input's linear class
info (ceiling ~0.29 gray / ~0.37 color) does the reservoir readout preserve?

Reports best-over-C test accuracy per layer + the raw-retina input in the same eval.
"""
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from snn_classification_realtime.activity_dataset_builder.vision_datasets import (
    load_dataset_by_name,
)
from snn_classification_realtime.foveation.minibrain.core import MiniBrainConfig, MiniBrain

Cs = (0.0005, 0.002, 0.01, 0.05)

def best_acc(X, ys, ntr):
    X = np.asarray(X, float)
    mu, sd = X[:ntr].mean(0), X[:ntr].std(0) + 1e-6
    Xn = (X - mu) / sd
    best = 0.0
    for C in Cs:
        clf = LogisticRegression(max_iter=3000, C=C)
        clf.fit(Xn[:ntr], ys[:ntr])
        best = max(best, clf.score(Xn[ntr:], ys[ntr:]))
    return best

def run(dataset_name, N=1500):
    ds_cfg = load_dataset_by_name(dataset_name, train=True)
    ds = ds_cfg.dataset
    cfg = MiniBrainConfig(dataset_name=dataset_name, tonic_drive=0.15,
                          target_participation=0.05, dwell=40, seed=0)
    brain = MiniBrain(cfg, ds_cfg)
    lay = brain.sub.layer_of_pos
    layers = sorted(set(lay.tolist()))
    brain.sub.set_learning(False)
    rng = np.random.RandomState(1)
    order = rng.randint(0, len(ds), size=N)
    per = max(1, cfg.input_period)
    feats = {L: [] for L in layers}; feats["all"] = []; feats["retina"] = []
    ys = []
    for idx in order:
        img, y = ds[idx]; ys.append(int(y))
        r = brain.retina.render(img)
        feats["retina"].append(r.flatten().numpy())
        sig = brain.sub.patch_to_signals(r)
        states = []
        for t in range(cfg.dwell):
            d = (sig if t % per == 0 else []) + brain._tonic
            states.append(brain.sub.step(d))
        S = np.mean([s.S for s in states], 0); F = np.mean([s.F_avg for s in states], 0)
        O = np.mean([s.O for s in states], 0)
        for L in layers:
            mk = lay == L
            feats[L].append(np.concatenate([S[mk], F[mk], O[mk]]))
        feats["all"].append(np.concatenate([S, F, O]))
    ys = np.array(ys); ntr = int(N * 0.7)
    nm = {0: "input", 1: "buffer", 2: "pool"}
    print(f"\n=== {dataset_name} (frozen substrate, N={N}, best-over-C test acc) ===")
    print(f"  retina input (ceiling)  : {best_acc(feats['retina'], ys, ntr):.3f}")
    for L in layers:
        print(f"  layer {L} {nm.get(L,''):6s} ({(lay==L).sum():3d}u): {best_acc(feats[L], ys, ntr):.3f}")
    print(f"  all layers concat       : {best_acc(feats['all'], ys, ntr):.3f}")

for dn in ["cifar10_grayscale", "cifar10_color"]:
    run(dn)
