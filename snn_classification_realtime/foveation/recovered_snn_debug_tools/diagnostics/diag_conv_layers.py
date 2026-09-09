"""With retinal conv ON, where does the conv headroom (features ~0.34 gray) survive?
Frozen substrate, per-layer [S|F|O] separability. If input/buffer >> pool, the
reservoir is the bottleneck and we should read a shallower layer (or fix the pool)."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from snn_classification_realtime.activity_dataset_builder.vision_datasets import (
    load_dataset_by_name,
)
from snn_classification_realtime.foveation.minibrain.core import MiniBrainConfig, MiniBrain

Cs = (0.0005, 0.002, 0.01, 0.05)
def best_acc(X, ys, ntr):
    X = np.asarray(X, float); mu, sd = X[:ntr].mean(0), X[:ntr].std(0)+1e-6; Xn=(X-mu)/sd
    return max(LogisticRegression(max_iter=1500, C=C).fit(Xn[:ntr], ys[:ntr]).score(Xn[ntr:], ys[ntr:]) for C in Cs)

for dn in ["cifar10_grayscale", "cifar10_color"]:
    ds_cfg = load_dataset_by_name(dn, train=True); ds = ds_cfg.dataset
    cfg = MiniBrainConfig(dataset_name=dn, retinal_conv=True, tonic_drive=0.15,
                          target_participation=0.05, dwell=40, seed=0)
    brain = MiniBrain(cfg, ds_cfg)
    lay = brain.sub.layer_of_pos; layers = sorted(set(int(x) for x in lay))
    brain.sub.set_learning(False)
    N = 1000; rng = np.random.RandomState(1); order = rng.randint(0, len(ds), size=N)
    per = max(1, cfg.input_period)
    feats = {L: [] for L in layers}; feats["conv_in"] = []; feats["all"] = []; ys = []
    for idx in order:
        img, y = ds[idx]; ys.append(int(y))
        enc = brain._encode(img)
        feats["conv_in"].append(enc.flatten().numpy())
        sig = brain.sub.patch_to_signals(enc)
        st = [brain.sub.step((sig if t % per == 0 else []) + brain._tonic) for t in range(cfg.dwell)]
        S=np.mean([s.S for s in st],0); F=np.mean([s.F_avg for s in st],0); O=np.mean([s.O for s in st],0)
        for L in layers:
            mk = lay == L; feats[L].append(np.concatenate([S[mk], F[mk], O[mk]]))
        feats["all"].append(np.concatenate([S, F, O]))
    ys = np.array(ys); ntr = int(N*0.7)
    nm = {layers[0]:"input", layers[1] if len(layers)>1 else -1:"buffer", layers[-1]:"pool"}
    print(f"\n=== {dn} (conv ON, frozen, N={N}, n_in={cfg.n_in}) ===")
    print(f"  conv features (raw)     : {best_acc(feats['conv_in'], ys, ntr):.3f}")
    for L in layers:
        print(f"  layer {L} {nm.get(L,''):6s} ({(lay==L).sum():3d}u): {best_acc(feats[L], ys, ntr):.3f}")
    print(f"  ALL layers concat ({brain.n_neurons}u): {best_acc(feats['all'], ys, ntr):.3f}")
