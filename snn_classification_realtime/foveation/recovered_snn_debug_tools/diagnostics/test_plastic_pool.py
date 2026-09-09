"""Should the pool be frozen or plastic? The decay-spiral that motivated freezing
predates tonic drive + input-ignition. Re-test now.

For freeze_reservoir in {True, False}: run a teacher-ON stream, then measure
  (a) pool aliveness over time (does it die / rail?)
  (b) frozen-eval pool separability BEFORE vs AFTER the run (does teacher plasticity
      RAISE the readout's class info toward the 0.18 retina ceiling?)
Grayscale, N reasonable for speed.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from snn_classification_realtime.activity_dataset_builder.vision_datasets import (
    load_dataset_by_name,
)
from snn_classification_realtime.foveation.minibrain.core import MiniBrainConfig, MiniBrain

ds_cfg = load_dataset_by_name("cifar10_grayscale", train=True)
ds = ds_cfg.dataset
Cs = (0.0005, 0.002, 0.01, 0.05)

def best_acc(X, ys):
    X = np.asarray(X, float); n = len(ys); ntr = int(n*0.7)
    mu, sd = X[:ntr].mean(0), X[:ntr].std(0)+1e-6; Xn = (X-mu)/sd
    return max(LogisticRegression(max_iter=2000, C=C).fit(Xn[:ntr], ys[:ntr]).score(Xn[ntr:], ys[ntr:]) for C in Cs)

def probe_pool_sep(brain, idxs):
    """Frozen-eval pool readout separability over idxs."""
    brain.sub.set_learning(False)
    m = brain._readout_mask; per = max(1, brain.cfg.input_period)
    X, Y = [], []
    for i in idxs:
        img, y = ds[i]; Y.append(int(y))
        sig = brain.sub.patch_to_signals(brain.retina.render(img))
        st = [brain.sub.step((sig if t % per == 0 else []) + brain._tonic) for t in range(brain.cfg.dwell)]
        S = np.mean([s.S[m] for s in st],0); F=np.mean([s.F_avg[m] for s in st],0); O=np.mean([s.O[m] for s in st],0)
        X.append(np.concatenate([S,F,O]))
    brain.sub.set_learning(True)
    return best_acc(X, np.array(Y))

rng = np.random.RandomState(0)
probe_idx = rng.randint(0, len(ds), size=600)
train_idx = rng.randint(0, len(ds), size=1200)

for freeze in (True, False):
    cfg = MiniBrainConfig(dataset_name="cifar10_grayscale", tonic_drive=0.15,
                          target_participation=0.05, dwell=40, seed=0,
                          freeze_reservoir=freeze)
    brain = MiniBrain(cfg, ds_cfg)
    m = brain._readout_mask
    sep_before = probe_pool_sep(brain, probe_idx)
    # teacher-ON training stream
    alive = []
    for j, idx in enumerate(train_idx):
        img, y = ds[idx]
        x, _ = brain.present(img, int(y), learn=True, teach=True)
        if (j+1) % 200 == 0:
            # aliveness: pool participation on a quick continuous probe
            sig = brain.sub.patch_to_signals(brain.retina.render(img))
            frac = np.mean([brain.sub.step(sig + brain._tonic).O[m].mean() for _ in range(20)])
            alive.append(frac)
    eff_pool = np.mean([brain.sub.sim.network.neurons[nid].postsynaptic_points[sid].u_i.info
                        for nid in brain.sub._ids
                        for sid in brain.sub.sim.network.neurons[nid].postsynaptic_points
                        if int(brain.sub.sim.network.neurons[nid].metadata.get("layer",0)) == 2])
    sep_after = probe_pool_sep(brain, probe_idx)
    print(f"\n=== freeze_reservoir={freeze} ===")
    print(f"  pool separability  BEFORE {sep_before:.3f}  ->  AFTER {sep_after:.3f}  "
          f"(retina ceiling 0.18)")
    print(f"  pool aliveness over training (every 200): "
          f"{['%.3f'%a for a in alive]}")
    print(f"  mean pool synapse efficacy after: {eff_pool:.3f}")
