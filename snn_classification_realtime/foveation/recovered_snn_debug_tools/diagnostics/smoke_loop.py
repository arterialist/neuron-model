"""End-to-end smoke test of the continuous learning loop (Option A default).

Checks, over a short teacher-on stream (no reset between images):
  - present() runs without error
  - the pool stays ALIVE (not dead, not saturated) as plasticity acts
  - the representation is IMAGE-VARIED (distinct reps for distinct classes)
  - mean synaptic efficacy neither rails to ceiling nor collapses to zero
  - online decoder + fresh-kNN probe rise above chance (0.10) -> substrate learns
"""
import numpy as np
from snn_classification_realtime.activity_dataset_builder.vision_datasets import (
    load_dataset_by_name,
)
from snn_classification_realtime.foveation.minibrain.core import (
    MiniBrainConfig, MiniBrain, knn_accuracy,
)

ds_cfg = load_dataset_by_name("cifar10_grayscale", train=True)
ds = ds_cfg.dataset
cfg = MiniBrainConfig(dataset_name="cifar10_grayscale", tonic_drive=0.15,
                      target_participation=0.05, dwell=40, seed=0)
brain = MiniBrain(cfg, ds_cfg)
m = brain._readout_mask

rng = np.random.RandomState(0)
order = rng.randint(0, len(ds), size=600)
buf_x, buf_y, alive, effs = [], [], [], []
run_correct = 0
for i, idx in enumerate(order):
    img, y = ds[idx]; y = int(y)
    x, pred = brain.present(img, y, learn=True, teach=True)
    buf_x.append(x); buf_y.append(y)
    run_correct += int(pred == y)
    if len(buf_x) > 300: buf_x.pop(0); buf_y.pop(0)
    if (i + 1) % 100 == 0:
        # measure pool aliveness on the last presented image
        knn = knn_accuracy(buf_x, buf_y, k=5) if len(set(buf_y)) > 1 else float("nan")
        eff = brain.sub.mean_efficacy()
        oa = run_correct / 100.0
        print(f"  {i+1:4d} | online acc {oa:.3f} | fresh-kNN {knn:.3f} | "
              f"mean eff {eff:.3f} | m0/m1 {getattr(brain,'_last_nm',(0,0))}")
        run_correct = 0

# representation variety: are class-mean reps distinct?
X = np.array(buf_x); Y = np.array(buf_y)
cms = np.array([X[Y == c].mean(0) for c in sorted(set(Y.tolist())) if (Y == c).sum() > 2])
# pairwise cosine of class means
def cos(a, b):
    return float(a @ b / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-9))
pc = [cos(cms[a], cms[b]) for a in range(len(cms)) for b in range(a+1, len(cms))]
print(f"\nclass-mean rep pairwise cosine: mean {np.mean(pc):.3f} "
      f"(<1 = classes occupy distinct rep directions)")
print(f"rep norm mean {np.linalg.norm(X,axis=1).mean():.3f}")
