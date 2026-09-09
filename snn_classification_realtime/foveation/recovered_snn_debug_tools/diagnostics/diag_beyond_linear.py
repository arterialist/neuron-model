"""Two questions:
(1) Is fovea32/grid32 retina really ABOVE the full-image linear ceiling, or noise?
    -> multi-seed split; report mean +/- std. Theory says the retina (an affine map
       of pixels) cannot exceed the image's linear separability.
(2) How much separability lies BEYOND a linear probe? -> nonlinear MLP probe on the
    same pixels. If it's far above ~0.31, the linear ceiling is not the info ceiling
    and the substrate+teacher's job is to expose that nonlinear structure to a
    simple readout.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from snn_classification_realtime.activity_dataset_builder.vision_datasets import (
    load_dataset_by_name,
)
from snn_classification_realtime.foveation.retina import Retina

ds = load_dataset_by_name("cifar10_color", train=True).dataset
H = ds[0][0].shape[1]
N = 2500
Cs = (0.0005, 0.002, 0.01, 0.05)

def lin_best(X, ys, tr, te):
    mu, sd = X[tr].mean(0), X[tr].std(0)+1e-6; Xn = (X-mu)/sd
    return max(LogisticRegression(max_iter=2000, C=C).fit(Xn[tr], ys[tr]).score(Xn[te], ys[te]) for C in Cs)

def build(order):
    full = np.array([ds[i][0].flatten().numpy() for i in order])
    ret = Retina(H, H, grid=32, fovea_extent=32, periph_extent=32); ret.center()
    retX = np.array([ret.render(ds[i][0]).flatten().numpy() for i in order])
    ys = np.array([int(ds[i][1]) for i in order])
    return full, retX, ys

# (1) multi-seed linear: full image vs full-fovea retina
print("(1) LINEAR separability, 5 seeds (mean +/- std), color N=%d:" % N)
fu, re, _ = None, None, None
accs_full, accs_ret = [], []
for s in range(5):
    rng = np.random.RandomState(s)
    order = rng.randint(0, len(ds), size=N)
    full, retX, ys = build(order)
    idx = rng.permutation(N); tr, te = idx[:int(N*0.7)], idx[int(N*0.7):]
    accs_full.append(lin_best(full, ys, tr, te))
    accs_ret.append(lin_best(retX, ys, tr, te))
    fu, re = full, retX  # keep last for nonlinear probe
    ys_last = ys; tr_last, te_last = tr, te
print(f"   full 32x32 image      : {np.mean(accs_full):.3f} +/- {np.std(accs_full):.3f}")
print(f"   full-fovea grid32 retina: {np.mean(accs_ret):.3f} +/- {np.std(accs_ret):.3f}")
print(f"   delta {np.mean(accs_ret)-np.mean(accs_full):+.3f}  "
      f"(theory: retina is an affine map of pixels -> cannot exceed; expect ~0)")

# (2) nonlinear probe on the full image (true separability beyond linear)
print("\n(2) NONLINEAR probe (MLP) on full color image:")
mu, sd = fu[tr_last].mean(0), fu[tr_last].std(0)+1e-6; Xn = (fu-mu)/sd
for hid in [(128,), (256, 128)]:
    clf = MLPClassifier(hidden_layer_sizes=hid, max_iter=60, alpha=1e-3,
                        early_stopping=True, random_state=0)
    clf.fit(Xn[tr_last], ys_last[tr_last])
    print(f"   MLP{hid}: test acc {clf.score(Xn[te_last], ys_last[te_last]):.3f}")
print("   (linear ceiling ~0.31 -> the gap is the nonlinear class structure the")
print("    substrate+teacher must expose to a simple readout)")
