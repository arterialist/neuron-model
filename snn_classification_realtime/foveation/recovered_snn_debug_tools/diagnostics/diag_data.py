"""Sanity: is the mini-brain's dataset degraded? Compare its linear separability to
a clean torchvision CIFAR-10 grayscale baseline. Inspect value ranges + transforms."""
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from snn_classification_realtime.activity_dataset_builder.vision_datasets import (
    load_dataset_by_name,
)

ds_cfg = load_dataset_by_name("cifar10_grayscale", train=True)
ds = ds_cfg.dataset
img0, y0 = ds[0]
print("mini-brain ds[0]: shape", tuple(img0.shape), "dtype", img0.dtype,
      "range", (float(img0.min()), float(img0.max())), "mean", float(img0.mean()))
print("dataset object type:", type(ds).__name__)
for attr in ("transform", "transforms", "dataset"):
    if hasattr(ds, attr):
        print(f"  ds.{attr} = {getattr(ds, attr)}")

def evalX(X, ys, ntr, C=0.02):
    X = np.asarray(X, float)
    mu, sd = X[:ntr].mean(0), X[:ntr].std(0) + 1e-6
    Xn = (X - mu) / sd
    clf = LogisticRegression(max_iter=3000, C=C)
    clf.fit(Xn[:ntr], ys[:ntr]); return clf.score(Xn[ntr:], ys[ntr:])

N = 3000
rng = np.random.RandomState(1)
order = rng.randint(0, len(ds), size=N)
ys = np.array([int(ds[i][1]) for i in order]); ntr = int(N*0.7)
X = np.array([ds[i][0].flatten().numpy() for i in order])
print(f"\nmini-brain grayscale full-image, C sweep (N={N}):")
for C in (0.001, 0.01, 0.05, 0.2, 1.0):
    print(f"   C={C:<6}: {evalX(X, ys, ntr, C):.3f}")

# clean torchvision reference (same roots the loader uses)
import torchvision
from snn_classification_realtime.activity_dataset_builder.vision_datasets import _ROOT_CANDIDATES
tv = None
for root in _ROOT_CANDIDATES:
    try:
        tv = torchvision.datasets.CIFAR10(root=root, train=True, download=True); break
    except Exception:
        continue
Xtv = tv.data[order].astype(np.float32)           # N,32,32,3 uint8
Xg = Xtv.mean(3).reshape(N, -1) / 255.0            # grayscale
ytv = np.array(tv.targets)[order]
print(f"\nclean torchvision grayscale full-image, C sweep:")
for C in (0.001, 0.01, 0.05, 0.2):
    print(f"   C={C:<6}: {evalX(Xg, ytv, ntr, C):.3f}")
print("labels match mini-brain?", bool((ytv == ys).mean() > 0.99), f"(agree {float((ytv==ys).mean()):.2f})")

# clean torchvision RGB (color) reference
Xrgb = Xtv.reshape(N, -1) / 255.0                  # N, 32*32*3
print(f"\nclean torchvision RGB (color) full-image, C sweep:")
for C in (0.001, 0.01, 0.05, 0.2):
    print(f"   C={C:<6}: {evalX(Xrgb, ytv, ntr, C):.3f}")

# mini-brain's own color dataset (cifar10 color) pipeline
dc2 = load_dataset_by_name("cifar10", train=True)
ds2 = dc2.dataset
i0, _ = ds2[0]
print(f"\nmini-brain color ds[0]: shape {tuple(i0.shape)} range "
      f"({float(i0.min()):.2f},{float(i0.max()):.2f})")
Xc = np.array([ds2[i][0].flatten().numpy() for i in order])
yc = np.array([int(ds2[i][1]) for i in order])
print(f"mini-brain color full-image, C sweep:")
for C in (0.001, 0.01, 0.05, 0.2):
    print(f"   C={C:<6}: {evalX(Xc, yc, ntr, C):.3f}")
