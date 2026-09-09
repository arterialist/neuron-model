"""Where is the class info lost? Compare linear separability of:
  (a) full 32x32 grayscale CIFAR image (true ceiling)
  (b) single CENTERED foveated glimpse (current mini-brain input)
  (c) a few glimpses at different fixation points, concatenated (proxy for gaze)
This isolates whether the retina/central-glimpse is the bottleneck (-> need gaze).
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from snn_classification_realtime.activity_dataset_builder.vision_datasets import (
    load_dataset_by_name,
)
from snn_classification_realtime.foveation.retina import Retina

ds_cfg = load_dataset_by_name("cifar10_grayscale", train=True)
ds = ds_cfg.dataset
N = 1500
rng = np.random.RandomState(1)
order = rng.randint(0, len(ds), size=N)
ys = np.array([int(ds[i][1]) for i in order])
ntr = int(N * 0.7)

def evalX(X):
    X = np.asarray(X, float)
    mu, sd = X[:ntr].mean(0), X[:ntr].std(0) + 1e-6
    Xn = (X - mu) / sd
    clf = LogisticRegression(max_iter=2000, C=0.5)
    clf.fit(Xn[:ntr], ys[:ntr])
    return clf.score(Xn[ntr:], ys[ntr:])

# (a) full image
full = np.array([ds[i][0].flatten().numpy() for i in order])
print(f"(a) full 32x32 grayscale image        : {evalX(full):.3f}   (true info ceiling)")

# (b) single centered foveated glimpse
H = ds[0][0].shape[1]
ret = Retina(H, H, grid=12, fovea_extent=8, periph_extent=24); ret.center()
cen = np.array([ret.render(ds[i][0]).flatten().numpy() for i in order])
print(f"(b) single CENTERED foveated glimpse   : {evalX(cen):.3f}   (current mini-brain input)")

# (b2) bigger fovea / bigger grid — does a sharper glimpse help?
ret2 = Retina(H, H, grid=16, fovea_extent=16, periph_extent=32); ret2.center()
cen2 = np.array([ret2.render(ds[i][0]).flatten().numpy() for i in order])
print(f"(b2) centered, fovea16 grid16          : {evalX(cen2):.3f}")

# (c) multi-fixation: 5 glimpses (center + 4 quadrants), concatenated
pts = [(0.5,0.5),(0.3,0.3),(0.3,0.7),(0.7,0.3),(0.7,0.7)]
def multi(i):
    parts = []
    img = ds[i][0]
    for fx, fy in pts:
        ret.set_center(fy * H, fx * H)
        parts.append(ret.render(img).flatten().numpy())
    return np.concatenate(parts)
mg = np.array([multi(i) for i in order])
print(f"(c) 5-fixation concatenation (gaze proxy): {evalX(mg):.3f}   (what learned gaze could reach)")
