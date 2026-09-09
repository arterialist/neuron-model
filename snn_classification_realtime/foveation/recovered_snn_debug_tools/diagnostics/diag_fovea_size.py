"""Does a full-image-size fovea recover the class info the small fovea loses?

The single centered glimpse separates at ~0.18 gray vs ~0.29 full image. Is the
loss from (a) the small sharp fovea + blurred periphery, or (b) downsampling to the
12x12 retinal grid? Sweep fovea_extent and grid; full-size fovea + full grid should
approach the full-image ceiling if foveation (not grid) is the culprit.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from snn_classification_realtime.activity_dataset_builder.vision_datasets import (
    load_dataset_by_name,
)
from snn_classification_realtime.foveation.retina import Retina

Cs = (0.0005, 0.002, 0.01, 0.05)
def best_acc(X, ys, ntr):
    X = np.asarray(X, float); mu, sd = X[:ntr].mean(0), X[:ntr].std(0)+1e-6
    Xn = (X-mu)/sd
    return max(LogisticRegression(max_iter=2000, C=C).fit(Xn[:ntr], ys[:ntr]).score(Xn[ntr:], ys[ntr:]) for C in Cs)

for dn in ["cifar10_grayscale", "cifar10_color"]:
    ds = load_dataset_by_name(dn, train=True).dataset
    H = ds[0][0].shape[1]
    N = 1500; rng = np.random.RandomState(1)
    order = rng.randint(0, len(ds), size=N); ntr = int(N*0.7)
    ys = np.array([int(ds[i][1]) for i in order])
    full = np.array([ds[i][0].flatten().numpy() for i in order])
    print(f"\n=== {dn} (N={N}) ===")
    print(f"  full {H}x{H} image (ceiling)            : {best_acc(full, ys, ntr):.3f}")
    configs = [
        ("small fovea8 periph24 grid12 (current)", 8, 24, 12),
        ("fovea16 periph32 grid16",                16, 32, 16),
        ("FULL fovea32 periph32 grid16",           32, 32, 16),
        ("FULL fovea32 periph32 grid32 (=image)",  32, 32, 32),
    ]
    for name, fov, per, grid in configs:
        ret = Retina(H, H, grid=grid, fovea_extent=fov, periph_extent=per); ret.center()
        X = np.array([ret.render(ds[i][0]).flatten().numpy() for i in order])
        print(f"  {name:40s}: {best_acc(X, ys, ntr):.3f}  (dim {X.shape[1]})")
