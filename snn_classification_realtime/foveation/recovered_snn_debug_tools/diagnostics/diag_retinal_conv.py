"""Do RETINAL CONVOLUTIONS raise the separability ceiling above the raw glimpse?

Biological retina/LGN: center-surround (DoG) + oriented edge receptive fields, then
ON/OFF half-wave RECTIFICATION (the nonlinearity). Convolution alone is linear (can't
raise linear separability); rectification is what exposes nonlinear structure to a
linear probe. Measure: raw retina vs retinal-conv features, grayscale + color.
"""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from snn_classification_realtime.activity_dataset_builder.vision_datasets import (
    load_dataset_by_name,
)
from snn_classification_realtime.foveation.retina import Retina

Cs = (0.0005, 0.002, 0.01, 0.05)
def best_acc(X, ys, ntr):
    X = np.asarray(X, float); mu, sd = X[:ntr].mean(0), X[:ntr].std(0)+1e-6; Xn=(X-mu)/sd
    return max(LogisticRegression(max_iter=2000, C=C).fit(Xn[:ntr], ys[:ntr]).score(Xn[ntr:], ys[ntr:]) for C in Cs)

def _dog(sigma1, sigma2, k=5):
    ax = np.arange(k) - k // 2
    xx, yy = np.meshgrid(ax, ax)
    g1 = np.exp(-(xx**2+yy**2)/(2*sigma1**2)); g1/=g1.sum()
    g2 = np.exp(-(xx**2+yy**2)/(2*sigma2**2)); g2/=g2.sum()
    return g1 - g2

def _kernels():
    ks = [_dog(0.6, 1.2), _dog(1.0, 2.0)]                       # center-surround, 2 scales
    sob = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], float)
    ks += [sob, sob.T, np.rot90(sob), np.rot90(sob).T]          # 4 oriented edges (pad to 5x5)
    out = []
    for k in ks:
        if k.shape[0] == 3:
            kk = np.zeros((5,5)); kk[1:4,1:4] = k; k = kk
        out.append(k)
    return np.stack(out)   # (F,5,5)

KER = torch.tensor(_kernels(), dtype=torch.float32).unsqueeze(1)  # (F,1,5,5)

def retinal_conv(img_2ck):
    """img: (C, grid, grid) -> ON/OFF rectified conv features flattened."""
    x = img_2ck.unsqueeze(1)                       # (C,1,H,W) treat channels as batch
    y = F.conv2d(x, KER, padding=2)                # (C,F,H,W)
    on = F.relu(y); off = F.relu(-y)               # ON / OFF rectification
    # light 2x2 avg pool (bipolar->ganglion pooling) to add spatial invariance
    on = F.avg_pool2d(on, 2); off = F.avg_pool2d(off, 2)
    return torch.cat([on.flatten(), off.flatten()]).numpy()

for dn in ["cifar10_grayscale", "cifar10_color"]:
    ds = load_dataset_by_name(dn, train=True).dataset
    H = ds[0][0].shape[1]
    N = 2000; rng = np.random.RandomState(1)
    order = rng.randint(0, len(ds), size=N); ntr = int(N*0.7)
    ys = np.array([int(ds[i][1]) for i in order])
    ret = Retina(H, H, grid=12, fovea_extent=8, periph_extent=24); ret.center()
    raw = np.array([ret.render(ds[i][0]).flatten().numpy() for i in order])
    conv = np.array([retinal_conv(ret.render(ds[i][0])) for i in order])
    # also full-image retinal conv (fovea32 grid16) to see conv on max info
    retF = Retina(H, H, grid=16, fovea_extent=32, periph_extent=32); ret.center()
    convF = np.array([retinal_conv(retF.render(ds[i][0])) for i in order])
    print(f"\n=== {dn} (N={N}) ===")
    print(f"  raw small-glimpse retina         : {best_acc(raw, ys, ntr):.3f}  (dim {raw.shape[1]})")
    print(f"  retinal-CONV small glimpse       : {best_acc(conv, ys, ntr):.3f}  (dim {conv.shape[1]})")
    print(f"  retinal-CONV full fovea32 grid16 : {best_acc(convF, ys, ntr):.3f}  (dim {convF.shape[1]})")
