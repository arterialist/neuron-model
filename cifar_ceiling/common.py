"""Shared apparatus for the CIFAR10 fixed-feature ceiling + learned-feature ladder experiments.

Everything that must be IDENTICAL across arms lives here, so that an arm difference can only
come from the intended manipulation (how the dictionary D is obtained), never from a stray
preprocessing/encoding/readout difference.

The canonical pipeline is Coates & Ng (2011), "An Analysis of Single-Layer Networks in
Unsupervised Feature Learning":
    patches -> per-patch contrast normalization -> ZCA whitening -> dictionary (K x d)
            -> triangle encoding -> 2x2 quadrant sum-pooling -> L2-regularized linear readout

PIXEL SCALE IS LOAD-BEARING. Coates' constants (contrast-norm eps=10, ZCA eps=0.1) are tuned
for pixels in [0, 255]. Under torchvision's ToTensor() the range is [0, 1], where patch
variance ~ 0.01, so sqrt(var + 10) ~ 3.16 -- a near-constant divisor. Contrast normalization
silently becomes a no-op and ZCA is over-damped. Default here is scale=255 for that reason;
scale=1 is retained as an explicit ablation lever because it is the suspected cause of the
63% we measured against the paper's ~71% at matched K.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

# Published Coates & Ng (2011) k-means + triangle accuracy vs dictionary size, CIFAR10 test.
# Used ONLY as an instrument-calibration target: if our pipeline is far below this at matched
# K, the pipeline is broken and any K-sweep reading is meaningless. Not a result of ours.
COATES_NG_2011_REFERENCE = {100: 64.6, 200: 68.9, 400: 71.0, 800: 76.5, 1200: 77.9, 1600: 79.6}


def pick_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# --------------------------------------------------------------------------- data

def load_cifar(root: str = "data", pixel_scale: float = 255.0):
    """Return Xtr (N,3,32,32) float32 in [0, pixel_scale], ytr, Xte, yte. No augmentation,
    no normalization beyond the scale -- all preprocessing is explicit downstream."""
    import torchvision
    import torchvision.transforms as T

    tf = T.ToTensor()  # [0,1]
    tr = torchvision.datasets.CIFAR10(root, train=True, download=False, transform=tf)
    te = torchvision.datasets.CIFAR10(root, train=False, download=False, transform=tf)
    Xtr = torch.stack([tr[i][0] for i in range(len(tr))]).numpy().astype(np.float32) * pixel_scale
    Xte = torch.stack([te[i][0] for i in range(len(te))]).numpy().astype(np.float32) * pixel_scale
    return Xtr, np.asarray(tr.targets), Xte, np.asarray(te.targets)


def train_val_split(n: int, val_frac: float, seed: int):
    """Disjoint train/val index split. The TEST set is never involved in any fitting or
    hyperparameter choice -- it is touched exactly once, at the end of an arm."""
    if n < 2 or not 0.0 < val_frac < 1.0:
        raise ValueError("a train/validation split needs at least two examples and 0 < val_frac < 1")
    nval = int(round(n * val_frac))
    if not 0 < nval < n:
        raise ValueError("train and validation splits must both contain examples")
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    va, tr = perm[:nval], perm[nval:]
    if np.intersect1d(tr, va).size:
        raise RuntimeError("train/validation split overlap")
    return tr, va


# --------------------------------------------------------------------------- patches

def extract_patches(X: np.ndarray, P: int, stride: int = 1) -> np.ndarray:
    """(B,C,H,W) -> (B*L*L, C*P*P) with L = (H-P)//stride + 1, patch-major per image."""
    t = torch.from_numpy(X)
    u = Fn.unfold(t, P, stride=stride)             # (B, C*P*P, L*L)
    return u.permute(0, 2, 1).reshape(-1, u.shape[1]).numpy()


def sample_patches(X: np.ndarray, n_patches: int, P: int, seed: int,
                   n_images: int = 10000) -> np.ndarray:
    """Random patches from random TRAIN images (dictionary-learning input)."""
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), min(n_images, len(X)), replace=False)
    pw = extract_patches(X[idx], P, stride=1)
    sel = rng.choice(len(pw), min(n_patches, len(pw)), replace=False)
    return pw[sel]


def contrast_normalize(p: np.ndarray, eps: float = 10.0, enabled: bool = True) -> np.ndarray:
    """Per-patch brightness/contrast normalization. eps=10 assumes [0,255] pixels."""
    if not enabled:
        return p
    return (p - p.mean(1, keepdims=True)) / np.sqrt(p.var(1, keepdims=True) + eps)


def fit_zca(p: np.ndarray, eps: float = 0.1):
    """Fit ZCA whitening on (n,d) patches -> (mean, W) with whitened = (p - mean) @ W."""
    mean = p.mean(0)
    pc = p - mean
    cov = np.cov(pc, rowvar=False)
    U, S, _ = np.linalg.svd(cov)
    W = (U @ np.diag(1.0 / np.sqrt(S + eps)) @ U.T).astype(np.float32)
    return mean.astype(np.float32), W


class Preproc:
    """The exact, arm-invariant patch preprocessing. Fit on TRAIN patches only."""

    def __init__(self, P: int, cn_eps: float, cn_on: bool, zca_eps: float, zca_on: bool):
        self.P, self.cn_eps, self.cn_on, self.zca_eps, self.zca_on = P, cn_eps, cn_on, zca_eps, zca_on
        self.mean = None
        self.W = None

    def fit(self, patches: np.ndarray) -> "Preproc":
        p = contrast_normalize(patches, self.cn_eps, self.cn_on)
        if self.zca_on:
            self.mean, self.W = fit_zca(p, self.zca_eps)
        else:
            self.mean, self.W = p.mean(0).astype(np.float32), np.eye(p.shape[1], dtype=np.float32)
        return self

    def apply(self, patches: np.ndarray) -> np.ndarray:
        p = contrast_normalize(patches, self.cn_eps, self.cn_on)
        return ((p - self.mean) @ self.W).astype(np.float32)

    def torch_apply(self, p: torch.Tensor) -> torch.Tensor:
        """Differentiable/GPU version of apply() for (n,d) patch tensors."""
        if self.cn_on:
            p = (p - p.mean(1, keepdim=True)) / torch.sqrt(p.var(1, keepdim=True, unbiased=False) + self.cn_eps)
        mean = torch.as_tensor(self.mean, device=p.device, dtype=p.dtype)
        W = torch.as_tensor(self.W, device=p.device, dtype=p.dtype)
        return (p - mean) @ W


# --------------------------------------------------------------------------- dictionaries

def kmeans_lloyd(X: np.ndarray, K: int, iters: int, seed: int, device, batch: int = 100_000,
                 verbose: bool = True) -> np.ndarray:
    """Hard-assignment Lloyd k-means, GPU-batched, matching Coates' released run_kmeans.m:
    randn*0.1 init, mean update, random re-init of empty clusters. (The paper calls this
    'spherical k-means'; the released code is plain Lloyd -- we follow the code.)"""
    rng = np.random.RandomState(seed)
    Xt = torch.as_tensor(X, device=device)
    D = torch.as_tensor(rng.randn(K, X.shape[1]).astype(np.float32) * 0.1, device=device)
    for it in range(iters):
        c2 = 0.5 * (D * D).sum(1)                                   # (K,)
        summ = torch.zeros_like(D)
        counts = torch.zeros(K, device=device)
        loss = 0.0
        for i in range(0, len(Xt), batch):
            xb = Xt[i:i + batch]
            lab = (xb @ D.T - c2).argmax(1)                         # (n,)
            summ.index_add_(0, lab, xb)
            counts.index_add_(0, lab, torch.ones(len(xb), device=device))
            loss += float(((xb * xb).sum(1) * 0.5).sum())
        empty = counts == 0
        D = summ / counts.clamp(min=1).unsqueeze(1)
        if empty.any():  # re-seed dead centroids on random data points (Coates zeroes them)
            ridx = torch.as_tensor(rng.choice(len(Xt), int(empty.sum())), device=device)
            D[empty] = Xt[ridx]
        if verbose:
            print(f"    kmeans iter {it + 1}/{iters} empty={int(empty.sum())}", flush=True)
    return D.cpu().numpy()


def krotov_hopfield(X: np.ndarray, K: int, epochs: int, seed: int, device,
                    p: float = 2.0, k_rank: int = 2, delta: float = 0.4,
                    eps0: float = 2e-2, batch: int = 100, prec: float = 1e-30,
                    verbose: bool = True) -> np.ndarray:
    """Krotov & Hopfield (2019, PNAS) 'Unsupervised learning by competing hidden units'.

    A purely LOCAL, unsupervised, competitive Hebbian rule -- no labels, no backprop, no
    global loss. Rank-based competition: the winning unit is potentiated, the k-th ranked
    unit is depressed by -delta, all others are unchanged (a soft kWTA with anti-Hebbian
    runner-up). This is the arm that maps onto the proposed PAULA model-change #1 (lateral
    inhibition / kWTA), so its measured gain over `random` is the payoff estimate for
    building competition into the substrate.

    Currents use the Lebesgue-p dual inner product <W,v>_p = sum |W|^(p-1) sign(W) v.
    Follows Krotov's reference implementation (Biological_Learning): update normalized by
    max|dW|, linearly decaying learning rate.

    DEVIATION FROM THE PAPER (documented, deliberate): the original learns on whole raw
    images; here it learns on the SAME ZCA-whitened patches every other arm sees, so the
    ladder stays matched. This is an adaptation of their rule, not a replication of their
    protocol, and their published CIFAR number is NOT a target for this arm.
    """
    rng = np.random.RandomState(seed)
    Xt = torch.as_tensor(X, device=device)
    N, d = Xt.shape
    W = torch.as_tensor(rng.normal(0.0, 1.0, (K, d)).astype(np.float32), device=device)
    for ep in range(epochs):
        eps = eps0 * (1.0 - ep / epochs)
        order = torch.as_tensor(rng.permutation(N), device=device)
        for i in range(0, N, batch):
            v = Xt[order[i:i + batch]]                               # (n,d)
            sig = torch.sign(W)
            cur = (sig * W.abs().pow(p - 1)) @ v.T                   # (K,n) currents
            # rank-based competition: winner +1, k-th ranked -delta, rest 0
            idx = cur.argsort(0)
            g = torch.zeros_like(cur)
            ar = torch.arange(cur.shape[1], device=device)
            g[idx[K - 1], ar] = 1.0
            g[idx[K - k_rank], ar] = -delta
            xx = (g * cur).sum(1)                                    # (K,)
            dW = g @ v - xx.unsqueeze(1) * W
            nc = dW.abs().max().clamp(min=prec)
            W = W + eps * dW / nc
        if verbose:
            print(f"    krotov epoch {ep + 1}/{epochs} eps={eps:.4g} |W|={float(W.abs().mean()):.4f}",
                  flush=True)
    return W.cpu().numpy()


def random_gaussian_dict(d: int, K: int, seed: int) -> np.ndarray:
    return np.random.RandomState(seed).randn(K, d).astype(np.float32)


def random_patch_dict(Xp: np.ndarray, K: int, seed: int) -> np.ndarray:
    """Coates' own control: the dictionary is just random (whitened) data patches, no
    learning at all. He found this nearly matches k-means under triangle encoding -- i.e.
    the *encoding+pooling*, not the dictionary learning, carries most of the performance.
    Directly relevant to whether learning the substrate's filters is load-bearing."""
    rng = np.random.RandomState(seed)
    return Xp[rng.choice(len(Xp), K, replace=False)].copy()


def normalize_dict(D: np.ndarray) -> np.ndarray:
    """Matched row normalization applied to EVERY arm's dictionary before encoding, so that
    triangle encoding is comparable across arms whose rules produce different weight scales."""
    return (D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-8)).astype(np.float32)


# --------------------------------------------------------------------------- encoding

def encode_batch(pz: torch.Tensor, D: torch.Tensor, mode: str = "triangle",
                 alpha: float = 0.25) -> torch.Tensor:
    """(n,d) whitened patches x (K,d) dictionary -> (n,K) activations."""
    d2 = (pz * pz).sum(1, keepdim=True) + (D * D).sum(1).unsqueeze(0) - 2.0 * (pz @ D.T)
    dist = d2.clamp(min=0).sqrt()
    if mode == "triangle":                       # Coates: max(0, mean_k(dist) - dist_k)
        return (dist.mean(1, keepdim=True) - dist).clamp(min=0)
    if mode == "hard":                           # 1-of-K
        out = torch.zeros_like(dist)
        out.scatter_(1, dist.argmin(1, keepdim=True), 1.0)
        return out
    if mode == "thresh":                         # soft-threshold on the dot product
        return (pz @ D.T - alpha).clamp(min=0)
    raise ValueError(f"unknown encoder {mode}")


def pool_quadrants(act: torch.Tensor, L: int) -> torch.Tensor:
    """(B,L*L,K) -> (B,4K) 2x2 quadrant SUM pooling (Coates)."""
    B, _, K = act.shape
    a = act.reshape(B, L, L, K)
    h = L // 2
    quads = [a[:, :h, :h], a[:, :h, h:], a[:, h:, :h], a[:, h:, h:]]
    return torch.cat([q.sum((1, 2)) for q in quads], 1)


def encode_dataset(X: np.ndarray, D: np.ndarray, pre: Preproc, device, P: int,
                   stride: int = 1, mode: str = "triangle", alpha: float = 0.25,
                   batch: int = 64, dtype=np.float32, verbose: bool = False) -> np.ndarray:
    """Full feature extraction: (N,3,32,32) -> (N, 4K)."""
    Dt = torch.as_tensor(D, device=device)
    L = (X.shape[2] - P) // stride + 1
    out = np.zeros((len(X), 4 * D.shape[0]), dtype=dtype)
    for i in range(0, len(X), batch):
        xb = X[i:i + batch]
        pb = torch.as_tensor(extract_patches(xb, P, stride), device=device)
        pz = pre.torch_apply(pb)
        act = encode_batch(pz, Dt, mode, alpha).reshape(len(xb), L * L, D.shape[0])
        out[i:i + len(xb)] = pool_quadrants(act, L).cpu().numpy().astype(dtype)
        if verbose and (i // batch) % 50 == 0:
            print(f"    encode {i}/{len(X)}", flush=True)
    return out


# --------------------------------------------------------------------------- readout

def fit_readout(Ftr, ytr, Fva, yva, device, C_grid=(3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1),
                epochs: int = 60, batch: int = 512, lr: float = 3e-3, seed: int = 0,
                verbose: bool = True):
    """L2-regularized multinomial logistic readout (torch, GPU), C selected on the VAL split.

    C is the inverse regularization strength (sklearn convention): weight_decay = 1/(C*n).
    Coates used a cross-validated L2-SVM; this is the same regularized-linear family and
    lets us sweep C cheaply. Features are standardized using TRAIN statistics only.
    """
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-5
    Xtr = torch.as_tensor((Ftr - mu) / sd, device=device, dtype=torch.float32)
    Xva = torch.as_tensor((Fva - mu) / sd, device=device, dtype=torch.float32)
    Ytr = torch.as_tensor(ytr, device=device, dtype=torch.long)
    Yva = torch.as_tensor(yva, device=device, dtype=torch.long)
    best = (-1.0, None, None)
    for C in C_grid:
        torch.manual_seed(seed)
        lin = torch.nn.Linear(Xtr.shape[1], int(ytr.max()) + 1).to(device)
        opt = torch.optim.AdamW(lin.parameters(), lr=lr, weight_decay=1.0 / (C * len(Xtr)))
        for ep in range(epochs):
            perm = torch.randperm(len(Xtr), device=device)
            for i in range(0, len(Xtr), batch):
                b = perm[i:i + batch]
                opt.zero_grad()
                torch.nn.functional.cross_entropy(lin(Xtr[b]), Ytr[b]).backward()
                opt.step()
        with torch.no_grad():
            acc = float((lin(Xva).argmax(1) == Yva).float().mean())
        if verbose:
            print(f"    readout C={C:<8g} val={acc * 100:.2f}%", flush=True)
        if acc > best[0]:
            best = (acc, C, lin)
    return {"val_acc": best[0], "C": best[1], "model": best[2], "mu": mu, "sd": sd}


def eval_test(readout, Fte, yte, device) -> float:
    """The ONLY place the test set is scored. Called once per arm."""
    Xte = torch.as_tensor((Fte - readout["mu"]) / readout["sd"], device=device, dtype=torch.float32)
    with torch.no_grad():
        pred = readout["model"](Xte).argmax(1).cpu().numpy()
    return float((pred == yte).mean())


# --------------------------------------------------------------------------- stats

def bootstrap_ci(vals, nboot: int = 5000, seed: int = 0, pct=(2.5, 97.5)):
    """Bootstrap CI over seeds. With <2 seeds returns nan bounds (reported as such, never
    silently treated as a CI)."""
    v = np.asarray(vals, dtype=float)
    if len(v) < 2:
        return float(v.mean()) if len(v) else float("nan"), (float("nan"), float("nan"))
    rng = np.random.RandomState(seed)
    boots = [v[rng.randint(0, len(v), len(v))].mean() for _ in range(nboot)]
    return float(v.mean()), (float(np.percentile(boots, pct[0])), float(np.percentile(boots, pct[1])))


def experiment_protocol(args, **experiment):
    """Identify settings and source when deciding whether a local result can be reused."""
    settings = {
        key: value for key, value in vars(args).items()
        if key not in {"out", "verbose", "stage", "seeds", "arms", "ablations", "k_sweep"}
    }
    source = {}
    for name in ("common.py", "run_kmeans_ceiling.py", "run_learned_ladder.py"):
        source[name] = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
    return {"schema": 1, "settings": settings, "experiment": experiment, "source_sha256": source}


def read_cached_record(path, protocol):
    """Reject legacy or mismatched caches rather than relabel an old experiment."""
    with Path(path).open(encoding="utf-8") as stream:
        record = json.load(stream)
    if not isinstance(record, dict) or record.get("protocol") != protocol:
        raise ValueError(
            f"cached result does not match this protocol: {path}. "
            "Preserve that record and choose a new --out directory."
        )
    return record


def write_record(path, record, protocol):
    """Create a new result; a race or an existing artifact must not replace old evidence."""
    record = {**record, "protocol": protocol}
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
    return record
