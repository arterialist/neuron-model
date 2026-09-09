"""Faithful reimplementation of T-Randman (Zenke & Vogels 2021): a principled spiking benchmark
where class info is PURELY in spike TIMING. Each of n_units neurons fires ONCE per trial; its
spike time is a smooth random function (truncated Fourier, smoothness alpha) of a low-dim manifold
coordinate. Each class = its own random manifold. A rate/spike-count readout IGNORES timing -> must
fail; a temporal processor that integrates spike times can classify. Tunable difficulty via
dim_manifold (higher = more nonlinear-temporal). Self-generating, no identity leakage."""
import numpy as np

def _smooth_funcs(n_units, dim_manifold, ncomp, alpha, rng):
    # f_i(x) in [0,1]^dim -> [0,1], smooth via sum_k A sin(2pi k x + phi)/k^alpha
    A = rng.randn(n_units, dim_manifold, ncomp)
    phi = rng.uniform(0, 2*np.pi, (n_units, dim_manifold, ncomp))
    ks = np.arange(1, ncomp+1) ** alpha
    def f(X):  # X: (batch, dim_manifold) in [0,1]
        out = np.zeros((X.shape[0], n_units))
        for d in range(dim_manifold):
            # (batch, ncomp)
            ang = 2*np.pi * np.outer(X[:, d], np.arange(1, ncomp+1))
            comp = np.sin(ang[:, None, :] + phi[None, :, d, :]) / ks[None, None, :]  # (batch,units,ncomp)
            out += (comp * A[None, :, d, :]).sum(-1)
        return out
    return f

def generate(n_classes=10, n_units=20, dim_manifold=1, ncomp=6, alpha=2.0,
             n_per_class=200, seed=0):
    rng = np.random.RandomState(seed)
    funcs = [_smooth_funcs(n_units, dim_manifold, ncomp, alpha, rng) for _ in range(n_classes)]
    Xtimes = []; y = []
    for c in range(n_classes):
        coords = rng.uniform(0, 1, (n_per_class, dim_manifold))
        t = funcs[c](coords)  # (n_per_class, n_units) raw
        Xtimes.append(t); y += [c]*n_per_class
    Xt = np.concatenate(Xtimes, 0); y = np.array(y)
    # normalize spike times per unit to [0.05,0.95] across the WHOLE dataset (class-agnostic)
    lo = Xt.min(0, keepdims=True); hi = Xt.max(0, keepdims=True)
    Xt = 0.05 + 0.9 * (Xt - lo) / (hi - lo + 1e-9)
    perm = rng.permutation(len(y))
    return Xt[perm], y[perm]  # Xt in [0,1] spike-time fraction per (sample, unit)

if __name__ == "__main__":
    Xt, y = generate(seed=0)
    print(f"randman: X{Xt.shape} y{y.shape} classes={len(np.unique(y))} times[{Xt.min():.2f},{Xt.max():.2f}]")
    # sanity: rate baseline is USELESS (every unit fires exactly once -> identical rate vector)
    print(f"rate-info check: per-sample spike count is constant ={ (np.ones_like(Xt)).sum(1)[0] } -> rate readout MUST be chance")
