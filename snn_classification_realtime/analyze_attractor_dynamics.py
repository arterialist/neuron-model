"""Does the JOINT oscillatory attractor carry class info BEYOND the static mean state?

Reads the standard activity HDF5 (per-tick x per-neuron membrane S). For each sample it splits
the substrate response into:
  - meanS (DC)        : per-neuron time-average = the static centroid of the joint attractor.
  - DYNAMICS (AC)     : per-neuron de-meaned oscillation (S - per-neuron temporal mean),
                        so it contains ZERO mean-state info. Two ways to describe it:

    A) joint-dynamics SUMMARY (per sample, sign-invariant): normalized eigenvalue spectrum of the
       across-neuron covariance (collective-mode energy distribution) + per-neuron AC power +
       per-neuron dominant frequency + per-neuron spectral centroid.

    B) full high-dim joint TRAJECTORY in a SHARED basis: one global PCA basis (fit on pooled
       de-meaned trajectories) -> project each sample to P dims -> describe limit-cycle geometry
       via the projected-trajectory covariance (upper-tri) + per-PC power + per-PC dominant freq.

Then a linear/kNN probe compares, on one shared 70/30 split:
  chance | meanS(DC) | A-alone | B-alone | meanS+A | meanS+B | full(meanS+A+B)
Answering: does pure dynamics classify above chance / approach meanS, and does adding it beat
meanS alone? If yes -> the joint oscillatory attractor is load-bearing, not just the mean.
CLI-only. No re-recording; runs on existing recordings.
"""
import argparse
import glob
import json
import os
import time

import h5py
import numpy as np


def resolve_h5(p):
    if os.path.isdir(p):
        c = os.path.join(p, "activity_dataset.h5")
        if os.path.exists(c):
            return c
        g = glob.glob(os.path.join(p, "*.h5"))
        if g:
            return g[0]
    return p


def per_neuron_spectral(ac):
    """ac: (T, N) de-meaned. Return per-neuron (power, dom_freq, centroid) each length N."""
    T = ac.shape[0]
    mag = np.abs(np.fft.rfft(ac, axis=0))          # (F, N)
    mag[0] = 0.0
    freqs = np.fft.rfftfreq(T)                       # (F,)
    power = (ac.var(0))                              # AC power per neuron
    tot = mag.sum(0) + 1e-9
    dom = freqs[mag.argmax(0)]                       # dominant freq per neuron
    centroid = (mag * freqs[:, None]).sum(0) / tot   # spectral centroid per neuron
    return power.astype(np.float32), dom.astype(np.float32), centroid.astype(np.float32)


def choose_indices(Y_all, max_per_class, seed):
    classes = sorted(set(Y_all.tolist()))
    by = {c: [] for c in classes}
    for i in np.random.RandomState(seed).permutation(len(Y_all)):
        c = int(Y_all[i])
        if len(by[c]) < max_per_class:
            by[c].append(int(i))
    return classes, sorted(i for c in classes for i in by[c])


def fit_global_pca(U, idxs, settle, P, seed, n_fit=200):
    """Global PCA basis on pooled de-meaned trajectories (subsample of samples)."""
    rng = np.random.RandomState(seed + 1)
    fit_ids = rng.choice(idxs, size=min(n_fit, len(idxs)), replace=False)
    rows = []
    for i in sorted(fit_ids):
        s = U[i, settle:, :].astype(np.float32)
        rows.append(s - s.mean(0))                  # de-meaned (AC)
    X = np.concatenate(rows, 0)                      # (sum_T, N)
    mu = X.mean(0)
    _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    return mu.astype(np.float32), Vt[:P].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="foveation_results/substrate_attractors")
    ap.add_argument("--settle-frac", type=float, default=0.4)
    ap.add_argument("--max-per-class", type=int, default=200)
    ap.add_argument("--eig-k", type=int, default=30, help="# eigenvalues in the joint-dynamics spectrum (A)")
    ap.add_argument("--pca-p", type=int, default=30, help="# PCA dims for the shared trajectory basis (B)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    h5 = resolve_h5(a.h5)
    print(f"[{a.tag}] reading {h5}", flush=True)
    f = h5py.File(h5, "r")
    U = f["u"]; Y_all = np.asarray(f["labels"]).astype(int)
    n, ticks, N = U.shape
    settle = int(ticks * a.settle_frac)
    K = min(a.eig_k, N); P = min(a.pca_p, N)
    classes, idxs = choose_indices(Y_all, a.max_per_class, a.seed)
    print(f"[{a.tag}] n={len(idxs)} neurons={N} ticks={ticks} settle={settle} K={K} P={P}", flush=True)

    print(f"[{a.tag}] fitting global PCA basis (B)...", flush=True)
    pmu, pV = fit_global_pca(U, idxs, settle, P, a.seed)   # (N,), (P,N)
    iuP = np.triu_indices(P)

    meanS, Aeig, Apow, Adom, Acen, Bcov, Bpow, Bdom, Y = ([] for _ in range(9))
    t0 = time.time()
    for pos, i in enumerate(idxs):
        s = U[i, settle:, :].astype(np.float32)
        ac = s - s.mean(0)                                  # (T,N) DC-removed
        Y.append(int(Y_all[i]))
        meanS.append(s.mean(0))                             # DC baseline (N)
        # ---- A: joint-dynamics summary (sign-invariant) ----
        C = np.cov(ac.T)                                    # (N,N) across-neuron cov
        ev = np.linalg.eigvalsh(C)[::-1][:K]                # top-K eigenvalues (sorted desc)
        ev = ev / (ev.sum() + 1e-9)                         # normalized collective-mode energy
        Aeig.append(ev.astype(np.float32))
        pw, dm, cn = per_neuron_spectral(ac)
        Apow.append(pw); Adom.append(dm); Acen.append(cn)
        # ---- B: shared-basis joint trajectory geometry ----
        proj = (ac - pmu) @ pV.T                            # (T,P) in shared basis
        Bcov.append(np.cov(proj.T)[iuP].astype(np.float32)) # limit-cycle covariance (upper-tri)
        magP = np.abs(np.fft.rfft(proj, axis=0)); magP[0] = 0
        freqs = np.fft.rfftfreq(proj.shape[0])
        Bpow.append(proj.var(0).astype(np.float32))         # per-PC power
        Bdom.append(freqs[magP.argmax(0)].astype(np.float32))  # per-PC dominant freq
        if pos % 300 == 0:
            print(f"  {pos}/{len(idxs)} ({time.time()-t0:.0f}s)", flush=True)
    f.close()
    Y = np.asarray(Y)
    meanS = np.asarray(meanS, np.float32)
    A = np.concatenate([np.asarray(Aeig, np.float32), np.asarray(Apow, np.float32),
                        np.asarray(Adom, np.float32), np.asarray(Acen, np.float32)], 1)
    B = np.concatenate([np.asarray(Bcov, np.float32), np.asarray(Bpow, np.float32),
                        np.asarray(Bdom, np.float32)], 1)
    print(f"[{a.tag}] dims: meanS={meanS.shape[1]} A={A.shape[1]} B={B.shape[1]}", flush=True)

    # ---- shared split + linear/kNN probe ----
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    rng = np.random.RandomState(0); perm = rng.permutation(len(Y))
    Y = Y[perm]; ntr = int(len(Y) * 0.7)
    feats = {"meanS_DC": meanS[perm], "A_dyn": A[perm], "B_traj": B[perm],
             "meanS+A": np.concatenate([meanS, A], 1)[perm],
             "meanS+B": np.concatenate([meanS, B], 1)[perm],
             "full": np.concatenate([meanS, A, B], 1)[perm]}

    def clf(X):
        Xtr, Xte, ytr, yte = X[:ntr], X[ntr:], Y[:ntr], Y[ntr:]
        m, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; Xtr, Xte = (Xtr - m) / sd, (Xte - m) / sd
        lin = max(LogisticRegression(max_iter=2000, C=C).fit(Xtr, ytr).score(Xte, yte)
                  for C in (0.01, 0.05, 0.2, 1.0))
        knn = KNeighborsClassifier(min(15, max(1, len(ytr) - 1))).fit(Xtr, ytr).score(Xte, yte)
        return dict(linear=round(float(lin), 3), knn=round(float(knn), 3))

    res = {k: clf(X) for k, X in feats.items()}
    chance = round(1.0 / len(classes), 3)
    verdict = dict(
        chance=chance,
        dynamics_alone_beats_chance=res["A_dyn"]["linear"] > 2 * chance or res["B_traj"]["linear"] > 2 * chance,
        A_adds_over_mean=round(res["meanS+A"]["linear"] - res["meanS_DC"]["linear"], 3),
        B_adds_over_mean=round(res["meanS+B"]["linear"] - res["meanS_DC"]["linear"], 3),
        best_dynamics_alone=round(max(res["A_dyn"]["linear"], res["B_traj"]["linear"]), 3),
        meanS_linear=res["meanS_DC"]["linear"],
    )

    out = dict(exp="attractor_dynamics_vs_mean", tag=a.tag, h5=h5,
               n=int(len(Y)), neurons=int(N), ticks=int(ticks), settle=int(settle),
               dims={k: int(v.shape[1]) for k, v in feats.items()},
               classification=res, verdict=verdict, secs=round(time.time() - t0))
    json.dump(out, open(os.path.join(a.out, f"{a.tag}_dynamics.json"), "w"), indent=2)

    # bar figure
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    order = ["meanS_DC", "A_dyn", "B_traj", "meanS+A", "meanS+B", "full"]
    vals = [res[k]["linear"] for k in order]
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#6b7280", "#0d818f", "#7c3aed", "#0891b2", "#9333ea", "#059669"]
    bars = ax.bar(order, vals, color=colors)
    ax.axhline(chance, color="k", ls=":", lw=1); ax.text(0, chance + 0.005, f"chance {chance}", fontsize=8)
    ax.axhline(res["meanS_DC"]["linear"], color="#6b7280", ls="--", lw=1)
    ax.set_ylabel("linear held-out acc")
    ax.set_title(f"{a.tag}: does joint DYNAMICS (DC-removed) carry class info beyond the mean state?")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.2f}", ha="center", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, f"{a.tag}_dynamics.png"), dpi=130); plt.close(fig)

    print(f"[{a.tag}] meanS={res['meanS_DC']['linear']} | A_dyn={res['A_dyn']['linear']} "
          f"B_traj={res['B_traj']['linear']} | meanS+A={res['meanS+A']['linear']} "
          f"meanS+B={res['meanS+B']['linear']} full={res['full']['linear']} (chance {chance})", flush=True)
    print(f"[{a.tag}] VERDICT dyn-alone-beats-chance={verdict['dynamics_alone_beats_chance']} "
          f"A_adds={verdict['A_adds_over_mean']:+} B_adds={verdict['B_adds_over_mean']:+}", flush=True)


if __name__ == "__main__":
    main()
