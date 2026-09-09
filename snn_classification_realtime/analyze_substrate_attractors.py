"""Attractor analysis of a PAULA substrate's recorded activity (NO fovea).

Consumes the STANDARD activity dataset (build_activity_dataset HDF5): per-tick, per-neuron
u (=S, membrane potential), t_ref (plastic learning window), fr (firing rate), plus labels.
Analyzes the substrate's ATTRACTORS in the paper's (<S>, <t_ref>) phase space (see PAULA paper
S4.5 / S6):

  1. TYPE per exemplar   -- limit-cycle / quasi-periodic / chaotic / fixed-point
                            (spectrum, autocorr period, Poincare return map, recurrence,
                             largest-Lyapunov via Rosenstein).
  2. WITHIN/BETWEEN class -- do same-class images share an attractor? Fisher ratio, silhouette,
                            class-distance heatmap, 2D embedding.
  3. CLASSIFY-BY-ATTRACTOR -- linear / kNN / nearest-centroid on a compact attractor descriptor,
                            plus ablations (which part of the attractor carries the class:
                            mean-S / mean-t_ref / mean-fr / dynamics-shape / population-spectrum).
  4. VISUALS + ANIMATIONS -- per-class attractor landscape KDE in (<S>,<t_ref>) (paper Fig 5),
                            rotating 3D PCA attractor, phase portraits, embedding, classify bar.

Memory-lean: reads the HDF5 sample-by-sample, reduces each trajectory to a small descriptor,
keeps only a few full exemplar trajectories per class. CLI-only.
"""
import argparse
import glob
import json
import os
import time

import h5py
import numpy as np

DR = 8          # random-projection dim for dynamics-shape descriptor
KSPEC = 40      # population-spectrum bins
EXEMPLARS_PER_CLASS = 5


# ------------------------------- helpers ---------------------------------- #
def _proj_matrix(N, seed=0):
    return (np.random.RandomState(seed + 99).randn(DR, N) / np.sqrt(N)).astype(np.float32)


def _spec(sig, k=KSPEC):
    mag = np.abs(np.fft.rfft(sig - sig.mean()))
    if len(mag) < 2:
        return np.zeros(k, np.float32)
    return np.interp(np.linspace(0, len(mag) - 1, k), np.arange(len(mag)), mag).astype(np.float32)


def lyap_rosenstein(x, m=3, tau=2, k=8):
    """Largest Lyapunov exponent (Rosenstein): >0 chaotic, ~0 periodic."""
    x = (x - x.mean()) / (x.std() + 1e-9)
    n = len(x) - (m - 1) * tau
    if n < 20:
        return 0.0
    emb = np.stack([x[i * tau:i * tau + n] for i in range(m)], 1)
    d = np.linalg.norm(emb[:, None] - emb[None], axis=2)
    np.fill_diagonal(d, np.inf)
    for i in range(len(d)):
        d[i, max(0, i - k):min(len(d), i + k)] = np.inf
    nn = d.argmin(1); div = []
    for j in range(1, min(20, n - 1)):
        dl = [np.linalg.norm(emb[i + j] - emb[nn[i] + j])
              for i in range(len(nn) - j) if nn[i] + j < len(emb)]
        if dl:
            div.append(np.log(np.mean(dl) + 1e-12))
    if len(div) < 3:
        return 0.0
    return float(np.polyfit(np.arange(len(div)), div, 1)[0])


# ------------------------------ extraction -------------------------------- #
def extract(h5path, settle_frac, max_per_class, seed=0):
    """Stream the HDF5; per sample build a compact attractor descriptor + keep exemplar
    trajectories. Returns descriptors D, labels Y, slice map, exemplar bundles, meta."""
    f = h5py.File(h5path, "r")
    U, TR, FR = f["u"], f["t_ref"], f["fr"]           # (n, ticks, neurons)
    Y_all = np.asarray(f["labels"]).astype(int)
    n, ticks, N = U.shape
    settle = int(ticks * settle_frac)
    R = _proj_matrix(N, seed)
    classes = sorted(set(Y_all.tolist()))

    # choose up to max_per_class indices per class (deterministic)
    by_cls = {c: [] for c in classes}
    for i in np.random.RandomState(seed).permutation(n):
        c = int(Y_all[i])
        if len(by_cls[c]) < max_per_class:
            by_cls[c].append(int(i))
    idxs = sorted(i for c in classes for i in by_cls[c])
    exemplars = {int(by_cls[c][k]) for c in classes for k in range(min(EXEMPLARS_PER_CLASS, len(by_cls[c])))}

    D, Y, slc = [], [], None
    ex_pop, ex_full, ex_y, ex_id = [], [], [], []     # exemplar (<S>,<t_ref>) 2D + PCA-3D traj
    t0 = time.time()
    for pos, i in enumerate(idxs):
        s = U[i, settle:, :].astype(np.float32)        # (T, N) membrane
        tr = TR[i, settle:, :].astype(np.float32)      # (T, N) t_ref
        fr = FR[i, settle:, :].astype(np.float32)      # (T, N) firing rate
        y = int(Y_all[i]); Y.append(y)
        # per-neuron attractor location (mean over settled window) + temporal spread
        mS, mT, mF = s.mean(0), tr.mean(0), fr.mean(0)
        sdS, sdT = s.std(0), tr.std(0)
        # population phase-space trajectory (paper's (<S>,<t_ref>))
        popS, popT, popF = s.mean(1), tr.mean(1), fr.mean(1)
        # dynamics SHAPE: cov of random-projected membrane trajectory (upper-tri)
        proj = s @ R.T                                 # (T, DR)
        iu = np.triu_indices(DR)
        shape = np.cov(proj.T)[iu].astype(np.float32)
        # population spectrum of membrane rhythm
        spec = _spec(popS)
        # assemble descriptor + record component slices once
        parts = [mS, mT, mF, sdS, sdT, shape, spec]
        d = np.concatenate(parts).astype(np.float32)
        if slc is None:
            names = ["meanS", "mean_tref", "mean_fr", "stdS", "std_tref", "shape", "spec"]
            off, slc = 0, {}
            for nm, p in zip(names, parts):
                slc[nm] = (off, off + len(p)); off += len(p)
        D.append(d)
        if i in exemplars:
            ex_pop.append(np.stack([popS, popT], 1).astype(np.float32))   # (T,2)
            ex_full.append(proj.astype(np.float32))                        # (T,DR) for PCA-3D
            ex_y.append(y); ex_id.append(i)
        if pos % 200 == 0:
            print(f"  extract {pos}/{len(idxs)} ({time.time()-t0:.0f}s)", flush=True)
    f.close()
    meta = dict(n_used=len(Y), ticks=ticks, settle=settle, neurons=N, classes=classes)
    return (np.asarray(D, np.float32), np.asarray(Y), slc,
            ex_pop, ex_full, np.asarray(ex_y), np.asarray(ex_id), meta)


# ------------------------------ type analysis ----------------------------- #
def analyze_type(ex_pop, ex_full, ex_y, out, tag):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib import animation
    if not ex_full:
        return {}
    allpts = np.concatenate([e for e in ex_full], 0); mu = allpts.mean(0)
    _, _, Vt = np.linalg.svd(allpts - mu, full_matrices=False); comps = Vt[:3]
    cmap = plt.get_cmap("tab10")
    types, rows = [], []
    fig_s, axs = plt.subplots(2, 3, figsize=(15, 8)); axs = axs.ravel()
    for e in range(len(ex_full)):
        tr = ex_full[e]; y = int(ex_y[e])
        pc = (tr - mu) @ comps.T; p0 = pc[:, 0]
        mag = np.abs(np.fft.rfft(p0 - p0.mean())); mag[0] = 0
        flat = float(np.exp(np.mean(np.log(mag + 1e-9))) / (mag.mean() + 1e-9))
        npk = int((mag > 0.5 * mag.max()).sum())
        ac = np.correlate(p0 - p0.mean(), p0 - p0.mean(), "full")[len(p0) - 1:]
        ac = ac / (ac[0] + 1e-9)
        period = int(np.argmax(ac[5:] > 0.5) + 5) if (ac[5:] > 0.5).any() else 0
        lyap = lyap_rosenstein(p0)
        # recurrence rate in 3D PCA space
        dd = np.linalg.norm(pc[:, None] - pc[None], axis=2)
        thr = np.percentile(dd, 10); rr = float((dd < thr).mean())
        # classify
        if flat > 0.4 and lyap > 0.02:
            typ = "chaotic"
        elif npk <= 2 and period > 0 and flat < 0.25:
            typ = "limit-cycle"
        elif npk <= 6 and flat < 0.4:
            typ = "quasi-periodic"
        else:
            typ = "complex"
        types.append(typ)
        rows.append(dict(exemplar=e, label=y, type=typ, flatness=round(flat, 3),
                         n_peaks=npk, period=period, lyap=round(lyap, 4), recurrence=round(rr, 3)))
        if e < 6:
            ax = axs[e]; ax.plot(pc[:, 0], pc[:, 1], lw=0.7, color=cmap(y % 10))
            ax.scatter(pc[0, 0], pc[0, 1], c="g", s=20); ax.scatter(pc[-1, 0], pc[-1, 1], c="r", s=20)
            ax.set_title(f"cls {y}: {typ} (flat {flat:.2f}, lyap {lyap:+.3f})", fontsize=9)
    fig_s.suptitle(f"{tag}: exemplar attractors (PCA phase portraits)")
    fig_s.tight_layout(); fig_s.savefig(os.path.join(out, f"{tag}_phase_portraits.png"), dpi=130)
    plt.close(fig_s)

    # rotating 3D attractor animation (all exemplars)
    try:
        fig = plt.figure(figsize=(7, 6)); ax = fig.add_subplot(111, projection="3d")
        for e in range(len(ex_full)):
            pc = (ex_full[e] - mu) @ comps.T
            ax.plot(pc[:, 0], pc[:, 1], pc[:, 2], lw=0.5, color=cmap(int(ex_y[e]) % 10), alpha=0.6)
        ax.set_title(f"{tag}: substrate attractors (PCA-3D)")

        def rot(i):
            ax.view_init(elev=20, azim=i * 4); return []
        anim = animation.FuncAnimation(fig, rot, frames=90, interval=60, blit=False)
        _save_anim(anim, os.path.join(out, f"{tag}_attractors_rotate")); plt.close(fig)
    except Exception as e:
        print(f"rotate anim failed: {e}", flush=True)

    from collections import Counter
    cnt = Counter(types)
    dominant = cnt.most_common(1)[0][0] if cnt else "n/a"
    return dict(dominant_type=dominant, type_counts=dict(cnt), per_exemplar=rows)


# ------------------------------ geometry ---------------------------------- #
def analyze_geometry(D, Y, slc, out, tag, seed=0):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import silhouette_score
    classes = sorted(set(Y.tolist())); K = len(classes)
    n = len(Y); rng = np.random.RandomState(0); perm = rng.permutation(n)
    D, Y = D[perm], Y[perm]; ntr = int(n * 0.7)
    mu, sd = D[:ntr].mean(0), D[:ntr].std(0) + 1e-6; Dn = (D - mu) / sd

    cents = np.stack([D[Y == c].mean(0) for c in classes])
    within = np.mean([np.linalg.norm(D[i] - cents[classes.index(Y[i])]) for i in range(n)])
    between = np.mean([np.linalg.norm(cents[i] - cents[j]) for i in range(K) for j in range(i + 1, K)])
    fisher = float(between / (within + 1e-9))
    try:
        sil = float(silhouette_score(Dn[:3000], Y[:3000]))
    except Exception:
        sil = float("nan")

    def clf(feat):
        Xtr, Xte, ytr, yte = feat[:ntr], feat[ntr:], Y[:ntr], Y[ntr:]
        m, s = Xtr.mean(0), Xtr.std(0) + 1e-6; Xtr, Xte = (Xtr - m) / s, (Xte - m) / s
        lin = max(LogisticRegression(max_iter=1500, C=C).fit(Xtr, ytr).score(Xte, yte)
                  for C in (0.02, 0.1, 0.5))
        knn = KNeighborsClassifier(min(15, max(1, len(ytr) - 1))).fit(Xtr, ytr).score(Xte, yte)
        ctr = np.stack([Xtr[ytr == c].mean(0) for c in classes])
        nc = float((np.array([classes[j] for j in np.argmin(
            np.linalg.norm(Xte[:, None] - ctr[None], axis=2), 1)]) == yte).mean())
        return dict(linear=round(lin, 3), knn=round(knn, 3), nearest_centroid=round(nc, 3))

    res = {"full": clf(D)}
    for name, (a, b) in slc.items():
        res[name] = clf(D[:, a:b])
    chance = round(1.0 / K, 3)

    # class-distance heatmap
    dmat = np.linalg.norm(cents[:, None] - cents[None], axis=2)
    fig, ax = plt.subplots(figsize=(6, 5.2)); im = ax.imshow(dmat, cmap="magma")
    ax.set_title(f"{tag}: class-attractor centroid distances"); fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xticks(range(K)); ax.set_yticks(range(K)); ax.set_xticklabels(classes); ax.set_yticklabels(classes)
    fig.tight_layout(); fig.savefig(os.path.join(out, f"{tag}_classdist.png"), dpi=130); plt.close(fig)

    # 2D PCA embedding
    Dc = D - D.mean(0); _, _, Vt = np.linalg.svd(Dc[:4000], full_matrices=False); emb = Dc @ Vt[:2].T
    fig, ax = plt.subplots(figsize=(7.4, 6.6)); cmap = plt.get_cmap("tab10")
    for c in classes:
        ax.scatter(emb[Y == c, 0], emb[Y == c, 1], s=6, color=cmap(c % 10), alpha=0.45, label=str(c))
    ax.set_title(f"{tag}: attractor-descriptor embedding (Fisher={fisher:.2f}, sil={sil:.2f})")
    ax.legend(fontsize=7, ncol=5, markerscale=2); fig.tight_layout()
    fig.savefig(os.path.join(out, f"{tag}_embedding.png"), dpi=130); plt.close(fig)

    # ablation bar (which part of the attractor carries the class)
    keys = ["full"] + list(slc.keys())
    vals = [res[k]["linear"] for k in keys]
    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(keys, vals, color=["#059669"] + ["#0d818f"] * len(slc))
    ax.axhline(chance, color="k", ls=":", lw=1); ax.text(0, chance + 0.005, f"chance {chance}", fontsize=8)
    ax.set_ylabel("linear held-out acc"); ax.set_title(f"{tag}: classify-by-attractor + ablation")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.2f}", ha="center", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, f"{tag}_classify.png"), dpi=130); plt.close(fig)

    return dict(fisher_ratio=round(fisher, 3), silhouette=round(sil, 3), chance=chance,
                within_class_dist=round(float(within), 3), between_class_dist=round(float(between), 3),
                classification=res,
                key_delta_full_minus_meanS=round(res["full"]["linear"] - res["meanS"]["linear"], 3))


# --------------------- (<S>,<t_ref>) attractor landscapes ------------------ #
def landscape_figure(h5path, settle_frac, max_per_class, out, tag, seed=0):
    """Paper Fig-5 style: KDE density of the population state in (<S>,<t_ref>) per class."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    f = h5py.File(h5path, "r")
    U, TR = f["u"], f["t_ref"]; Y_all = np.asarray(f["labels"]).astype(int)
    n, ticks, N = U.shape; settle = int(ticks * settle_frac)
    classes = sorted(set(Y_all.tolist()))
    by_cls = {c: [] for c in classes}
    for i in np.random.RandomState(seed).permutation(n):
        c = int(Y_all[i])
        if len(by_cls[c]) < min(max_per_class, 60):
            by_cls[c].append(int(i))
    pts = {c: [] for c in classes}
    for c in classes:
        for i in by_cls[c]:
            popS = U[i, settle:, :].mean(1); popT = TR[i, settle:, :].mean(1)
            pts[c].append(np.stack([popS, popT], 1))
    f.close()
    K = len(classes); cols = 5; rows = (K + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows)); axs = np.atleast_1d(axs).ravel()
    allp = np.concatenate([np.concatenate(pts[c], 0) for c in classes], 0)
    xlim = (allp[:, 0].min(), allp[:, 0].max()); ylim = (allp[:, 1].min(), allp[:, 1].max())
    for k, c in enumerate(classes):
        ax = axs[k]; P = np.concatenate(pts[c], 0)
        ax.hexbin(P[:, 0], P[:, 1], gridsize=30, cmap="viridis", bins="log",
                  extent=(*xlim, *ylim))
        ax.set_title(f"class {c}", fontsize=9); ax.set_xlabel("<S>", fontsize=7); ax.set_ylabel("<t_ref>", fontsize=7)
    for k in range(K, len(axs)):
        axs[k].axis("off")
    fig.suptitle(f"{tag}: attractor landscapes in (<S>, <t_ref>) phase space")
    fig.tight_layout(); fig.savefig(os.path.join(out, f"{tag}_landscapes.png"), dpi=130); plt.close(fig)


def _save_anim(anim, base):
    try:
        anim.save(base + ".mp4", writer="ffmpeg", fps=20, dpi=110); print(f"saved {base}.mp4", flush=True); return
    except Exception as e:
        print(f"ffmpeg failed ({e}); trying gif", flush=True)
    try:
        anim.save(base + ".gif", writer="pillow", fps=15); print(f"saved {base}.gif", flush=True)
    except Exception as e:
        print(f"gif failed ({e})", flush=True)


# ---------------------------------- main ---------------------------------- #
def resolve_h5(p):
    if os.path.isdir(p):
        c = os.path.join(p, "activity_dataset.h5")
        if os.path.exists(c):
            return c
        g = glob.glob(os.path.join(p, "*.h5"))
        if g:
            return g[0]
    return p


def main():
    ap = argparse.ArgumentParser(description="Attractor analysis of a PAULA substrate's recorded activity.")
    ap.add_argument("--h5", required=True, help="activity_dataset.h5 file or its directory")
    ap.add_argument("--out", default="foveation_results/substrate_attractors")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--settle-frac", type=float, default=0.4, help="fraction of leading ticks to drop as transient")
    ap.add_argument("--max-per-class", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    h5 = resolve_h5(a.h5)
    print(f"[{a.tag}] reading {h5}", flush=True)
    t0 = time.time()
    D, Y, slc, ex_pop, ex_full, ex_y, ex_id, meta = extract(h5, a.settle_frac, a.max_per_class, a.seed)
    print(f"[{a.tag}] extracted {len(Y)} descriptors dim={D.shape[1]}, {len(ex_full)} exemplars ({time.time()-t0:.0f}s)", flush=True)
    typ = analyze_type(ex_pop, ex_full, ex_y, a.out, a.tag)
    geo = analyze_geometry(D, Y, slc, a.out, a.tag, a.seed)
    landscape_figure(h5, a.settle_frac, a.max_per_class, a.out, a.tag, a.seed)
    result = dict(exp="substrate_attractors", tag=a.tag, h5=h5, meta=meta,
                  desc_dim=int(D.shape[1]), type=typ, geometry=geo,
                  secs=round(time.time() - t0), complete=True)
    json.dump(result, open(os.path.join(a.out, f"{a.tag}.json"), "w"), indent=2)
    print(f"[{a.tag}] TYPE={typ.get('dominant_type')} | Fisher={geo['fisher_ratio']} sil={geo['silhouette']} "
          f"| classify full-linear={geo['classification']['full']['linear']} "
          f"(meanS={geo['classification']['meanS']['linear']}, chance={geo['chance']})", flush=True)


if __name__ == "__main__":
    main()
