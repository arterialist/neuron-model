"""Attractor-as-representation study (user idea, 2026-07-11).

Yesterday's attribution showed the MEAN activity through the substrate is lossy (linear
~0.22 vs retina ~0.38). But the phase portraits *look* class-clustered for the reservoir
substrates. Hypothesis: the class lives in the ATTRACTOR ITSELF (its identity / shape /
location), not in the time-averaged activity. So the right question is: if you classify by
which attractor a trajectory settles into, does it work -- and are same-class attractors
uniform/close while different-class ones are far?

Design (full image, NO foveation: fovea=32 periph=32 grid=16; settle 500 + observe 500;
500 images/label x 10 = 5000 per arm; arms = {reservoir_retino, reservoir_random} x
{cifar10, cifar10_grayscale}).

Per image we compute a compact ATTRACTOR DESCRIPTOR (storage-lean, on the fly):
  cen   : per-neuron mean rate over the observe window        (N)  -- the "location" (== mean readout)
  std   : per-neuron temporal std over observe                (N)  -- oscillation amplitude
  shape : cov of a fixed random Dr-projection of the traj     (Dr*(Dr+1)/2) -- limit-cycle SHAPE
  spec  : power spectrum of the population rate               (K)  -- periodicity signature
Full low-D (Dr) observe trajectories are kept for a few EXEMPLARS/class for the attractor-
TYPE analysis (spectrum / Poincare / recurrence / autocorr period / largest-Lyapunov) and
the animations.

Analyses per arm:
  (1) TYPE: is each attractor a limit cycle / quasi-periodic torus / chaotic? (spectra,
      Poincare first-return maps, recurrence plots, Lyapunov) -> animated + still figures.
  (2) UNIFORMITY & CLOSENESS: within-class vs between-class descriptor distance, Fisher
      ratio, silhouette, 10x10 class-distance heatmap, 2D embedding of all 5000 attractors.
  (3) ATTRACTOR-CLASSIFICATION: nearest-class-attractor + linear + kNN on the descriptor,
      ablated (centroid-only == mean baseline / +std / +shape / +spectrum). Does the
      attractor descriptor beat centroid-only (retina reference ~0.38 from Part D)?

    PYTHONPATH=. OMP_NUM_THREADS=1 .venv/bin/python -m \
        snn_classification_realtime.foveation.minibrain.exp_attractor_class \
        --arch reservoir_retino --dataset cifar10 --per-class 500 --settle 500 --observe 500 --shards 9
"""
from __future__ import annotations
import argparse, os, json, time, glob
from multiprocessing import Process
import numpy as np
from tqdm import tqdm

from snn_classification_realtime.activity_dataset_builder.vision_datasets import load_dataset_by_name
from snn_classification_realtime.foveation.minibrain.core import MiniBrainConfig, MiniBrain

DR = 12                       # random-projection dim for shape/trajectory
KSPEC = 40                    # population-spectrum bins
EXEMPLARS_PER_CLASS = 5       # full low-D trajectories kept for type analysis + animation


def make_cfg(arm, dataset, out_dir, seed, dwell):
    """Full image (no foveated crop). Three arms, ALL run with the model's default
    learning ON (via present(); we do NOT freeze the substrate):
      default     -- reservoir, legacy_multiplicative rule, default freeze (shipped model)
      reward_hebb -- reservoir UNFROZEN + native three-factor reward-gated rule (new plasticity)
      conv        -- convolutional-PAULA substrate (the deep conv architecture)
    perceive_frac=0.9 pushes the teacher reward late so we can read the attractor from the
    settled PERCEIVE window BEFORE the current image's reward (avoids trivial reward leakage)."""
    base = dict(dataset_name=dataset, dwell=dwell, seed=seed, conv_bank="rich", readout="all",
                fovea=32, periph=32, grid=16, perceive_frac=0.9, output_dir=out_dir)
    if arm in ("default", "reservoir_retino"):
        return MiniBrainConfig(substrate_type="reservoir", wiring="retinotopic",
                               plasticity_mode="legacy_multiplicative", **base)
    if arm == "reservoir_random":
        return MiniBrainConfig(substrate_type="reservoir", wiring="random",
                               plasticity_mode="legacy_multiplicative", **base)
    if arm == "reward_hebb":
        return MiniBrainConfig(substrate_type="reservoir", wiring="retinotopic",
                               plasticity_mode="reward_hebb", freeze_reservoir=False,
                               nm_kappa=1.0, rh_decay=0.1, **base)
    if arm == "conv":
        return MiniBrainConfig(substrate_type="conv",
                               plasticity_mode="legacy_multiplicative", **base)
    raise ValueError(f"unknown arm {arm}")


def warmup(brain, ds, order, n_warm):
    """Warm through the REAL default loop -- learning + teacher ON (no freezing)."""
    for i in order[:n_warm]:
        img, y = ds[int(i)]
        brain.present(img, int(y), learn=True, teach=True)


def _proj_matrix(N, seed):
    return (np.random.RandomState(seed + 99).randn(DR, N) / np.sqrt(N)).astype(np.float32)


def descriptor(cen, std, proj, spec):
    """Assemble the flat attractor descriptor and record component slices."""
    iu = np.triu_indices(DR)
    shape = np.cov(proj.T)[iu]                      # Dr*(Dr+1)/2
    return np.concatenate([cen, std, shape, spec]).astype(np.float32), \
        dict(cen=len(cen), std=len(std), shape=len(shape), spec=len(spec))


def shard_extract(arm, dataset, seed, idx_slice, exemplar_set, dwell, observe, spath, pos, desc):
    if os.path.exists(spath):
        return
    ds = load_dataset_by_name(dataset, train=True).dataset
    cfg = make_cfg(arm, dataset, spath + "_net", seed, dwell)
    brain = MiniBrain(cfg, load_dataset_by_name(dataset, train=True))
    warm_order = np.random.RandomState(seed + 5).randint(0, len(ds), size=30)
    warmup(brain, ds, warm_order, 30)               # real default loop, learning ON
    m = brain._readout_mask; N = int(m.sum()); R = _proj_matrix(N, seed)
    split = max(1, int(dwell * getattr(cfg, "perceive_frac", 0.9)))
    lo = max(0, split - observe)                     # observe the settled late-PERCEIVE window
    D, Y, slc = [], [], None
    ex_traj, ex_idx = [], []
    for i in tqdm(idx_slice, desc=desc, position=pos, ncols=90, leave=False):
        img, y = ds[int(i)]; Y.append(int(y))
        # present through the REAL default loop: learning + teacher ON; read the trajectory
        _, _, states = brain.present(img, int(y), learn=True, teach=True, return_states=True)
        obs = states[lo:split]                       # settled response, before this image's reward
        O = np.asarray([(s.O[m] > 0) for s in obs], np.float32)   # observe x N
        cen = O.mean(0); std = O.std(0)
        proj = O @ R.T                              # observe x Dr
        pop = O.mean(1)                             # population rate over time
        mag = np.abs(np.fft.rfft(pop - pop.mean()))
        spec = np.interp(np.linspace(0, len(mag) - 1, KSPEC), np.arange(len(mag)), mag).astype(np.float32)
        d, slc = descriptor(cen, std, proj, spec)
        D.append(d)
        if int(i) in exemplar_set:
            ex_traj.append(proj.astype(np.float32)); ex_idx.append((int(i), int(y)))
    np.savez_compressed(spath, D=np.asarray(D, np.float32), Y=np.asarray(Y),
                        ex_traj=np.asarray(ex_traj, np.float32) if ex_traj else np.zeros((0, observe, DR), np.float32),
                        ex_idx=np.asarray(ex_idx, np.int64) if ex_idx else np.zeros((0, 2), np.int64),
                        slc=np.asarray([slc["cen"], slc["std"], slc["shape"], slc["spec"]]))


# ----------------------------- attractor TYPE ------------------------------ #
def analyze_type(ex_traj, ex_idx, observe, out, tag):
    """Per exemplar low-D trajectory (observe x Dr): spectrum, Poincare return map,
    recurrence rate, autocorr period, largest-Lyapunov (Rosenstein). Classify the type."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib import animation
    types = []; rows = []
    # PCA basis over all exemplar points for consistent 3D view
    allpts = ex_traj.reshape(-1, DR); mu = allpts.mean(0)
    U, S, Vt = np.linalg.svd(allpts - mu, full_matrices=False); comps = Vt[:3]
    cmap = plt.get_cmap("tab10")

    def lyap_rosenstein(x, m=3, tau=2, k=8):
        x = (x - x.mean()) / (x.std() + 1e-9); n = len(x) - (m - 1) * tau
        if n < 20:
            return 0.0
        emb = np.stack([x[i * tau:i * tau + n] for i in range(m)], 1)
        d = np.linalg.norm(emb[:, None] - emb[None], axis=2)
        np.fill_diagonal(d, np.inf)
        for i in range(len(d)):
            lo = max(0, i - k); hi = min(len(d), i + k)
            d[i, lo:hi] = np.inf
        nn = d.argmin(1); div = []
        for j in range(1, min(20, n - 1)):
            dl = []
            for i in range(len(nn) - j):
                if nn[i] + j < len(emb):
                    dl.append(np.linalg.norm(emb[i + j] - emb[nn[i] + j]))
            if dl:
                div.append(np.log(np.mean(dl) + 1e-12))
        if len(div) < 3:
            return 0.0
        return float(np.polyfit(np.arange(len(div)), div, 1)[0])

    for e in range(len(ex_traj)):
        tr = ex_traj[e]; y = int(ex_idx[e][1])
        pc = (tr - mu) @ comps.T                    # observe x 3
        p0 = pc[:, 0]
        # spectrum of PC1
        mag = np.abs(np.fft.rfft(p0 - p0.mean())); freqs = np.fft.rfftfreq(len(p0))
        mag[0] = 0; peak = mag.argmax()
        flat = float(np.exp(np.mean(np.log(mag + 1e-9))) / (mag.mean() + 1e-9))  # spectral flatness
        npk = int((mag > 0.5 * mag.max()).sum())    # # strong peaks
        # autocorr period
        ac = np.correlate(p0 - p0.mean(), p0 - p0.mean(), "full")[len(p0) - 1:]
        ac = ac / (ac[0] + 1e-9); period = int(np.argmax(ac[5:] > 0.5) + 5) if (ac[5:] > 0.5).any() else 0
        lyap = lyap_rosenstein(p0)
        # Poincare: crossings of PC1 = mean (upward), record (PC2,PC3)
        thr = p0.mean(); cross = np.where((p0[:-1] < thr) & (p0[1:] >= thr))[0]
        poincare = pc[cross, 1:] if len(cross) else np.zeros((0, 2))
        # recurrence rate at 10th-percentile distance
        dm = np.linalg.norm(pc[:, None] - pc[None], axis=2)
        rr = float((dm < np.percentile(dm, 10)).mean())
        # classify type
        if lyap > 0.02 and flat > 0.3:
            typ = "chaotic"
        elif npk <= 2 and flat < 0.2:
            typ = "limit-cycle"
        elif 2 <= npk <= 4:
            typ = "quasi-periodic"
        else:
            typ = "complex/weak-cycle"
        types.append(typ)
        rows.append(dict(cls=y, type=typ, lyap=round(lyap, 4), flatness=round(flat, 3),
                         n_peaks=npk, period=period, recurrence=round(rr, 3),
                         n_poincare=int(len(poincare))))

    # ---- figures ----
    # (a) per-class spectra (mean over exemplars)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for c in range(10):
        idc = [e for e in range(len(ex_traj)) if int(ex_idx[e][1]) == c]
        if not idc:
            continue
        specs = []
        for e in idc:
            pc = (ex_traj[e] - mu) @ comps.T
            mg = np.abs(np.fft.rfft(pc[:, 0] - pc[:, 0].mean())); mg[0] = 0
            specs.append(mg / (mg.max() + 1e-9))
        ax.plot(np.fft.rfftfreq(observe), np.mean(specs, 0), color=cmap(c), lw=1.1, label=str(c))
    ax.set_xlabel("frequency (1/tick)"); ax.set_ylabel("norm power (PC1)")
    ax.set_title(f"{tag}: attractor power spectra by class"); ax.legend(fontsize=7, ncol=5)
    fig.tight_layout(); fig.savefig(os.path.join(out, f"{tag}_spectra.png"), dpi=130); plt.close(fig)

    # (b) Poincare maps (all exemplars, colored by class)
    fig, ax = plt.subplots(figsize=(7, 6.5))
    for e in range(len(ex_traj)):
        tr = ex_traj[e]; pc = (tr - mu) @ comps.T; p0 = pc[:, 0]; thr = p0.mean()
        cross = np.where((p0[:-1] < thr) & (p0[1:] >= thr))[0]
        if len(cross):
            ax.scatter(pc[cross, 1], pc[cross, 2], s=10, color=cmap(int(ex_idx[e][1])), alpha=0.6)
    ax.set_xlabel("PC2"); ax.set_ylabel("PC3")
    ax.set_title(f"{tag}: Poincare first-return (PC1=mean plane)\nfew points=limit cycle, loop=torus, scatter=chaos")
    fig.tight_layout(); fig.savefig(os.path.join(out, f"{tag}_poincare.png"), dpi=130); plt.close(fig)

    # (c) recurrence plot for one exemplar per (first 6) classes
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for a_i, ax in enumerate(axes.ravel()):
        idc = [e for e in range(len(ex_traj)) if int(ex_idx[e][1]) == a_i]
        if not idc:
            ax.axis("off"); continue
        pc = (ex_traj[idc[0]] - mu) @ comps.T
        dm = np.linalg.norm(pc[:, None] - pc[None], axis=2)
        ax.imshow(dm < np.percentile(dm, 12), cmap="binary", origin="lower")
        ax.set_title(f"class {a_i} ({types[idc[0]]})", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{tag}: recurrence plots (diagonal bands = periodicity)")
    fig.tight_layout(); fig.savefig(os.path.join(out, f"{tag}_recurrence.png"), dpi=130); plt.close(fig)

    # (d) rotating 3D attractor overlay (all exemplars, colored by class)
    fig = plt.figure(figsize=(8, 7)); ax = fig.add_subplot(111, projection="3d")
    for e in range(len(ex_traj)):
        pc = (ex_traj[e] - mu) @ comps.T
        ax.plot(pc[:, 0], pc[:, 1], pc[:, 2], color=cmap(int(ex_idx[e][1])), lw=0.6, alpha=0.7)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    ax.set_title(f"{tag}: attractors (colored by class)")

    def rot(i):
        ax.view_init(elev=20, azim=i * 4); return []
    anim = animation.FuncAnimation(fig, rot, frames=90, interval=60, blit=False)
    _save_anim(anim, os.path.join(out, f"{tag}_attractors_rotate")); plt.close(fig)

    from collections import Counter
    type_counts = dict(Counter(types))
    return dict(per_exemplar=rows, type_counts=type_counts,
                dominant_type=max(type_counts, key=type_counts.get) if type_counts else "n/a")


# --------------------- UNIFORMITY & CLASSIFICATION ------------------------- #
def analyze(D, Y, slc, out, tag, seed):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import silhouette_score, confusion_matrix
    cen_n, std_n, shp_n, spec_n = slc
    sl = dict(centroid=slice(0, cen_n), std=slice(cen_n, cen_n + std_n),
              shape=slice(cen_n + std_n, cen_n + std_n + shp_n),
              spec=slice(cen_n + std_n + shp_n, None))
    n = len(Y); rng = np.random.RandomState(0); perm = rng.permutation(n)
    D = D[perm]; Y = Y[perm]; ntr = int(n * 0.7)
    mu, sd = D[:ntr].mean(0), D[:ntr].std(0) + 1e-6; Dn = (D - mu) / sd

    # ---- within/between class geometry on the FULL descriptor ----
    cents = np.stack([D[Y == c].mean(0) for c in range(10)])
    within = np.mean([np.linalg.norm(D[i] - cents[Y[i]]) for i in range(n)])
    between = np.mean([np.linalg.norm(cents[i] - cents[j]) for i in range(10) for j in range(i + 1, 10)])
    fisher = float(between / (within + 1e-9))
    try:
        sil = float(silhouette_score(Dn[:3000], Y[:3000]))
    except Exception:
        sil = float("nan")

    # ---- classification: full + ablations, linear / kNN / nearest-centroid ----
    def clf(feat):
        Xtr, Xte, ytr, yte = feat[:ntr], feat[ntr:], Y[:ntr], Y[ntr:]
        m, s = Xtr.mean(0), Xtr.std(0) + 1e-6; Xtr, Xte = (Xtr - m) / s, (Xte - m) / s
        lin = max(LogisticRegression(max_iter=1200, C=C).fit(Xtr, ytr).score(Xte, yte) for C in (0.01, 0.05, 0.2))
        knn = KNeighborsClassifier(min(15, max(1, len(ytr) - 1))).fit(Xtr, ytr).score(Xte, yte)
        ctr = np.stack([Xtr[ytr == c].mean(0) for c in range(10)])
        nc = float((np.argmin(np.linalg.norm(Xte[:, None] - ctr[None], axis=2), 1) == yte).mean())
        return dict(linear=round(lin, 3), knn=round(knn, 3), nearest_centroid=round(nc, 3))

    res = {"full": clf(D)}
    for name, s in sl.items():
        res[name] = clf(D[:, s])
    res["centroid+shape+spec"] = clf(np.concatenate([D[:, sl["centroid"]], D[:, sl["shape"]], D[:, sl["spec"]]], 1))

    # confusion matrix on best full-linear
    Xtr, Xte = Dn[:ntr], Dn[ntr:]
    best = LogisticRegression(max_iter=1200, C=0.05).fit(Xtr, Y[:ntr])
    cm = confusion_matrix(Y[ntr:], best.predict(Xte), labels=list(range(10)))

    # ---- figures ----
    # class-mean descriptor distance heatmap
    dmat = np.linalg.norm(cents[:, None] - cents[None], axis=2)
    fig, ax = plt.subplots(figsize=(6.2, 5.4)); im = ax.imshow(dmat, cmap="magma")
    ax.set_title(f"{tag}: class-attractor centroid distances"); ax.set_xticks(range(10)); ax.set_yticks(range(10))
    fig.colorbar(im, ax=ax, shrink=0.8); fig.tight_layout()
    fig.savefig(os.path.join(out, f"{tag}_classdist.png"), dpi=130); plt.close(fig)

    # 2D PCA embedding of all attractors colored by class
    Dc = D - D.mean(0); U, S, Vt = np.linalg.svd(Dc[:4000], full_matrices=False)
    emb = Dc @ Vt[:2].T
    fig, ax = plt.subplots(figsize=(7.5, 6.8)); cmap = plt.get_cmap("tab10")
    for c in range(10):
        ax.scatter(emb[Y == c, 0], emb[Y == c, 1], s=5, color=cmap(c), alpha=0.4, label=str(c))
    ax.set_title(f"{tag}: attractor-descriptor embedding (Fisher={fisher:.2f}, sil={sil:.2f})")
    ax.legend(fontsize=7, ncol=5, markerscale=2); fig.tight_layout()
    fig.savefig(os.path.join(out, f"{tag}_embedding.png"), dpi=130); plt.close(fig)

    # confusion matrix + accuracy-vs-baseline bar
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    im = a1.imshow(cm, cmap="Blues"); a1.set_title(f"{tag}: confusion (full-linear {res['full']['linear']:.3f})")
    a1.set_xlabel("pred"); a1.set_ylabel("true"); a1.set_xticks(range(10)); a1.set_yticks(range(10))
    fig.colorbar(im, ax=a1, shrink=0.8)
    names = ["centroid\n(=mean)", "std", "shape", "spec", "cen+shape+spec", "FULL"]
    keys = ["centroid", "std", "shape", "spec", "centroid+shape+spec", "full"]
    vals = [res[k]["linear"] for k in keys]
    bars = a2.bar(names, vals, color=["#6b7280", "#94a3b8", "#0d818f", "#b45309", "#7c3aed", "#059669"])
    a2.axhline(0.383, color="k", ls=":", lw=1); a2.text(0, 0.39, "retina ref 0.38", fontsize=8)
    a2.axhline(0.1, color="gray", ls=":", lw=1); a2.set_ylabel("linear held-out acc")
    a2.set_title("attractor-descriptor ablation (does dynamics beat the mean?)")
    for b, v in zip(bars, vals):
        a2.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.2f}", ha="center", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, f"{tag}_classify.png"), dpi=130); plt.close(fig)

    return dict(fisher_ratio=round(fisher, 3), silhouette=round(sil, 3),
                within_class_dist=round(float(within), 3), between_class_dist=round(float(between), 3),
                classification=res,
                key_delta_full_minus_centroid=round(res["full"]["linear"] - res["centroid"]["linear"], 3))


def _save_anim(anim, base):
    try:
        anim.save(base + ".mp4", writer="ffmpeg", fps=20, dpi=110); print(f"saved {base}.mp4", flush=True); return
    except Exception as e:
        print(f"ffmpeg failed ({e}); gif", flush=True)
    try:
        anim.save(base + ".gif", writer="pillow", fps=15); print(f"saved {base}.gif", flush=True)
    except Exception as e:
        print(f"gif failed ({e})", flush=True)


def run_arm(arch, dataset, per_class, dwell, observe, seed, shards, out, pres_order="grouped"):
    tag = f"attr_{arch}_{dataset.replace('cifar10', 'c10')}"
    if pres_order != "grouped":
        tag += f"_{pres_order}"   # keep interleaved outputs from colliding with the grouped run
    jpath = os.path.join(out, f"{tag}.json")
    if os.path.exists(jpath):
        try:
            if json.load(open(jpath)).get("complete"):
                print(f"[skip] {tag}", flush=True); return
        except Exception:
            pass
    t0 = time.time()
    ds = load_dataset_by_name(dataset, train=True).dataset
    # pick per_class images/label + a small exemplar set/label
    by_cls = {c: [] for c in range(10)}
    order = np.random.RandomState(seed + 5).permutation(len(ds))
    for i in order:
        _, y = ds[int(i)]; y = int(y)
        if len(by_cls[y]) < per_class:
            by_cls[y].append(int(i))
        if all(len(v) >= per_class for v in by_cls.values()):
            break
    if pres_order == "interleaved":
        # round-robin over classes: every 10-image window spans all classes, so no class-block
        # can form -> disentangles learned class structure from "current-block-tuning" drift.
        kmax = max(len(v) for v in by_cls.values())
        idxs = np.array([by_cls[c][k] for k in range(kmax) for c in range(10) if k < len(by_cls[c])])
    else:  # "grouped" (default, legacy): all class-0, then class-1, ...
        idxs = np.array([i for c in range(10) for i in by_cls[c]])
    exemplars = set(int(by_cls[c][k]) for c in range(10) for k in range(min(EXEMPLARS_PER_CLASS, len(by_cls[c]))))
    sdir = os.path.join(out, tag + "_shards"); os.makedirs(sdir, exist_ok=True)
    slices = np.array_split(idxs, shards); procs = []
    for si, sl in enumerate(slices):
        spath = os.path.join(sdir, f"sh{si}.npz")
        pr = Process(target=shard_extract, args=(arch, dataset, seed, sl, exemplars, dwell,
                                                 observe, spath, si, f"{tag} sh{si}"))
        pr.start(); procs.append(pr)
    for pr in procs:
        pr.join()
    parts = [np.load(os.path.join(sdir, f"sh{si}.npz")) for si in range(shards)]
    D = np.concatenate([p["D"] for p in parts]); Y = np.concatenate([p["Y"] for p in parts])
    ex_traj = np.concatenate([p["ex_traj"] for p in parts]) if sum(len(p["ex_traj"]) for p in parts) else np.zeros((0, observe, DR), np.float32)
    ex_idx = np.concatenate([p["ex_idx"] for p in parts]) if len(ex_traj) else np.zeros((0, 2), np.int64)
    slc = parts[0]["slc"]
    print(f"[{tag}] extracted {len(Y)} descriptors dim={D.shape[1]}, {len(ex_traj)} exemplar trajs "
          f"({time.time()-t0:.0f}s)", flush=True)

    typ = analyze_type(ex_traj, ex_idx, observe, out, tag) if len(ex_traj) else {}
    ana = analyze(D, Y, slc, out, tag, seed)
    json.dump(dict(exp="attractor_class", tag=tag, arch=arch, dataset=dataset,
                   per_class=per_class, dwell=dwell, observe=observe, n=int(len(Y)),
                   desc_dim=int(D.shape[1]), type=typ, analysis=ana,
                   secs=round(time.time() - t0), complete=True), open(jpath, "w"))
    print(f"[{tag}] TYPE={typ.get('dominant_type')} | Fisher={ana['fisher_ratio']} sil={ana['silhouette']} | "
          f"attractor-classify full-linear={ana['classification']['full']['linear']} "
          f"(centroid/mean={ana['classification']['centroid']['linear']}, "
          f"delta={ana['key_delta_full_minus_centroid']:+.3f})", flush=True)
    for f in glob.glob(os.path.join(sdir, "*.npz")):
        os.remove(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--arch",
        default="default",
        choices=["default", "reservoir_retino", "reservoir_random", "reward_hebb", "conv"],
        help="default/reward_hebb/conv plus recovered reservoir aliases",
    )
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--per-class", type=int, default=100)
    p.add_argument("--dwell", type=int, default=2000); p.add_argument("--observe", type=int, default=1000)
    p.add_argument("--shards", type=int, default=9); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="foveation_results/minibrain/attractor_learn")
    p.add_argument("--order", default="grouped", choices=["grouped", "interleaved"],
                   help="grouped=legacy class-blocked order; interleaved=round-robin control for block-tuning confound")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    run_arm(a.arch, a.dataset, a.per_class, a.dwell, a.observe, a.seed, a.shards, a.out, pres_order=a.order)


if __name__ == "__main__":
    main()
