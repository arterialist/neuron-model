"""Part C -- attractor / limit-cycle analysis of the substrate dynamics.

Records the FULL state trajectory (only for a handful of exemplars per class, so storage
is tens of MB) while the network settles on a fixed image, then asks:

  1. FIXED POINT vs LIMIT CYCLE: does state velocity ||x(t)-x(t-1)|| decay to ~0 (the
     network relaxes to a fixed point) or settle to a nonzero periodic value (limit
     cycle)? Plotted per class.
  2. CLASS GEOMETRY: are trajectories from the same class in the same region of state
     space? PCA to 2D/3D, colored by class; centroid separation vs within-class spread.
  3. TIME MATTERS?: is the LATE-window state more class-separable than the EARLY window
     (ties Part C back to Part B's settling sweep).
  4. RE-ENTRY / MEMORY: present image A, let activity fade under tonic-only drive, then
     re-present A -- does the state return to A's attractor region? (the savings mechanism
     the spatiotemporal-representation thesis predicts: the remnant in weights/t_ref
     re-enters the same regime). Compared against re-entry to a DIFFERENT image B.

Artifacts (in --out):
  attractor_velocity.png        -- velocity(t) per class (fixed-point vs limit-cycle)
  attractor_pca2d.png           -- PC1-PC2 phase portrait, all trajectories
  attractor_pca3d.png           -- 3D snapshot
  attractor_rotate.mp4/.gif     -- rotating 3D trajectories
  attractor_unfold.mp4/.gif     -- 2D trajectories drawing over time
  attractor_reentry.png         -- state distance to A-attractor across present/fade/re-present
  attractor_summary.json        -- numeric diagnoses

    PYTHONPATH=. .venv/bin/python -m \
        snn_classification_realtime.foveation.minibrain.exp_attractor --arch reservoir_retino
"""
from __future__ import annotations

import argparse, os, json, time
import numpy as np

from snn_classification_realtime.activity_dataset_builder.vision_datasets import load_dataset_by_name
from snn_classification_realtime.foveation.minibrain.core import MiniBrainConfig, MiniBrain


def make_cfg(arch, dataset, out_dir, seed, dwell):
    substrate = "conv" if arch == "conv" else "reservoir"
    wiring = "retinotopic" if arch == "reservoir_retino" else "random"
    return MiniBrainConfig(dataset_name=dataset, dwell=dwell, seed=seed, substrate_type=substrate,
                           wiring=wiring, conv_bank="rich", readout="all",
                           fovea=32, periph=32, grid=16, output_dir=out_dir)


def warmup(brain, ds, ticks, seed):
    order = np.random.RandomState(seed).randint(0, len(ds), size=50)
    per = max(1, brain.cfg.input_period); done = 0; i = 0
    brain.sub.set_learning(False)
    while done < ticks:
        sig = brain.sub.patch_to_signals(brain._encode(ds[int(order[i % 50])][0]))
        for t in range(brain.cfg.dwell):
            brain.sub.step((sig if t % per == 0 else []) + brain._tonic); done += 1
            if done >= ticks:
                break
        i += 1
    brain.sub.set_learning(True)


def record_traj(brain, img, ticks, drive=True):
    """Return firing-rate trajectory smoothed as (ticks, N) over the readout layer. If
    drive=False, present only the tonic background (activity fades / free relaxation)."""
    m = brain._readout_mask; per = max(1, brain.cfg.input_period); N = int(m.sum())
    sig = brain.sub.patch_to_signals(brain._encode(img)) if drive else []
    O = np.zeros((ticks, N), np.float32)
    brain.sub.set_learning(False)
    for t in range(ticks):
        inp = (sig if (drive and t % per == 0) else []) + brain._tonic
        st = brain.sub.step(inp)
        O[t] = (st.O[m] > 0)
    brain.sub.set_learning(True)
    return O


def smooth(O, w=15):
    """Boxcar-smoothed firing rate -> continuous state trajectory."""
    k = np.ones(w) / w
    return np.stack([np.convolve(O[:, j], k, mode="same") for j in range(O.shape[1])], axis=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--arch", default="reservoir_retino")
    p.add_argument("--per-class", type=int, default=3)
    p.add_argument("--ticks", type=int, default=1000)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="foveation_results/minibrain/attractor")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True); t0 = time.time()
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    dstr = load_dataset_by_name(a.dataset, train=True)
    ds = dstr.dataset; dste = load_dataset_by_name(a.dataset, train=False).dataset
    cfg = make_cfg(a.arch, a.dataset, os.path.join(a.out, f"net_{a.arch}"), a.seed, a.ticks)
    brain = MiniBrain(cfg, dstr)
    warmup(brain, ds, a.warmup, a.seed)

    # pick exemplars: per_class images per label from the test set
    by_cls = {c: [] for c in range(10)}
    for i in range(len(dste)):
        _, y = dste[i]; y = int(y)
        if len(by_cls[y]) < a.per_class:
            by_cls[y].append(i)
        if all(len(v) >= a.per_class for v in by_cls.values()):
            break

    trajs = []; labels = []; imgs = []
    print(f"recording {10 * a.per_class} trajectories x {a.ticks} ticks ...", flush=True)
    for c in range(10):
        for i in by_cls[c]:
            img, _ = dste[i]
            O = record_traj(brain, img, a.ticks, drive=True)
            trajs.append(smooth(O)); labels.append(c); imgs.append(i)
    trajs = np.stack(trajs); labels = np.array(labels)  # (M, T, N)
    M, T, N = trajs.shape
    print(f"recorded {M} trajectories, N={N} neurons, T={T} ticks", flush=True)

    # ---- PCA on all trajectory points (drop first 50 ticks transient for the fit) ----
    flat = trajs[:, 50:, :].reshape(-1, N)
    mu = flat.mean(0); fc = flat - mu
    U, S, Vt = np.linalg.svd(fc, full_matrices=False)
    comps = Vt[:3]  # (3, N)
    evr = (S[:3] ** 2) / (S ** 2).sum()
    proj = (trajs - mu) @ comps.T  # (M, T, 3)

    cmap = plt.get_cmap("tab10")

    # ---- (1) velocity(t): fixed point vs limit cycle ----
    vel = np.linalg.norm(np.diff(trajs, axis=1), axis=2)  # (M, T-1)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for c in range(10):
        v = vel[labels == c].mean(0)
        ax.plot(v, color=cmap(c), lw=1.2, label=str(c))
    ax.set_xlabel("tick"); ax.set_ylabel("state velocity  ||x(t)-x(t-1)||")
    ax.set_title(f"{a.arch}: relaxation velocity (->0 = fixed point, plateau = limit cycle)")
    ax.legend(fontsize=7, ncol=5, title="class"); fig.tight_layout()
    fig.savefig(os.path.join(a.out, "attractor_velocity.png"), dpi=130); plt.close(fig)

    # velocity diagnosis: ratio of late to early velocity
    early = vel[:, 50:150].mean(); late = vel[:, -150:].mean()
    vel_ratio = float(late / (early + 1e-9))

    # ---- (2) 2D phase portrait ----
    fig, ax = plt.subplots(figsize=(7.5, 7))
    for c in range(10):
        for mi in np.where(labels == c)[0]:
            ax.plot(proj[mi, 50:, 0], proj[mi, 50:, 1], color=cmap(c), lw=0.8, alpha=0.7)
            ax.scatter(proj[mi, -1, 0], proj[mi, -1, 1], color=cmap(c), s=30, zorder=3,
                       edgecolor="k", linewidth=0.4)
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)"); ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
    ax.set_title(f"{a.arch}: state-space trajectories (dot = endpoint)")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "attractor_pca2d.png"), dpi=130); plt.close(fig)

    # ---- (3) 3D snapshot ----
    fig = plt.figure(figsize=(8, 7)); ax = fig.add_subplot(111, projection="3d")
    for c in range(10):
        for mi in np.where(labels == c)[0]:
            ax.plot(proj[mi, 50:, 0], proj[mi, 50:, 1], proj[mi, 50:, 2], color=cmap(c), lw=0.7, alpha=0.7)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    ax.set_title(f"{a.arch}: 3D trajectories ({evr[:3].sum()*100:.0f}% var)")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "attractor_pca3d.png"), dpi=130)

    # ---- (3b) rotating 3D animation ----
    def rot(i):
        ax.view_init(elev=20, azim=i * 3); return []
    anim = animation.FuncAnimation(fig, rot, frames=120, interval=50, blit=False)
    _save_anim(anim, os.path.join(a.out, "attractor_rotate"))
    plt.close(fig)

    # ---- (4) 2D unfolding animation ----
    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.set_xlim(proj[:, 50:, 0].min(), proj[:, 50:, 0].max())
    ax.set_ylim(proj[:, 50:, 1].min(), proj[:, 50:, 1].max())
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title(f"{a.arch}: trajectories unfolding")
    lines = [ax.plot([], [], color=cmap(labels[mi]), lw=1.0, alpha=0.8)[0] for mi in range(M)]
    heads = [ax.plot([], [], "o", color=cmap(labels[mi]), ms=4)[0] for mi in range(M)]
    txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=10)
    step = max(1, T // 120)

    def upd(f):
        tt = min(T, 50 + f * step)
        for mi in range(M):
            lines[mi].set_data(proj[mi, 50:tt, 0], proj[mi, 50:tt, 1])
            if tt > 50:
                heads[mi].set_data([proj[mi, tt - 1, 0]], [proj[mi, tt - 1, 1]])
        txt.set_text(f"tick {tt}")
        return lines + heads + [txt]
    anim2 = animation.FuncAnimation(fig, upd, frames=120, interval=60, blit=False)
    _save_anim(anim2, os.path.join(a.out, "attractor_unfold"))
    plt.close(fig)

    # ---- (2b) class geometry numbers: late-window centroid separation ratio ----
    late_state = trajs[:, -150:, :].mean(1)  # (M, N)
    cents = np.stack([late_state[labels == c].mean(0) for c in range(10)])
    between = np.mean([np.linalg.norm(cents[i] - cents[j])
                       for i in range(10) for j in range(i + 1, 10)])
    within = np.mean([np.linalg.norm(late_state[mi] - cents[labels[mi]]) for mi in range(M)])
    sep_late = float(between / (within + 1e-9))
    early_state = trajs[:, 50:200, :].mean(1)
    cents_e = np.stack([early_state[labels == c].mean(0) for c in range(10)])
    between_e = np.mean([np.linalg.norm(cents_e[i] - cents_e[j])
                         for i in range(10) for j in range(i + 1, 10)])
    within_e = np.mean([np.linalg.norm(early_state[mi] - cents_e[labels[mi]]) for mi in range(M)])
    sep_early = float(between_e / (within_e + 1e-9))

    # ---- (4) re-entry / memory test ----
    reentry = _reentry(brain, dste, by_cls, a.ticks, a.out, plt)

    summ = dict(arch=a.arch, dataset=a.dataset, n_traj=int(M), n_neurons=int(N), ticks=int(T),
                pca_evr=[round(float(x), 4) for x in evr],
                velocity_ratio_late_over_early=round(vel_ratio, 4),
                verdict_dynamics=("fixed-point-ish" if vel_ratio < 0.4 else
                                  "sustained (limit-cycle/chaotic)" if vel_ratio > 0.8 else
                                  "partially-relaxing"),
                class_sep_ratio_early=round(sep_early, 4),
                class_sep_ratio_late=round(sep_late, 4),
                time_helps_separation=bool(sep_late > sep_early),
                reentry=reentry)
    json.dump(summ, open(os.path.join(a.out, "attractor_summary.json"), "w"), indent=2)
    print(f"\n=== ATTRACTOR SUMMARY ({time.time()-t0:.0f}s) ===")
    print(json.dumps(summ, indent=2))
    print(f"artifacts in {a.out}/")


def _reentry(brain, dste, by_cls, ticks, out, plt):
    """Present A -> fade (tonic only) -> re-present A; track distance-to-A-attractor.
    Baseline: distance to a different image B's attractor. If re-entry to A is closer than
    to B, the network re-enters A's regime (savings)."""
    ia = by_cls[0][0]; ib = by_cls[1][0]
    imgA, _ = dste[ia]; imgB, _ = dste[ib]
    seg = ticks
    # establish attractor centroids (late window of a fresh presentation)
    Oa = smooth(record_traj(brain, imgA, seg, drive=True)); ca = Oa[-150:].mean(0)
    Ob = smooth(record_traj(brain, imgB, seg, drive=True)); cb = Ob[-150:].mean(0)
    # sequence: present A, fade, re-present A
    pa = smooth(record_traj(brain, imgA, seg, drive=True))
    fade = smooth(record_traj(brain, imgA, seg // 2, drive=False))
    ra = smooth(record_traj(brain, imgA, seg, drive=True))
    dist_a = np.linalg.norm(np.concatenate([pa, fade, ra]) - ca, axis=1)
    dist_b = np.linalg.norm(np.concatenate([pa, fade, ra]) - cb, axis=1)
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.plot(dist_a, color="#0d818f", label="dist to A attractor")
    ax.plot(dist_b, color="#b45309", label="dist to B attractor", alpha=0.7)
    b1 = seg; b2 = seg + seg // 2
    ax.axvspan(0, b1, color="#0d818f", alpha=0.06); ax.axvspan(b1, b2, color="gray", alpha=0.1)
    ax.axvspan(b2, len(dist_a), color="#0d818f", alpha=0.06)
    ax.text(b1 / 2, ax.get_ylim()[1] * 0.95, "present A", ha="center", fontsize=8)
    ax.text((b1 + b2) / 2, ax.get_ylim()[1] * 0.95, "fade", ha="center", fontsize=8)
    ax.text((b2 + len(dist_a)) / 2, ax.get_ylim()[1] * 0.95, "re-present A", ha="center", fontsize=8)
    ax.set_xlabel("tick"); ax.set_ylabel("state distance"); ax.legend(fontsize=8)
    ax.set_title("Re-entry: does re-presenting A return the state to A's attractor?")
    fig.tight_layout(); fig.savefig(os.path.join(out, "attractor_reentry.png"), dpi=130); plt.close(fig)
    # measure: mean distance-to-A in re-present window vs distance-to-B
    re = slice(b2, len(dist_a))
    return dict(reenter_dist_to_A=round(float(dist_a[re].mean()), 4),
                reenter_dist_to_B=round(float(dist_b[re].mean()), 4),
                reenters_A=bool(dist_a[re].mean() < dist_b[re].mean()))


def _save_anim(anim, base):
    """Try mp4 (ffmpeg) then gif (pillow)."""
    try:
        anim.save(base + ".mp4", writer="ffmpeg", fps=20, dpi=110)
        print(f"saved {base}.mp4", flush=True); return
    except Exception as e:
        print(f"ffmpeg failed ({e}); trying gif", flush=True)
    try:
        anim.save(base + ".gif", writer="pillow", fps=15)
        print(f"saved {base}.gif", flush=True)
    except Exception as e:
        print(f"gif failed ({e})", flush=True)


if __name__ == "__main__":
    main()
