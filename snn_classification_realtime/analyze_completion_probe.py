"""Analyze Test-A pattern-completion runs (run_completion_probe *.npz).

Per (arm, probe-mode, occlusion condition) it forms the held-out completion-accuracy
trajectory vs n_train, pooled across seeds, and reports:
  * baseline (n_train=0) and final accuracy vs chance (1/#classes),
  * the SLOPE of accuracy vs n_train with a bootstrap 95% CI over seeds (the sharpening test),
  * the plastic-arm vs negative-control-arm slope difference (paired by seed),
  * completion index = sim(occluded)/sim(whole) to own centroid.

Success (pre-registered): for a plastic arm at an occluded condition, slope CI excludes 0 AND
slope(plastic) − slope(control) CI excludes 0. Also emits per-condition curves and, for one
chosen arm/mode/condition, an animation of the occluded QUERY descriptors migrating toward
their class centroids as training proceeds. CLI-only.
"""
import argparse
import glob
import json
import os

import numpy as np


def load_all(d):
    runs = {}
    for p in sorted(glob.glob(os.path.join(d, "*.npz"))):
        z = np.load(p, allow_pickle=True)
        tag = os.path.basename(p)[:-4]
        # tag convention: <arm>_s<seed>  (e.g. rh05_s0)
        arm = tag.split("_s")[0]
        runs.setdefault(arm, []).append(z)
    return runs


def nc_acc(ref_D, ref_y, q_D, q_y):
    m, sd = ref_D.mean(0), ref_D.std(0) + 1e-9
    R = (ref_D - m) / sd; Q = (q_D - m) / sd
    classes = sorted(set(ref_y.tolist()))
    cents = np.stack([R[ref_y == c].mean(0) for c in classes])
    d = np.linalg.norm(Q[:, None, :] - cents[None], axis=2)
    pred = np.array(classes)[d.argmin(1)]
    return float((pred == q_y).mean())


def _slice(D, z, space):
    sd = int(z["static_dim"]) if "static_dim" in z else D.shape[-1]
    return D[..., :sd] if space == "static" else D[..., sd:]


def traj(z, mode, cond, space="static"):
    """held-out completion-accuracy trajectory over checkpoints, in the chosen descriptor
    space ('static' = Test A mean/tref/spec block; 'dyn' = Test B AC-power/freq/centroid)."""
    ref = _slice(z[f"ref_{mode}"], z, space); ref_y = z["ref_lab"]; q_y = z["q_lab"]
    f, r = cond.rsplit("_", 1)
    q = _slice(z[f"q_{mode}__{f}_{r}"], z, space)
    return np.array([nc_acc(ref[c], ref_y, q[c], q_y) for c in range(len(z["n_train"]))])


def slope_ci(nts, mat, nboot=2000, seed=0):
    """mat: (nseed, ncheck) accuracy. Bootstrap slope CI over seeds."""
    rng = np.random.RandomState(seed)
    def sl(rows):
        y = rows.mean(0)
        return np.polyfit(nts, y, 1)[0] * 100  # per 100 training samples
    point = sl(mat)
    if len(mat) < 2:
        return point, (float("nan"), float("nan"))
    boots = [sl(mat[rng.randint(0, len(mat), len(mat))]) for _ in range(nboot)]
    return point, (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="foveation_results/completion_probe")
    ap.add_argument("--out", default=None)
    ap.add_argument("--control-arm", default="legacy")
    ap.add_argument("--anim-arm", default=None)
    ap.add_argument("--anim-mode", default="frozen")
    ap.add_argument("--anim-cond", default="crop_50")
    a = ap.parse_args()
    out = a.out or a.dir
    os.makedirs(out, exist_ok=True)
    runs = load_all(a.dir)
    if not runs:
        print(f"no npz in {a.dir}"); return
    any_z = runs[next(iter(runs))][0]
    modes = [str(m) for m in any_z["modes"]]
    conds = [str(c) for c in any_z["conditions"]]
    ncl = len(any_z["classes"]); chance = 1.0 / ncl
    nts = np.asarray(any_z["n_train"])

    spaces = ["static", "dyn"]  # Test A (mean/tref/spec) and Test B (dynamical features)
    epochs = int(any_z["epochs"]) if "epochs" in any_z else 1
    train_len = int(any_z["train_len"]) if "train_len" in any_z else None
    # accuracy tensors: space -> arm -> mode -> cond -> (nseed, ncheck)
    acc = {sp: {} for sp in spaces}
    for sp in spaces:
        for arm, zs in runs.items():
            acc[sp][arm] = {}
            for m in modes:
                acc[sp][arm][m] = {}
                for c in conds:
                    mat = np.stack([traj(z, m, c, sp) for z in zs if z["n_train"].shape == nts.shape])
                    acc[sp][arm][m][c] = mat

    ctrl = a.control_arm
    summary = {"chance": chance, "n_train": nts.tolist(), "epochs": epochs,
               "train_len": train_len, "spaces": {}}
    for sp in spaces:
        summary["spaces"][sp] = {"arms": {}}
        title = "Test A: static (mean) descriptor" if sp == "static" else "Test B: dynamical descriptor"
        print(f"\n=== {title} — completion sharpening (chance={chance:.3f}, "
              f"{len(nts)} checkpoints, epochs={epochs}, seeds/arm="
              f"{ {k: len(v) for k, v in runs.items()} }) ===")
        for arm in sorted(acc[sp]):
            summary["spaces"][sp]["arms"][arm] = {}
            print(f"[{arm}]")
            for m in modes:
                for c in conds:
                    mat = acc[sp][arm][m][c]
                    base = mat[:, 0].mean(); fin = mat[:, -3:].mean()
                    pt, (lo, hi) = slope_ci(nts, mat)
                    sig = "*" if (lo > 0 or hi < 0) else " "
                    print(f"  {m:6s} {c:10s} acc {base:.3f}->{fin:.3f}  slope {pt:+.3f} "
                          f"[{lo:+.3f},{hi:+.3f}]/100smp {sig}")
                    summary["spaces"][sp]["arms"][arm].setdefault(m, {})[c] = dict(
                        base=round(float(base), 3), final=round(float(fin), 3),
                        slope=round(pt, 4), ci=[round(lo, 4), round(hi, 4)])
        # plastic vs negative-control slope difference (the anti-artifact test)
        if ctrl in acc[sp]:
            print(f"  -- plastic vs control ({ctrl}) Δslope (occluded conds; CI excl 0 = real) --")
            for arm in sorted(acc[sp]):
                if arm == ctrl:
                    continue
                for m in modes:
                    for c in conds:
                        if c == "whole_0":
                            continue
                        A = acc[sp][arm][m][c]; B = acc[sp][ctrl][m][c]
                        n = min(len(A), len(B))
                        pt, (lo, hi) = slope_ci(nts, A[:n] - B[:n])
                        sig = "*" if (lo > 0 or hi < 0) else " "
                        print(f"     {arm} vs {ctrl} {m:6s} {c:10s} Δslope {pt:+.3f} [{lo:+.3f},{hi:+.3f}] {sig}")
    json.dump(summary, open(os.path.join(out, "completion_summary.json"), "w"), indent=2)

    # ---- per-condition curves, one figure per space ----
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    cmap = plt.get_cmap("tab10")
    for sp in spaces:
        fig, axs = plt.subplots(1, len(conds), figsize=(5 * len(conds), 4.4), squeeze=False)
        for ci, c in enumerate(conds):
            ax = axs[0][ci]
            for ai, arm in enumerate(sorted(acc[sp])):
                for m, ls in zip(modes, ["-", "--"]):
                    mat = acc[sp][arm][m][c]; mean = mat.mean(0)
                    ax.plot(nts, mean, ls, color=cmap(ai), label=f"{arm} {m}")
                    if len(mat) > 1:
                        se = mat.std(0) / np.sqrt(len(mat))
                        ax.fill_between(nts, mean - se, mean + se, color=cmap(ai), alpha=0.15)
            ax.axhline(chance, color="k", ls=":", lw=1)
            if train_len and epochs > 1:
                for e in range(1, epochs):
                    ax.axvline(e * train_len, color="grey", ls="-", lw=0.6, alpha=0.5)
            ax.set_title(c); ax.set_xlabel("training samples seen")
            if ci == 0:
                ax.set_ylabel(f"{sp} completion acc"); ax.legend(fontsize=7)
        fig.suptitle(f"{'Test A static' if sp=='static' else 'Test B dynamical'}: "
                     f"occluded held-out completion vs training (epoch boundaries = grey)")
        fig.tight_layout(); fig.savefig(os.path.join(out, f"completion_curves_{sp}.png"), dpi=130); plt.close(fig)
    print(f"\nwrote completion_curves_static.png, completion_curves_dyn.png, completion_summary.json")

    # ---- animation: occluded QUERY migrating toward class centroids ----
    anim_arm = a.anim_arm or sorted(acc)[0]
    z = runs[anim_arm][0]
    _animate(z, anim_arm, a.anim_mode, a.anim_cond, out)


def _animate(z, arm, mode, cond, out):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    f, r = cond.rsplit("_", 1)
    try:
        ref = z[f"ref_{mode}"]; q = z[f"q_{mode}__{f}_{r}"]
    except KeyError:
        print(f"anim: {mode}/{cond} not in run; skipping"); return
    ref_y = z["ref_lab"]; q_y = z["q_lab"]; nts = z["n_train"]; classes = z["classes"].tolist()
    C = len(nts)
    # global PCA on pooled ref+query (standardized)
    allD = np.concatenate([ref.reshape(-1, ref.shape[-1]), q.reshape(-1, q.shape[-1])], 0)
    mu, sd = allD.mean(0), allD.std(0) + 1e-9
    gm = ((allD - mu) / sd).mean(0)
    _, _, Vt = np.linalg.svd((allD - mu) / sd - gm, full_matrices=False); V = Vt[:2]
    def proj(D): return ((D - mu) / sd - gm) @ V.T
    cmap = plt.get_cmap("tab10")
    col = {c: cmap(i % 10) for i, c in enumerate(classes)}
    allE = proj(allD); pad = 0.08 * (allE.max(0) - allE.min(0) + 1e-9)
    xlim = (allE[:, 0].min() - pad[0], allE[:, 0].max() + pad[0])
    ylim = (allE[:, 1].min() - pad[1], allE[:, 1].max() + pad[1])
    fig, ax = plt.subplots(figsize=(7.5, 7))

    def draw(ci):
        ax.clear()
        Rc = proj(ref[ci]); Qc = proj(q[ci])
        for c in classes:
            rm = ref_y == c
            if rm.any():
                ct = Rc[rm].mean(0)
                ax.scatter(*ct, color=col[c], s=320, marker="*", edgecolors="black", lw=1, zorder=5)
                ax.annotate(str(c), ct, fontsize=11, fontweight="bold", ha="center", va="center", zorder=6)
        ax.scatter(Qc[:, 0], Qc[:, 1], c=[col[int(y)] for y in q_y], s=42,
                   marker="o", edgecolors="white", lw=0.5, zorder=4)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{arm} {mode} {cond}: occluded held-out (o) vs class centroids (*)\n"
                     f"n_train={nts[ci]}")
    anim = animation.FuncAnimation(fig, draw, frames=C, interval=350)
    mp4 = os.path.join(out, f"completion_{arm}_{mode}_{cond}.mp4")
    try:
        anim.save(mp4, writer=animation.FFMpegWriter(fps=3, bitrate=2400), dpi=120); print(f"wrote {mp4}")
    except Exception as e:
        gif = mp4[:-4] + ".gif"; anim.save(gif, writer=animation.PillowWriter(fps=3), dpi=120)
        print(f"ffmpeg failed ({e}); wrote {gif}")
    plt.close(fig)


if __name__ == "__main__":
    main()
