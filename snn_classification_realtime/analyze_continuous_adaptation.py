"""Analyze continuous slow-adaptation runs: does sustained same-class exposure make successive
samples fall into progressively more similar attractors (class generalization), and which
plasticity mode / rh_decay enables it?

Reads run_continuous_adaptation *.npz for every arm x {blocked, interleaved} in a directory.
Per run computes, per class block (samples ordered by block-position k):
  - convergence: distance(sample_k, late-block centroid) vs k -> slope<0 means converging.
  - tightening : within-class descriptor spread, first-third vs last-third (ratio<1 = tightening).
  - separability: Fisher(between/within) on early (first third) vs late (last third) samples.
  - collapse guard: late between-class centroid distance (->0 would mean input-independent collapse).
  - weight integration: wtrace shape (integrating vs saturated/plateaued).
Then contrasts blocked vs interleaved (class-specific convergence should need sustained exposure)
and summarizes across arms. Produces figures + a cross-arm verdict JSON. CLI-only.
"""
import argparse
import glob
import json
import os

import numpy as np


def load_run(npz):
    z = np.load(npz, allow_pickle=True)
    return dict(desc=z["desc"], cls=z["cls"], pos=z["pos"], gpos=z["gpos"],
               wtrace=z["wtrace"], w0=float(z["w0"]), slc=json.loads(str(z["slc"])),
               classes=z["classes"].tolist(), order=str(z["order"]),
               ex_traj=z["ex_traj"], ex_meta=z["ex_meta"], settle=int(z["settle"]))


def znorm(D):
    return (D - D.mean(0)) / (D.std(0) + 1e-9)


def analyze_run(r):
    D = znorm(r["desc"]); cls = r["cls"]; pos = r["pos"]
    classes = sorted(set(cls.tolist()))
    per_class = {}
    conv_slopes, tighten, ncls = [], [], len(classes)
    for c in classes:
        m = cls == c
        Dc = D[m]; kc = pos[m]
        o = np.argsort(kc); Dc = Dc[o]; kc = kc[o]
        n = len(Dc)
        if n < 6:
            continue
        late = Dc[int(0.66 * n):].mean(0)
        dist = np.linalg.norm(Dc - late, axis=1)
        # convergence slope of distance vs block-position (normalized k)
        kk = (kc - kc.min()) / (kc.max() - kc.min() + 1e-9)
        slope = float(np.polyfit(kk, dist, 1)[0])
        conv_slopes.append(slope)
        early_spread = float(np.mean(np.linalg.norm(Dc[:n // 3] - Dc[:n // 3].mean(0), axis=1)))
        late_spread = float(np.mean(np.linalg.norm(Dc[-n // 3:] - Dc[-n // 3:].mean(0), axis=1)))
        tighten.append(late_spread / (early_spread + 1e-9))
        per_class[int(c)] = dict(conv_slope=round(slope, 4),
                                 tighten_ratio=round(late_spread / (early_spread + 1e-9), 3),
                                 dist_early=round(float(dist[:n // 3].mean()), 3),
                                 dist_late=round(float(dist[-n // 3:].mean()), 3))

    # early vs late separability (Fisher) using first/last third of each class block
    def fisher(sel):
        cents, within, allc = [], [], []
        for c in classes:
            m = cls == c
            Dc = D[m][np.argsort(pos[m])]
            n = len(Dc)
            sub = Dc[:n // 3] if sel == "early" else Dc[-n // 3:]
            if len(sub) < 2:
                return float("nan"), float("nan")
            ct = sub.mean(0); cents.append(ct)
            within.append(np.mean(np.linalg.norm(sub - ct, axis=1)))
        cents = np.stack(cents)
        betw = np.mean([np.linalg.norm(cents[i] - cents[j])
                        for i in range(len(cents)) for j in range(i + 1, len(cents))]) if len(cents) > 1 else 0.0
        return float(betw / (np.mean(within) + 1e-9)), float(betw)

    fe, be = fisher("early"); fl, bl = fisher("late")
    w = r["wtrace"]
    # weight integration: slope of second half (still climbing?) vs total change
    half = len(w) // 2
    w_slope_late = float(np.polyfit(np.arange(len(w) - half), w[half:], 1)[0]) if len(w) - half > 2 else 0.0
    return dict(
        order=r["order"],
        mean_conv_slope=round(float(np.mean(conv_slopes)), 4) if conv_slopes else None,
        mean_tighten_ratio=round(float(np.mean(tighten)), 3) if tighten else None,
        fisher_early=round(fe, 3), fisher_late=round(fl, 3),
        fisher_gain=round(fl - fe, 3), between_late=round(bl, 3),
        w0=round(r["w0"], 1), w_final=round(float(w[-1]), 1),
        w_late_slope=round(w_slope_late, 3), per_class=per_class,
    )


def figures(arm, runs, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    # convergence curves (blocked) + weight trajectories (both orders)
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
    cmap = plt.get_cmap("tab10")
    for r in runs:
        D = znorm(r["desc"]); cls = r["cls"]; pos = r["pos"]; classes = sorted(set(cls.tolist()))
        ls = "-" if r["order"] == "blocked" else "--"
        # panel 0: distance-to-late-centroid vs k (mean over classes)
        curves = []
        for c in classes:
            m = cls == c; Dc = D[m][np.argsort(pos[m])]; n = len(Dc)
            if n < 6:
                continue
            late = Dc[int(0.66 * n):].mean(0)
            dist = np.linalg.norm(Dc - late, axis=1)
            curves.append(np.interp(np.linspace(0, 1, 50), np.linspace(0, 1, n), dist))
        if curves:
            axs[0].plot(np.linspace(0, 1, 50), np.mean(curves, 0), ls, label=r["order"])
        # panel 1: weight trajectory
        axs[1].plot(r["wtrace"], ls, label=r["order"])
    axs[0].set_title(f"{arm}: dist to late-block centroid vs block position")
    axs[0].set_xlabel("normalized block position k"); axs[0].set_ylabel("descriptor distance"); axs[0].legend()
    axs[1].set_title(f"{arm}: total synaptic weight over the run")
    axs[1].set_xlabel("sample #"); axs[1].set_ylabel("sum |w|"); axs[1].legend()
    # panel 2: PCA drift of blocked run colored by block position
    rb = next((r for r in runs if r["order"] == "blocked"), runs[0])
    D = znorm(rb["desc"]); Dc = D - D.mean(0)
    _, _, Vt = np.linalg.svd(Dc, full_matrices=False); emb = Dc @ Vt[:2].T
    sc = axs[2].scatter(emb[:, 0], emb[:, 1], c=rb["pos"], cmap="viridis", s=14)
    for c in sorted(set(rb["cls"].tolist())):
        mm = rb["cls"] == c
        axs[2].annotate(str(c), emb[mm].mean(0), fontsize=12, fontweight="bold")
    axs[2].set_title(f"{arm}: descriptor drift (color=block position)")
    plt.colorbar(sc, ax=axs[2], shrink=0.8)
    fig.tight_layout(); fig.savefig(os.path.join(out, f"{arm}_continuous.png"), dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="foveation_results/continuous_adapt")
    ap.add_argument("--out", default="foveation_results/continuous_adapt")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    npzs = sorted(glob.glob(os.path.join(a.dir, "*.npz")))
    # group by arm (tag = <arm>_<order>)
    arms = {}
    for p in npzs:
        base = os.path.basename(p)[:-4]
        for od in ("blocked", "interleaved"):
            if base.endswith("_" + od):
                arms.setdefault(base[:-len(od) - 1], {})[od] = p
    summary = {}
    for arm, runs_p in sorted(arms.items()):
        runs = [load_run(p) for p in runs_p.values()]
        figures(arm, runs, a.out)
        summary[arm] = {r["order"]: analyze_run(r) for r in map(load_run, runs_p.values())}
    json.dump(summary, open(os.path.join(a.out, "continuous_summary.json"), "w"), indent=2)
    # console verdict
    print("\n=== continuous slow-adaptation: cross-arm summary ===")
    for arm, od in summary.items():
        b = od.get("blocked", {}); iv = od.get("interleaved", {})
        print(f"\n[{arm}]")
        print(f"  BLOCKED     conv_slope={b.get('mean_conv_slope')} tighten={b.get('mean_tighten_ratio')} "
              f"Fisher {b.get('fisher_early')}->{b.get('fisher_late')} (gain {b.get('fisher_gain')}) "
              f"W {b.get('w0')}->{b.get('w_final')} (late_slope {b.get('w_late_slope')})")
        if iv:
            print(f"  INTERLEAVED conv_slope={iv.get('mean_conv_slope')} tighten={iv.get('mean_tighten_ratio')} "
                  f"Fisher {iv.get('fisher_early')}->{iv.get('fisher_late')} (gain {iv.get('fisher_gain')})")
    print(f"\nwrote {os.path.join(a.out, 'continuous_summary.json')}")


if __name__ == "__main__":
    main()
