"""Pool + plot the CIFAR10 ceiling (Experiment 1) and learned-feature ladder (Experiment 2).

Reads the per-config JSON records written by run_kmeans_ceiling.py / run_learned_ladder.py, so
every figure is recomputable offline without re-running anything. Reports bootstrap CIs over
seeds and prints "<2 seeds: no CI" rather than inventing an interval.

  --what ablate  one-factor-at-a-time deltas from canonical (which lever costs what)
  --what sweep   ceiling curve vs the published Coates & Ng reference (instrument check)
  --what ladder  arm bars + the pre-registered paired contrasts

CLI-only; run from neuron-model/.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

from cifar_ceiling.common import COATES_NG_2011_REFERENCE, bootstrap_ci


def load(d: str):
    recs = []
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        if os.path.basename(p) == "ladder_summary.json":
            continue
        with open(p, encoding="utf-8") as stream:
            recs.append(json.load(stream))
    return recs


def group(recs, key):
    g = defaultdict(list)
    for r in recs:
        g[r.get(key)].append(r["test_acc"])
    return g


def _fig():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def do_ablate(recs, out):
    g = group(recs, "name")
    if "canonical" not in g:
        print("no canonical record; run --stage calibrate/ablate first")
        return
    base, _ = bootstrap_ci(g["canonical"])
    rows = []
    for name, accs in g.items():
        m, (lo, hi) = bootstrap_ci(accs)
        d = np.asarray(accs) - base
        dm, (dlo, dhi) = bootstrap_ci(d)
        rows.append((name, m, lo, hi, dm, dlo, dhi, len(accs)))
    rows.sort(key=lambda r: r[4])
    print(f"\n=== ablation from canonical ({base:.2f}%) — how much each lever costs ===")
    for n, m, lo, hi, dm, dlo, dhi, k in rows:
        ci = f"[{lo:.2f},{hi:.2f}]" if np.isfinite(lo) else "[no CI]"
        print(f"  {n:20s} {m:6.2f}% {ci:18s} delta {dm:+6.2f}  (n={k})")
    plt = _fig()
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(rows) + 2))
    names = [r[0] for r in rows]
    deltas = [r[4] for r in rows]
    ax.barh(names, deltas, color=["#b4453c" if d < 0 else "#3c7fb4" for d in deltas])
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel(f"test accuracy delta from canonical ({base:.2f}%), points")
    ax.set_title("Which pipeline lever costs what (one-factor-at-a-time, K fixed)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "ablation.png"), dpi=130)
    print(f"wrote {os.path.join(out, 'ablation.png')}")


def do_sweep(recs, out):
    g = defaultdict(list)
    for r in recs:
        if r.get("name") == "canonical" or str(r.get("name", "")).startswith("sweep_K"):
            g[r["K"]].append(r["test_acc"])
    if not g:
        print("no sweep records")
        return
    Ks = sorted(g)
    means, los, his = [], [], []
    print("\n=== ceiling curve vs published Coates & Ng (2011) ===")
    for K in Ks:
        m, (lo, hi) = bootstrap_ci(g[K])
        means.append(m); los.append(lo); his.append(hi)
        ref = COATES_NG_2011_REFERENCE.get(K)
        rs = f"  published {ref:.1f}%  delta {m - ref:+.1f}" if ref else "  (no published point)"
        ci = f"[{lo:.2f},{hi:.2f}]" if np.isfinite(lo) else "[no CI]"
        print(f"  K={K:<5d} {m:6.2f}% {ci:18s}{rs}")
    plt = _fig()
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ok = np.isfinite(los)
    ax.plot(Ks, means, "o-", color="#3c7fb4", label="measured (ours)")
    if ok.all():
        ax.fill_between(Ks, los, his, color="#3c7fb4", alpha=0.18)
    rk = sorted(COATES_NG_2011_REFERENCE)
    ax.plot(rk, [COATES_NG_2011_REFERENCE[k] for k in rk], "s--", color="#888",
            label="Coates & Ng 2011 (published)")
    ax.axhline(62.0, color="#b4453c", ls=":", lw=1.4, label="PAULA conv substrate (62%)")
    ax.set_xscale("log")
    ax.set_xlabel("dictionary size K")
    ax.set_ylabel("CIFAR10 test top-1 (%)")
    ax.set_title("Fixed-feature ceiling: measured vs published")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "ceiling_curve.png"), dpi=130)
    print(f"wrote {os.path.join(out, 'ceiling_curve.png')}")


def paired_differences(recs, low_arm, high_arm):
    """Pair completed runs by their actual seed; omit unmatched seeds."""
    by_arm = {low_arm: {}, high_arm: {}}
    for record in recs:
        arm = record.get("arm")
        if arm not in by_arm:
            continue
        seed = record["seed"]
        if seed in by_arm[arm]:
            raise ValueError(f"duplicate {arm} record for seed {seed}; select one protocol directory")
        by_arm[arm][seed] = record["test_acc"]
    seeds = sorted(set(by_arm[low_arm]) & set(by_arm[high_arm]))
    return seeds, np.asarray([by_arm[high_arm][seed] - by_arm[low_arm][seed] for seed in seeds])


def do_ladder(recs, out):
    g = group(recs, "arm")
    order = [a for a in ["random", "randpatch", "kmeans", "krotov", "supervised", "e2e_cnn"] if a in g]
    print("\n=== learned-feature ladder (matched K, encoder, pooling, readout) ===")
    means, los, his = [], [], []
    for arm in order:
        m, (lo, hi) = bootstrap_ci(g[arm])
        means.append(m); los.append(lo); his.append(hi)
        ci = f"[{lo:.2f},{hi:.2f}]" if np.isfinite(lo) else "[<2 seeds: no CI]"
        note = "  (UNMATCHED reference)" if arm == "e2e_cnn" else ""
        print(f"  {arm:12s} {m:6.2f}% {ci}  (n={len(g[arm])}){note}")

    print("\n=== pre-registered contrasts (paired by seed) ===")
    for lo_a, hi_a, label in [("random", "krotov", "H1: competition load-bearing"),
                              ("randpatch", "kmeans", "H2: dictionary learning matters"),
                              ("kmeans", "supervised", "H3: labels are the gap")]:
        if lo_a in g and hi_a in g:
            seeds, d = paired_differences(recs, lo_a, hi_a)
            if not seeds:
                print(f"  {hi_a} - {lo_a}: no shared seeds")
                continue
            m, (l, h) = bootstrap_ci(d)
            sig = "*" if np.isfinite(l) and (l > 0 or h < 0) else " "
            ci = f"[{l:+.2f},{h:+.2f}]" if np.isfinite(l) else "[no CI]"
            print(f"  {sig} {hi_a} - {lo_a}: {m:+.2f} {ci}   ({label}; seeds={seeds})")

    plt = _fig()
    fig, ax = plt.subplots(figsize=(8, 4.6))
    cols = {"random": "#999", "randpatch": "#7fa8c4", "kmeans": "#3c7fb4", "krotov": "#4ca64c",
            "supervised": "#b4453c", "e2e_cnn": "#5a3c7f"}
    err = [[m - lo if np.isfinite(lo) else 0 for m, lo in zip(means, los)],
           [hi - m if np.isfinite(hi) else 0 for m, hi in zip(means, his)]]
    ax.bar(order, means, yerr=err, capsize=4, color=[cols[a] for a in order])
    ax.axhline(62.0, color="#b4453c", ls=":", lw=1.4, label="PAULA conv substrate (62%)")
    ax.axhline(10.0, color="k", ls="--", lw=0.8, label="chance")
    ax.set_ylabel("CIFAR10 test top-1 (%)")
    ax.set_title("What does learning the filters buy, at fixed topology?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "ladder.png"), dpi=130)
    print(f"wrote {os.path.join(out, 'ladder.png')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["ablate", "sweep", "ladder"], required=True)
    ap.add_argument("--dir", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = a.dir or ("cifar_ceiling/results/ladder" if a.what == "ladder" else "cifar_ceiling/results/kmeans")
    out = a.out or d
    os.makedirs(out, exist_ok=True)
    recs = load(d)
    if not recs:
        print(f"no records in {d}")
        return
    {"ablate": do_ablate, "sweep": do_sweep, "ladder": do_ladder}[a.what](recs, out)


if __name__ == "__main__":
    main()
