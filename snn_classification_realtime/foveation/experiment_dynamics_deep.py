"""EXP2-deep: is the settled dynamical regime input-distinct?

EXP2 (gain 3, 160 ticks, one averaged signature) said no. This tests why:

- regime: gain 1 (graded, edge-tracking per EXP4) vs gain 3 (clock-firing).
- time: run to `ticks` (1000+) and read separability as a function of time.
- aggregation: per-neuron ANOVA selectivity + PCA trajectory geometry + LOO-NN
  on population vectors, not a single averaged number.
- feature: S vs firing-rate vs t_ref, separately.
- color: grayscale vs RGB.

Population vectors keep every neuron (no cross-neuron averaging). Fovea is fixed
at center. Renders a PCA scatter of settled states colored by label.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from tqdm import tqdm
from scipy.stats import f_oneway

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from snn_classification_realtime.activity_dataset_builder.vision_datasets import (
    load_dataset_by_name,
)
from snn_classification_realtime.foveation.fovea import Fovea
from snn_classification_realtime.foveation.perception import (
    PerceptionNetwork,
    build_fovea_network_json,
)


def run_fixed(perception, fovea, image, y, x, ticks):
    perception.reset()
    fovea.set_position(y, x)
    sig = perception.patch_to_signals(fovea.crop(image))
    n = perception.num_neurons
    S = np.empty((ticks, n), np.float32)
    tr = np.empty((ticks, n), np.float32)
    O = np.empty((ticks, n), np.float32)
    for t in range(ticks):
        st = perception.step(sig)
        S[t], tr[t], O[t] = st.S, st.t_ref, st.O
    return {"S": S, "t_ref": tr, "O": O}


def window_vec(traj, feat, t_end, win):
    """Per-neuron mean of `feat` over ticks [t_end-win, t_end)."""
    a = traj[feat][max(0, t_end - win):t_end]
    if feat == "O":
        return a.mean(0)  # firing rate
    return a.mean(0)


def loo_nn(X, labels):
    Xc = X - X.mean(0, keepdims=True)
    nrm = np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-9
    Xn = Xc / nrm
    sim = Xn @ Xn.T
    np.fill_diagonal(sim, -np.inf)
    pred = labels[np.argmax(sim, axis=1)]
    return float(np.mean(pred == labels))


def neuron_anova_fraction(rate_by_img, labels, alpha=0.05):
    """Fraction of neurons whose firing rate differs across labels (ANOVA)."""
    uniq = np.unique(labels)
    n_neurons = rate_by_img.shape[1]
    sig = 0
    for j in range(n_neurons):
        groups = [rate_by_img[labels == u, j] for u in uniq]
        if any(len(g) < 2 for g in groups) or np.std(rate_by_img[:, j]) < 1e-9:
            continue
        try:
            _, p = f_oneway(*groups)
            if np.isfinite(p) and p < alpha:
                sig += 1
        except Exception:
            continue
    return sig / max(1, n_neurons)


def pca_2d(X):
    Xc = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T


def run(args):
    torch.manual_seed(0)
    np.random.seed(0)
    ds_cfg = load_dataset_by_name(args.dataset_name, train=True)
    ds_cfg.signal_gain = args.signal_gain
    ds = ds_cfg.dataset
    img0, _ = ds[0]
    channels, H, W = img0.shape

    net_path = os.path.join(
        args.output_dir, f"fovea_deep_{channels}x{args.fovea_size}.json"
    )
    os.makedirs(args.output_dir, exist_ok=True)
    build_fovea_network_json(
        net_path, channels=channels, size=args.fovea_size, seed=args.seed
    )
    perception = PerceptionNetwork(net_path, ds_cfg, ablation=args.ablation)
    fovea = Fovea(image_h=H, image_w=W, size=args.fovea_size)
    cy, cx = fovea.max_fy // 2, fovea.max_fx // 2

    by_label = {}
    for i in range(len(ds)):
        _, lbl = ds[i]
        lbl = int(lbl)
        if lbl < args.num_labels and len(by_label.get(lbl, [])) < args.per_label:
            by_label.setdefault(lbl, []).append(i)
        if all(len(by_label.get(l, [])) >= args.per_label for l in range(args.num_labels)):
            break

    print(f"{perception.num_neurons} neurons | {channels}ch | gain {args.signal_gain} "
          f"| {args.ticks} ticks | {args.num_labels}x{args.per_label} imgs")

    trajs, labels = [], []
    for lbl in tqdm(range(args.num_labels), desc="Labels"):
        for idx in by_label[lbl]:
            image, _ = ds[idx]
            trajs.append(run_fixed(perception, fovea, image, cy, cx, args.ticks))
            labels.append(lbl)
    labels = np.array(labels)

    # --- settling curve: variance of mean|S| in sliding windows ---
    win = args.window
    meanabsS = np.stack([np.abs(t["S"]).mean(1) for t in trajs]).mean(0)  # avg image
    n_win = args.ticks // win
    win_var = [float(np.var(meanabsS[i*win:(i+1)*win])) for i in range(n_win)]
    settling_ratio = (win_var[-1] + 1e-12) / (win_var[0] + 1e-12)

    # --- separability vs time, per feature ---
    checkpoints = [c for c in args.checkpoints if c <= args.ticks]
    sep = {f: {} for f in ("S", "O", "t_ref")}
    for f in sep:
        for tE in checkpoints:
            X = np.stack([window_vec(t, f, tE, win) for t in trajs])
            sep[f][tE] = loo_nn(X, labels)

    # --- per-neuron label selectivity (late window firing rate) ---
    rate_late = np.stack([window_vec(t, "O", args.ticks, win) for t in trajs])
    S_late = np.stack([window_vec(t, "S", args.ticks, win) for t in trajs])
    anova_frac = neuron_anova_fraction(rate_late, labels)
    anova_frac_S = neuron_anova_fraction(S_late, labels)

    # --- spike regularity + participation late ---
    def isi_cv_all(O):
        cvs = []
        for j in range(O.shape[1]):
            idx = np.flatnonzero(O[:, j] > 0)
            if len(idx) >= 3:
                isi = np.diff(idx).astype(float)
                if isi.mean() > 0:
                    cvs.append(isi.std() / isi.mean())
        return float(np.mean(cvs)) if cvs else float("nan")
    cv_late = float(np.mean([isi_cv_all(t["O"][args.ticks-win:]) for t in trajs]))
    part_late = float(np.mean([t["O"][args.ticks-win:].mean() for t in trajs]))

    # --- PCA scatter of settled S vectors ---
    proj = pca_2d(S_late)
    fig, ax = plt.subplots(figsize=(5, 4))
    sc = ax.scatter(proj[:, 0], proj[:, 1], c=labels, cmap="tab10", s=40)
    ax.set_title(f"settled S (PCA) — gain {args.signal_gain}, {channels}ch, "
                 f"{args.ticks}t\nLOO-NN(S)={sep['S'][checkpoints[-1]]:.2f}")
    plt.colorbar(sc, label="label")
    fig.tight_layout()
    png = os.path.join(args.output_dir,
                       f"exp2deep_pca_g{args.signal_gain}_{channels}ch_{int(time.time())}.png")
    fig.savefig(png, dpi=90)
    plt.close(fig)

    summary = {
        "config": vars(args),
        "num_neurons": perception.num_neurons,
        "settling": {
            "window_variance_curve": win_var,
            "late_over_early_ratio": settling_ratio,
            "spike_cv_late": cv_late,
            "participation_late": part_late,
        },
        "separability_over_time": {
            f: {str(t): sep[f][t] for t in checkpoints} for f in sep
        },
        "chance": 1.0 / args.num_labels,
        "per_neuron_label_selective_fraction": {
            "firing_rate": anova_frac,
            "S": anova_frac_S,
        },
        "pca_png": png,
    }
    out = os.path.join(args.output_dir,
                       f"exp2deep_g{args.signal_gain}_{channels}ch_{int(time.time())}.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    ch = 1.0 / args.num_labels
    print(f"\n=== EXP2-deep | gain {args.signal_gain} | {channels}ch | {args.ticks}t ===")
    print(f"participation(late) {part_late:.3f} | spike CV(late) {cv_late:.3f} "
          f"(higher CV = less clock-like)")
    print(f"settling var late/early: {settling_ratio:.3f}")
    print(f"LOO-NN label separability over time (chance {ch:.2f}):")
    for f in ("S", "O", "t_ref"):
        row = " ".join(f"t{t}:{sep[f][t]:.2f}" for t in checkpoints)
        print(f"  {f:<6} {row}")
    print(f"per-neuron label-selective: rate {anova_frac:.1%}, S {anova_frac_S:.1%}")
    print(f"PCA: {png}")
    print(f"Saved: {out}")
    return summary


def main():
    p = argparse.ArgumentParser(description="EXP2-deep dynamics investigation")
    p.add_argument("--dataset-name", default="cifar10_grayscale")
    p.add_argument("--fovea-size", type=int, default=16)
    p.add_argument("--ticks", type=int, default=1000)
    p.add_argument("--window", type=int, default=100)
    p.add_argument("--checkpoints", type=int, nargs="+",
                   default=[100, 200, 400, 700, 1000])
    p.add_argument("--num-labels", type=int, default=5)
    p.add_argument("--per-label", type=int, default=6)
    p.add_argument("--signal-gain", type=float, default=1.0)
    p.add_argument("--ablation", default="none")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="foveation_results")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
