"""EXP2: does the dynamical regime settle, and does it reflect the input?

The core world-modeling premise (ALERM/PAULA): under a fixed input the network
relaxes into an input-specific dynamical regime (attractor / limit cycle), and
that regime is distinct for different inputs. We test three things with a fixed
fovea (no motion):

1. Settling: does the S trajectory become more regular over time (late-window
   variance < early-window variance)? Does t_ref converge? Do spike trains
   become more regular (ISI CV drops)?
2. Label-distinctness: is the settled-state signature separable by class
   (leave-one-out nearest-neighbour accuracy vs chance)?
3. Region-distinctness: within one image, is the settled regime at the centre
   (object) different from the corners (background)?
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from snn_classification_realtime.activity_dataset_builder.vision_datasets import (
    load_dataset_by_name,
)
from snn_classification_realtime.foveation.fovea import Fovea
from snn_classification_realtime.foveation.perception import (
    PerceptionNetwork,
    build_fovea_network_json,
)


def run_fixed(
    perception: PerceptionNetwork,
    fovea: Fovea,
    image: torch.Tensor,
    y: int,
    x: int,
    ticks: int,
) -> dict[str, np.ndarray]:
    """Fixed input for `ticks`; return [T,N] trajectories of S, t_ref, O."""
    perception.reset()
    fovea.set_position(y, x)
    sig = perception.patch_to_signals(fovea.crop(image))
    n = perception.num_neurons
    S = np.empty((ticks, n), np.float32)
    tr = np.empty((ticks, n), np.float32)
    O = np.empty((ticks, n), np.float32)
    for t in range(ticks):
        st = perception.step(sig)
        S[t] = st.S
        tr[t] = st.t_ref
        O[t] = st.O
    return {"S": S, "t_ref": tr, "O": O}


def isi_cv(spikes_1d: np.ndarray) -> float:
    """Coefficient of variation of inter-spike intervals (regularity)."""
    idx = np.flatnonzero(spikes_1d > 0)
    if len(idx) < 3:
        return np.nan
    isi = np.diff(idx).astype(np.float64)
    if isi.mean() < 1e-9:
        return np.nan
    return float(isi.std() / isi.mean())


def settling_metrics(traj: dict[str, np.ndarray]) -> dict[str, float]:
    S, tr, O = traj["S"], traj["t_ref"], traj["O"]
    T = S.shape[0]
    half = T // 2
    # Population-activity regularity: variance of mean|S| over time, early/late.
    mean_absS = np.mean(np.abs(S), axis=1)
    early_var = float(np.var(mean_absS[:half]))
    late_var = float(np.var(mean_absS[half:]))
    # t_ref convergence: late-window drift and spread.
    mean_tref = np.mean(tr, axis=1)
    tref_late_std = float(np.std(mean_tref[half:]))
    tref_drift = float(abs(mean_tref[half:].mean() - mean_tref[:half].mean()))
    # Spike-train regularity: mean ISI CV, early vs late.
    cv_early = np.nanmean([isi_cv(O[:half, i]) for i in range(O.shape[1])])
    cv_late = np.nanmean([isi_cv(O[half:, i]) for i in range(O.shape[1])])
    return {
        "meanS_var_early": early_var,
        "meanS_var_late": late_var,
        "meanS_var_ratio_late_over_early": late_var / (early_var + 1e-9),
        "tref_late_std": tref_late_std,
        "tref_drift_early_to_late": tref_drift,
        "spike_cv_early": float(cv_early) if np.isfinite(cv_early) else float("nan"),
        "spike_cv_late": float(cv_late) if np.isfinite(cv_late) else float("nan"),
        "mean_participation": float(O.mean()),
    }


def signature(traj: dict[str, np.ndarray], last: int) -> np.ndarray:
    """Settled-state signature = late-window mean of [S | t_ref | rate]."""
    S, tr, O = traj["S"][-last:], traj["t_ref"][-last:], traj["O"][-last:]
    return np.concatenate([S.mean(0), tr.mean(0), O.mean(0)]).astype(np.float32)


def loo_nn_accuracy(sigs: np.ndarray, labels: np.ndarray) -> float:
    """Leave-one-out nearest-neighbour label accuracy on signatures (cosine)."""
    X = sigs - sigs.mean(0, keepdims=True)
    norm = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    Xn = X / norm
    sim = Xn @ Xn.T
    np.fill_diagonal(sim, -np.inf)
    pred = labels[np.argmax(sim, axis=1)]
    return float(np.mean(pred == labels))


def run(args: argparse.Namespace) -> dict:
    torch.manual_seed(0)
    np.random.seed(0)
    ds_cfg = load_dataset_by_name(args.dataset_name, train=True)
    ds_cfg.signal_gain = args.signal_gain
    ds = ds_cfg.dataset
    img0, _ = ds[0]
    channels, H, W = img0.shape

    net_path = os.path.join(
        args.output_dir, f"fovea_percept_{channels}x{args.fovea_size}.json"
    )
    os.makedirs(args.output_dir, exist_ok=True)
    build_fovea_network_json(
        net_path, channels=channels, size=args.fovea_size, seed=args.seed
    )
    perception = PerceptionNetwork(net_path, ds_cfg, ablation=args.ablation)
    fovea = Fovea(image_h=H, image_w=W, size=args.fovea_size)
    cy, cx = fovea.max_fy // 2, fovea.max_fx // 2

    # Collect a small labelled set: per_label images x num_labels.
    by_label: dict[int, list[int]] = {}
    for i in range(len(ds)):
        _, lbl = ds[i]
        lbl = int(lbl)
        if lbl < args.num_labels and len(by_label.get(lbl, [])) < args.per_label:
            by_label.setdefault(lbl, []).append(i)
        if all(len(by_label.get(l, [])) >= args.per_label for l in range(args.num_labels)):
            break

    print(
        f"{perception.num_neurons} neurons | {args.ticks} ticks fixed | "
        f"{args.num_labels} labels x {args.per_label} imgs | gain {args.signal_gain}"
    )

    # --- (1) settling + (2) label-distinctness (centre fovea) ---
    settle_rows: list[dict[str, float]] = []
    sigs: list[np.ndarray] = []
    labels: list[int] = []
    region_rows: list[dict[str, float]] = []
    for lbl in tqdm(range(args.num_labels), desc="Labels"):
        for img_idx in by_label[lbl]:
            image, _ = ds[img_idx]
            traj = run_fixed(perception, fovea, image, cy, cx, args.ticks)
            settle_rows.append(settling_metrics(traj))
            sigs.append(signature(traj, args.signature_last))
            labels.append(lbl)

    sigs_arr = np.stack(sigs)
    labels_arr = np.array(labels)
    loo = loo_nn_accuracy(sigs_arr, labels_arr)

    # --- (3) region-distinctness within one image (centre vs 4 corners) ---
    corners = [(0, 0), (0, fovea.max_fx), (fovea.max_fy, 0),
               (fovea.max_fy, fovea.max_fx)]
    region_sep = []
    for lbl in range(args.num_labels):
        image, _ = ds[by_label[lbl][0]]
        c_sig = signature(
            run_fixed(perception, fovea, image, cy, cx, args.ticks),
            args.signature_last,
        )
        corner_sigs = [
            signature(run_fixed(perception, fovea, image, y, x, args.ticks),
                      args.signature_last)
            for (y, x) in corners
        ]
        # cosine distance centre vs mean corner
        cm = np.mean(corner_sigs, axis=0)
        cos = float(np.dot(c_sig, cm) /
                    (np.linalg.norm(c_sig) * np.linalg.norm(cm) + 1e-9))
        region_sep.append(1.0 - cos)

    def agg(key: str) -> float:
        v = [r[key] for r in settle_rows if np.isfinite(r[key])]
        return float(np.mean(v)) if v else float("nan")

    summary = {
        "config": vars(args),
        "num_neurons": perception.num_neurons,
        "settling": {
            "meanS_var_ratio_late_over_early": agg("meanS_var_ratio_late_over_early"),
            "tref_late_std": agg("tref_late_std"),
            "tref_drift_early_to_late": agg("tref_drift_early_to_late"),
            "spike_cv_early": agg("spike_cv_early"),
            "spike_cv_late": agg("spike_cv_late"),
            "mean_participation": agg("mean_participation"),
        },
        "label_distinctness": {
            "loo_nn_accuracy": loo,
            "chance": 1.0 / args.num_labels,
            "n_samples": len(labels),
        },
        "region_distinctness": {
            "mean_center_vs_corner_cosine_distance": float(np.mean(region_sep)),
            "per_label": region_sep,
        },
    }
    out_path = os.path.join(args.output_dir, f"exp2_dynamics_{int(time.time())}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    s = summary["settling"]
    print("\n=== EXP2: dynamical regime under fixed input ===")
    print("(1) Settling (fixed input):")
    print(f"  mean|S| variance late/early: {s['meanS_var_ratio_late_over_early']:.3f} "
          "(<1 = settling)")
    print(f"  spike ISI CV early->late:    {s['spike_cv_early']:.3f} -> {s['spike_cv_late']:.3f} "
          "(lower = more regular)")
    print(f"  t_ref late-window std:       {s['tref_late_std']:.4f} "
          f"(drift {s['tref_drift_early_to_late']:.4f})")
    print(f"  mean participation:          {s['mean_participation']:.3f}")
    print("(2) Label-distinctness of settled signature:")
    print(f"  leave-one-out NN accuracy:   {loo:.3f}  (chance {1.0/args.num_labels:.3f})")
    print("(3) Region-distinctness within an image:")
    print(f"  centre vs corner cos-dist:   {np.mean(region_sep):.4f}")
    print(f"Saved: {out_path}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="EXP2 dynamics / world-modeling")
    p.add_argument("--dataset-name", default="cifar10_grayscale")
    p.add_argument("--fovea-size", type=int, default=16)
    p.add_argument("--ticks", type=int, default=160)
    p.add_argument("--signature-last", type=int, default=40)
    p.add_argument("--num-labels", type=int, default=5)
    p.add_argument("--per-label", type=int, default=4)
    p.add_argument("--signal-gain", type=float, default=3.0)
    p.add_argument("--ablation", default="none")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="foveation_results")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
