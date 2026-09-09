"""EXP3: edge / orientation selectivity (the V1 parallel).

EXP1 showed the reward's weak object bias is a *contrast* effect. Does that mean
individual perception neurons behave like V1 simple cells — tuned to oriented
edges? We present synthetic oriented edge stimuli spanning 0..180 deg and
measure each neuron's response (firing rate) as a function of orientation, then
compute an orientation-selectivity index (OSI) per neuron.

An untrained conv net has random receptive fields, so some orientation bias is
expected; the question is how sharp the tuning is and how preferred orientations
are distributed. This is a parallel direction, not on the fovea critical path.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from snn_classification_realtime.activity_dataset_builder.vision_datasets import (
    load_dataset_by_name,
)
from snn_classification_realtime.foveation.fovea import Fovea
from snn_classification_realtime.foveation.perception import (
    PerceptionNetwork,
    build_fovea_network_json,
)


def oriented_edge(size: int, theta_deg: float, phase: float = 0.0) -> torch.Tensor:
    """A normalized [-1,1] grating patch at orientation theta (1,size,size)."""
    theta = np.deg2rad(theta_deg)
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    ys -= size / 2.0
    xs -= size / 2.0
    proj = xs * np.cos(theta) + ys * np.sin(theta)
    freq = 2.0 * np.pi / (size / 2.0)  # ~2 cycles across the patch
    grating = np.sign(np.sin(freq * proj + phase)).astype(np.float32)
    return torch.from_numpy(grating).unsqueeze(0)


def response(
    perception: PerceptionNetwork, patch: torch.Tensor, ticks: int, last: int
) -> np.ndarray:
    """Per-neuron mean firing rate over the last `last` ticks of a fixed input."""
    perception.reset()
    sig = perception.patch_to_signals(patch)
    n = perception.num_neurons
    rates = np.zeros(n, np.float32)
    for t in range(ticks):
        st = perception.step(sig)
        if t >= ticks - last:
            rates += st.O
    return rates / last


def osi(rates_by_theta: np.ndarray, thetas: np.ndarray) -> np.ndarray:
    """Circular orientation-selectivity index per neuron (0=flat, 1=sharp).

    OSI = |sum r_k exp(i 2 theta_k)| / sum r_k  (orientation period = 180 deg).
    """
    ang = np.deg2rad(thetas) * 2.0
    vec = rates_by_theta @ np.exp(1j * ang)
    denom = rates_by_theta.sum(axis=1) + 1e-9
    return np.abs(vec) / denom


def run(args: argparse.Namespace) -> dict:
    torch.manual_seed(0)
    np.random.seed(0)
    ds_cfg = load_dataset_by_name(args.dataset_name, train=True)
    ds_cfg.signal_gain = args.signal_gain
    channels = 1

    net_path = os.path.join(
        args.output_dir, f"fovea_percept_{channels}x{args.fovea_size}.json"
    )
    os.makedirs(args.output_dir, exist_ok=True)
    build_fovea_network_json(
        net_path, channels=channels, size=args.fovea_size, seed=args.seed
    )
    perception = PerceptionNetwork(net_path, ds_cfg, ablation=args.ablation)

    thetas = np.linspace(0, 180, args.num_orientations, endpoint=False)
    n = perception.num_neurons
    # Average over two phases to reduce phase sensitivity.
    rates = np.zeros((n, len(thetas)), np.float32)
    for pi, phase in enumerate((0.0, np.pi / 2)):
        for ti, th in enumerate(thetas):
            patch = oriented_edge(args.fovea_size, float(th), phase)
            rates[:, ti] += response(perception, patch, args.ticks, args.last)
    rates /= 2.0

    osi_vals = osi(rates, thetas)
    active = rates.sum(axis=1) > args.min_rate  # neurons that respond at all
    pref = thetas[np.argmax(rates, axis=1)]

    summary = {
        "config": vars(args),
        "num_neurons": int(n),
        "num_active_neurons": int(active.sum()),
        "mean_osi_active": float(np.mean(osi_vals[active])) if active.any() else 0.0,
        "frac_orientation_selective": float(
            np.mean(osi_vals[active] > args.osi_threshold)
        ) if active.any() else 0.0,
        "preferred_orientation_hist": {
            f"{int(t)}": int(np.sum(np.round(pref[active]) == t)) for t in thetas
        } if active.any() else {},
        "osi_percentiles_active": {
            "p50": float(np.percentile(osi_vals[active], 50)) if active.any() else 0.0,
            "p90": float(np.percentile(osi_vals[active], 90)) if active.any() else 0.0,
        },
    }
    out_path = os.path.join(args.output_dir, f"exp3_orientation_{int(time.time())}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== EXP3: edge / orientation selectivity ===")
    print(f"active neurons: {summary['num_active_neurons']}/{n}")
    print(f"mean OSI (active): {summary['mean_osi_active']:.3f} "
          f"(p50 {summary['osi_percentiles_active']['p50']:.3f}, "
          f"p90 {summary['osi_percentiles_active']['p90']:.3f})")
    print(f"fraction orientation-selective (OSI>{args.osi_threshold}): "
          f"{summary['frac_orientation_selective']:.2%}")
    print(f"Saved: {out_path}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="EXP3 orientation selectivity")
    p.add_argument("--dataset-name", default="cifar10_grayscale")
    p.add_argument("--fovea-size", type=int, default=16)
    p.add_argument("--num-orientations", type=int, default=12)
    p.add_argument("--ticks", type=int, default=60)
    p.add_argument("--last", type=int, default=30)
    p.add_argument("--signal-gain", type=float, default=3.0)
    p.add_argument("--min-rate", type=float, default=0.01)
    p.add_argument("--osi-threshold", type=float, default=0.3)
    p.add_argument("--ablation", default="none")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="foveation_results")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
