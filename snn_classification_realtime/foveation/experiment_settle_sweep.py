"""EXP1b: settle-time sweep.

EXP1 showed reward~saliency is high at short settle windows (informativeness
still reflects input contrast) but collapses to ~0 by 15 ticks. Hypothesis:
the untrained perception network destroys the object/contrast signal as its
own dynamics take over, because it has no learned class attractors to settle
into. This sweep measures how reward~saliency, informativeness~saliency, and
settledness~saliency evolve with settle time on a FIXED image set.
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
from snn_classification_realtime.foveation.signals import FreeEnergyProbe
from snn_classification_realtime.foveation import saliency as sal


def position_curves(
    perception: PerceptionNetwork,
    fovea: Fovea,
    image: torch.Tensor,
    y: int,
    x: int,
    max_ticks: int,
    probe_history: int,
    checkpoints: list[int],
) -> dict[int, dict[str, float]]:
    """Run one settle window to max_ticks, snapshot probe at each checkpoint."""
    perception.reset()
    fovea.set_position(y, x)
    signals = perception.patch_to_signals(fovea.crop(image))
    probe = FreeEnergyProbe(history=probe_history)
    out: dict[int, dict[str, float]] = {}
    cps = set(checkpoints)
    for t in range(1, max_ticks + 1):
        probe.update(perception.step(signals))
        if t in cps:
            out[t] = {
                "reward": probe.reward,
                "informativeness": probe.informativeness,
                "settledness": probe.settledness,
            }
    return out


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
    positions = fovea.grid_positions(stride=args.grid_stride)
    checkpoints = sorted(int(c) for c in args.checkpoints.split(","))
    max_ticks = max(checkpoints)

    n = min(args.num_images, len(ds))
    stride = max(1, len(ds) // n)
    idxs = list(range(0, stride * n, stride))[:n]
    print(
        f"{perception.num_neurons} neurons | {len(positions)} positions | "
        f"{n} images | checkpoints {checkpoints}"
    )

    # corr[metric][tick] -> list over images
    corr: dict[str, dict[int, list[float]]] = {
        m: {t: [] for t in checkpoints}
        for m in ("reward", "informativeness", "settledness")
    }
    for img_idx in tqdm(idxs, desc="Images"):
        image, _ = ds[img_idx]
        saliency = sal.patch_saliency(image, positions, args.fovea_size)
        # metric_at[t][metric] -> array over positions
        by_tick: dict[int, dict[str, np.ndarray]] = {
            t: {m: np.empty(len(positions), np.float32)
                for m in ("reward", "informativeness", "settledness")}
            for t in checkpoints
        }
        for pi, (y, x) in enumerate(positions):
            snaps = position_curves(
                perception, fovea, image, y, x,
                max_ticks, args.probe_history, checkpoints,
            )
            for t in checkpoints:
                for m in ("reward", "informativeness", "settledness"):
                    by_tick[t][m][pi] = snaps[t][m]
        for t in checkpoints:
            for m in ("reward", "informativeness", "settledness"):
                p, _ = sal.safe_corr(by_tick[t][m], saliency)
                corr[m][t].append(p)

    summary = {
        "config": vars(args),
        "num_neurons": perception.num_neurons,
        "num_positions": len(positions),
        "checkpoints": checkpoints,
        "curves": {
            m: {
                str(t): {
                    "mean_pearson_vs_saliency": float(np.mean(corr[m][t])),
                    "std": float(np.std(corr[m][t])),
                    "frac_positive": float(np.mean(np.array(corr[m][t]) > 0)),
                }
                for t in checkpoints
            }
            for m in ("reward", "informativeness", "settledness")
        },
    }
    out_path = os.path.join(
        args.output_dir, f"exp1b_settle_sweep_{int(time.time())}.json"
    )
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== EXP1b: settle-time sweep (mean pearson vs saliency) ===")
    header = "tick  " + "".join(f"{t:>8}" for t in checkpoints)
    print(header)
    for m in ("reward", "informativeness", "settledness"):
        row = f"{m[:12]:<13}" + "".join(
            f"{summary['curves'][m][str(t)]['mean_pearson_vs_saliency']:+8.3f}"
            for t in checkpoints
        )
        print(row)
    print(f"Saved: {out_path}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="EXP1b settle-time sweep")
    p.add_argument("--dataset-name", default="cifar10_grayscale")
    p.add_argument("--fovea-size", type=int, default=16)
    p.add_argument("--grid-stride", type=int, default=4)
    p.add_argument("--checkpoints", default="2,4,6,10,15,25,40")
    p.add_argument("--probe-history", type=int, default=4)
    p.add_argument("--signal-gain", type=float, default=0.1)
    p.add_argument("--num-images", type=int, default=8)
    p.add_argument("--ablation", default="none")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="foveation_results")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
