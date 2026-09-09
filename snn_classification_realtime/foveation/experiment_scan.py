"""EXP1: reward-saliency correlation map (the crux test).

For each image, slide the fovea over a grid of positions; at each position run
a fresh settle window and record the label-free structured-stability reward.
Correlate the reward map against image saliency and a center prior. Records
only compact per-image metrics (no [T,N] activity), so it is storage-cheap.

Question: does structured stability peak on the object (reward ~ saliency), or
on the flattest patch (reward anti-correlated with saliency = dark room)?
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


def evaluate_position(
    perception: PerceptionNetwork,
    fovea: Fovea,
    image: torch.Tensor,
    y: int,
    x: int,
    settle_ticks: int,
    probe_history: int,
) -> dict[str, float]:
    """Reset the network, hold the fovea at (y,x), run the settle window."""
    perception.reset()
    fovea.set_position(y, x)
    patch = fovea.crop(image)
    signals = perception.patch_to_signals(patch)
    probe = FreeEnergyProbe(history=probe_history)
    for _ in range(settle_ticks):
        state = perception.step(signals)
        probe.update(state)
    return {
        "reward": probe.reward,
        "settledness": probe.settledness,
        "informativeness": probe.informativeness,
        "participation": probe.participation,
    }


def run(args: argparse.Namespace) -> dict:
    torch.manual_seed(0)
    np.random.seed(0)

    ds_cfg = load_dataset_by_name(args.dataset_name, train=True)
    ds = ds_cfg.dataset
    # Calibrated-ish gain so the retina is neither silent nor saturated.
    ds_cfg.signal_gain = args.signal_gain

    img0, _ = ds[0]
    channels, H, W = img0.shape

    net_path = args.network_path
    if net_path is None:
        net_path = os.path.join(
            args.output_dir, f"fovea_percept_{channels}x{args.fovea_size}.json"
        )
        build_fovea_network_json(
            net_path, channels=channels, size=args.fovea_size, seed=args.seed
        )

    perception = PerceptionNetwork(net_path, ds_cfg, ablation=args.ablation)
    fovea = Fovea(image_h=H, image_w=W, size=args.fovea_size)
    positions = fovea.grid_positions(stride=args.grid_stride)

    print(
        f"Perception net: {perception.num_neurons} neurons | "
        f"image {channels}x{H}x{W} | fovea {args.fovea_size} | "
        f"{len(positions)} grid positions | settle {args.settle_ticks} ticks"
    )

    per_image: list[dict] = []
    n = min(args.num_images, len(ds))
    # Spread the sample across classes for a representative mix.
    stride = max(1, len(ds) // n)
    idxs = list(range(0, stride * n, stride))[:n]

    for img_idx in tqdm(idxs, desc="Images"):
        image, label = ds[img_idx]
        reward = np.empty(len(positions), dtype=np.float32)
        settled = np.empty(len(positions), dtype=np.float32)
        info = np.empty(len(positions), dtype=np.float32)
        for pi, (y, x) in enumerate(positions):
            m = evaluate_position(
                perception, fovea, image, y, x,
                args.settle_ticks, args.probe_history,
            )
            reward[pi] = m["reward"]
            settled[pi] = m["settledness"]
            info[pi] = m["informativeness"]

        saliency = sal.patch_saliency(image, positions, args.fovea_size)
        center = sal.center_prior(positions, H, W, args.fovea_size)

        r_sal_p, r_sal_s = sal.safe_corr(reward, saliency)
        r_cen_p, _ = sal.safe_corr(reward, center)
        sal_cen_p, _ = sal.safe_corr(saliency, center)
        r_sal_partial = sal.partial_corr_saliency_given_center(
            reward, saliency, center
        )
        info_sal_p, _ = sal.safe_corr(info, saliency)
        set_sal_p, _ = sal.safe_corr(settled, saliency)

        # Where does the reward peak vs where saliency peaks?
        argmax_reward = positions[int(np.argmax(reward))]
        argmax_sal = positions[int(np.argmax(saliency))]
        peak_dist = float(
            np.hypot(argmax_reward[0] - argmax_sal[0], argmax_reward[1] - argmax_sal[1])
        )

        per_image.append({
            "image_idx": int(img_idx),
            "label": int(label),
            "reward_saliency_pearson": r_sal_p,
            "reward_saliency_spearman": r_sal_s,
            "reward_center_pearson": r_cen_p,
            "saliency_center_pearson": sal_cen_p,
            "reward_saliency_partial_given_center": r_sal_partial,
            "informativeness_saliency_pearson": info_sal_p,
            "settledness_saliency_pearson": set_sal_p,
            "reward_peak_to_saliency_peak_dist": peak_dist,
        })

    def agg(key: str) -> dict[str, float]:
        vals = np.array([d[key] for d in per_image], dtype=np.float64)
        return {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "frac_positive": float(np.mean(vals > 0)),
        }

    summary = {
        "config": {
            "dataset_name": args.dataset_name,
            "fovea_size": args.fovea_size,
            "grid_stride": args.grid_stride,
            "settle_ticks": args.settle_ticks,
            "probe_history": args.probe_history,
            "signal_gain": args.signal_gain,
            "num_images": n,
            "network_path": net_path,
            "num_neurons": perception.num_neurons,
            "num_positions": len(positions),
            "ablation": args.ablation,
        },
        "aggregate": {
            "reward_saliency_pearson": agg("reward_saliency_pearson"),
            "reward_saliency_spearman": agg("reward_saliency_spearman"),
            "reward_center_pearson": agg("reward_center_pearson"),
            "reward_saliency_partial_given_center": agg(
                "reward_saliency_partial_given_center"
            ),
            "informativeness_saliency_pearson": agg(
                "informativeness_saliency_pearson"
            ),
            "settledness_saliency_pearson": agg("settledness_saliency_pearson"),
            "reward_peak_to_saliency_peak_dist": agg(
                "reward_peak_to_saliency_peak_dist"
            ),
        },
        "per_image": per_image,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(
        args.output_dir, f"exp1_scan_{int(time.time())}.json"
    )
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    a = summary["aggregate"]
    print("\n=== EXP1: reward-saliency correlation ===")
    print(
        f"reward~saliency  pearson: {a['reward_saliency_pearson']['mean']:+.3f} "
        f"(±{a['reward_saliency_pearson']['std']:.3f}, "
        f"{a['reward_saliency_pearson']['frac_positive']:.0%} images positive)"
    )
    print(
        f"reward~saliency  spearman:{a['reward_saliency_spearman']['mean']:+.3f}"
    )
    print(
        f"reward~center    pearson: {a['reward_center_pearson']['mean']:+.3f} "
        "(is it just a center bias?)"
    )
    print(
        f"reward~saliency | center: {a['reward_saliency_partial_given_center']['mean']:+.3f} "
        "(saliency signal beyond center)"
    )
    print(
        f"  component: informativeness~saliency "
        f"{a['informativeness_saliency_pearson']['mean']:+.3f} | "
        f"settledness~saliency {a['settledness_saliency_pearson']['mean']:+.3f}"
    )
    print(f"Saved: {out_path}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="EXP1 reward-saliency scan")
    p.add_argument("--dataset-name", default="cifar10_grayscale")
    p.add_argument("--network-path", default=None)
    p.add_argument("--fovea-size", type=int, default=16)
    p.add_argument("--grid-stride", type=int, default=3)
    p.add_argument("--settle-ticks", type=int, default=12)
    p.add_argument("--probe-history", type=int, default=6)
    p.add_argument("--signal-gain", type=float, default=0.1)
    p.add_argument("--num-images", type=int, default=8)
    p.add_argument("--ablation", default="none")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="foveation_results")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
