"""EXP1c: reward-saliency after local adaptation.

Untrained baseline (EXP1b, spiking regime): reward~saliency ~ +0.12, carried by
informativeness (contrast); settledness~saliency is noise. Hypothesis: once the
net's LOCAL plasticity has formed class/patch-specific attractors, SETTLEDNESS
should become object-selective (settling is deeper on structured input it has
learned to model). Protocol: adapt (learning on) -> freeze eta -> probe scan.
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


def adapt(
    perception: PerceptionNetwork,
    fovea: Fovea,
    images: list[torch.Tensor],
    ticks_per_exposure: int,
    epochs: int,
) -> None:
    """Stream patches with learning ON so local plasticity shapes attractors."""
    positions = fovea.grid_positions(stride=fovea.size // 2)
    for _ep in range(epochs):
        for image in images:
            for (y, x) in positions:
                perception.reset()
                fovea.set_position(y, x)
                sig = perception.patch_to_signals(fovea.crop(image))
                for _ in range(ticks_per_exposure):
                    perception.step(sig)


def probe_scan(
    perception: PerceptionNetwork,
    fovea: Fovea,
    images: list[tuple[torch.Tensor, int]],
    positions: list[tuple[int, int]],
    settle: int,
    probe_history: int,
    fovea_size: int,
) -> dict[str, dict[str, float]]:
    corr = {m: [] for m in ("reward", "informativeness", "settledness")}
    for image, _ in images:
        saliency = sal.patch_saliency(image, positions, fovea_size)
        vals = {m: np.empty(len(positions), np.float32) for m in corr}
        for pi, (y, x) in enumerate(positions):
            perception.reset()
            fovea.set_position(y, x)
            sig = perception.patch_to_signals(fovea.crop(image))
            pr = FreeEnergyProbe(history=probe_history)
            for _ in range(settle):
                pr.update(perception.step(sig))
            vals["reward"][pi] = pr.reward
            vals["informativeness"][pi] = pr.informativeness
            vals["settledness"][pi] = pr.settledness
        for m in corr:
            corr[m].append(sal.safe_corr(vals[m], saliency)[0])
    return {
        m: {
            "mean_pearson_vs_saliency": float(np.mean(corr[m])),
            "std": float(np.std(corr[m])),
        }
        for m in corr
    }


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

    n = min(args.num_images, len(ds))
    stride = max(1, len(ds) // n)
    probe_images = [ds[i] for i in range(0, stride * n, stride)][:n]
    adapt_images = [img for img, _ in probe_images]

    print(
        f"{perception.num_neurons} neurons | {len(positions)} positions | "
        f"{n} images | adapt {args.epochs}ep x {args.adapt_ticks}t"
    )

    eff0 = perception.mean_efficacy()
    before = probe_scan(
        perception, fovea, probe_images, positions,
        args.settle_ticks, args.probe_history, args.fovea_size,
    )

    perception.set_learning(True)
    adapt(perception, fovea, adapt_images, args.adapt_ticks, args.epochs)
    eff1 = perception.mean_efficacy()
    perception.set_learning(False)  # freeze for a clean probe

    after = probe_scan(
        perception, fovea, probe_images, positions,
        args.settle_ticks, args.probe_history, args.fovea_size,
    )

    summary = {
        "config": vars(args),
        "mean_efficacy_before": eff0,
        "mean_efficacy_after": eff1,
        "efficacy_change_pct": 100.0 * (eff1 - eff0) / (abs(eff0) + 1e-9),
        "before_adaptation": before,
        "after_adaptation": after,
    }
    out_path = os.path.join(args.output_dir, f"exp1c_adapt_{int(time.time())}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== EXP1c: reward~saliency, untrained vs adapted ===")
    print(f"mean efficacy: {eff0:.4f} -> {eff1:.4f} "
          f"({summary['efficacy_change_pct']:+.1f}%)")
    print(f"{'metric':<16}{'untrained':>12}{'adapted':>12}{'delta':>10}")
    for m in ("reward", "informativeness", "settledness"):
        b = before[m]["mean_pearson_vs_saliency"]
        a = after[m]["mean_pearson_vs_saliency"]
        print(f"{m:<16}{b:>+12.3f}{a:>+12.3f}{a-b:>+10.3f}")
    print(f"Saved: {out_path}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="EXP1c adapted-net reward-saliency")
    p.add_argument("--dataset-name", default="cifar10_grayscale")
    p.add_argument("--fovea-size", type=int, default=16)
    p.add_argument("--grid-stride", type=int, default=4)
    p.add_argument("--settle-ticks", type=int, default=15)
    p.add_argument("--probe-history", type=int, default=4)
    p.add_argument("--signal-gain", type=float, default=3.0)
    p.add_argument("--num-images", type=int, default=8)
    p.add_argument("--adapt-ticks", type=int, default=30)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--ablation", default="none")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="foveation_results")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
