"""EXP4: retinotopic glow map.

Scan the fovea over an image; at each position measure perception-net activity
and paint it back onto image space at the fovea centre. If the net "sees" edges,
the glow should trace the object's contours. Renders image | edges | firing-rate
glow | informativeness glow as a PNG, and quantifies glow~edge correlation.

A smaller fovea localizes edges more sharply (its window straddles a contour for
fewer positions), so fovea size is a knob here.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import pearsonr

from snn_classification_realtime.activity_dataset_builder.vision_datasets import (
    load_dataset_by_name,
)
from snn_classification_realtime.foveation.fovea import Fovea
from snn_classification_realtime.foveation.perception import (
    PerceptionNetwork,
    build_fovea_network_json,
)
from snn_classification_realtime.foveation import saliency as sal


def scan_activity(
    perception: PerceptionNetwork,
    fovea: Fovea,
    image: torch.Tensor,
    stride: int,
    settle: int,
    last: int,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Return (rate_map, info_map) over the center grid + the positions."""
    positions = fovea.grid_positions(stride=stride)
    rate = np.zeros(len(positions), np.float32)
    info = np.zeros(len(positions), np.float32)
    for pi, (y, x) in enumerate(positions):
        perception.reset()
        fovea.set_position(y, x)
        sig = perception.patch_to_signals(fovea.crop(image))
        acc_rate = 0.0
        S_hist = []
        for t in range(settle):
            st = perception.step(sig)
            if t >= settle - last:
                acc_rate += float(st.O.mean())
                S_hist.append(st.S)
        rate[pi] = acc_rate / last
        info[pi] = float(np.mean([np.std(s) for s in S_hist]))
    return rate, info, positions


def to_image_map(
    values: np.ndarray,
    positions: list[tuple[int, int]],
    H: int,
    W: int,
    fov: int,
) -> np.ndarray:
    """Paint a per-position scalar onto an HxW map at each fovea center."""
    m = np.full((H, W), np.nan, np.float32)
    for v, (y, x) in zip(values, positions):
        cy, cx = y + fov // 2, x + fov // 2
        m[cy, cx] = v
    # Fill NaNs (unscanned pixels) by nearest scanned value via simple dilation.
    from scipy.ndimage import distance_transform_edt

    mask = np.isnan(m)
    if mask.any():
        idx = distance_transform_edt(mask, return_distances=False, return_indices=True)
        m = m[tuple(idx)]
    return m


def render(
    image: torch.Tensor,
    edge: np.ndarray,
    rate_map: np.ndarray,
    info_map: np.ndarray,
    label: int,
    corr_rate: float,
    corr_info: float,
    path: str,
) -> None:
    img = image.detach().cpu().numpy()
    if img.ndim == 3:
        img = img.mean(0)
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.4))
    ax[0].imshow(img, cmap="gray")
    ax[0].set_title(f"input (label {label})")
    ax[1].imshow(edge, cmap="magma")
    ax[1].set_title("edge energy")
    ax[2].imshow(info_map, cmap="inferno")
    ax[2].set_title(f"glow: informativeness\nr(edge)={corr_info:+.2f}")
    ax[3].imshow(rate_map, cmap="inferno")
    ax[3].set_title(f"glow: firing rate\nr(edge)={corr_rate:+.2f}")
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict:
    torch.manual_seed(0)
    np.random.seed(0)
    ds_cfg = load_dataset_by_name(args.dataset_name, train=True)
    ds_cfg.signal_gain = args.signal_gain
    ds = ds_cfg.dataset
    img0, _ = ds[0]
    channels, H, W = img0.shape

    # Single conv layer keeps a small fovea from collapsing to 1x1.
    layers = [{"type": "conv", "kernel_size": 3, "stride": 1, "filters": args.filters}]
    net_path = os.path.join(
        args.output_dir, f"fovea_glow_{channels}x{args.fovea_size}.json"
    )
    os.makedirs(args.output_dir, exist_ok=True)
    build_fovea_network_json(
        net_path, channels=channels, size=args.fovea_size, layers=layers, seed=args.seed
    )
    perception = PerceptionNetwork(net_path, ds_cfg, ablation=args.ablation)
    fovea = Fovea(image_h=H, image_w=W, size=args.fovea_size)

    glow_dir = os.path.join(args.output_dir, "glow")
    os.makedirs(glow_dir, exist_ok=True)

    n = min(args.num_images, len(ds))
    stride_idx = max(1, len(ds) // n)
    idxs = list(range(0, stride_idx * n, stride_idx))[:n]
    print(
        f"{perception.num_neurons} neurons | fovea {args.fovea_size} | "
        f"scan stride {args.scan_stride} | gain {args.signal_gain} | {n} images"
    )

    corr_rates, corr_infos = [], []
    for img_idx in tqdm(idxs, desc="Images"):
        image, label = ds[img_idx]
        rate, info, positions = scan_activity(
            perception, fovea, image, args.scan_stride, args.settle, args.last
        )
        window_edge = sal.patch_saliency(image, positions, args.fovea_size)
        cr = sal.safe_corr(rate, window_edge)[0]
        ci = sal.safe_corr(info, window_edge)[0]
        corr_rates.append(cr)
        corr_infos.append(ci)

        edge = sal.edge_energy_map(image)
        rate_map = to_image_map(rate, positions, H, W, args.fovea_size)
        info_map = to_image_map(info, positions, H, W, args.fovea_size)
        render(
            image, edge, rate_map, info_map, int(label), cr, ci,
            os.path.join(glow_dir, f"glow_{img_idx}_label{int(label)}.png"),
        )

    summary = {
        "config": vars(args),
        "num_neurons": perception.num_neurons,
        "glow_rate_vs_edge_pearson_mean": float(np.mean(corr_rates)),
        "glow_info_vs_edge_pearson_mean": float(np.mean(corr_infos)),
        "glow_rate_vs_edge_frac_positive": float(np.mean(np.array(corr_rates) > 0)),
        "glow_info_vs_edge_frac_positive": float(np.mean(np.array(corr_infos) > 0)),
        "per_image_rate": corr_rates,
        "per_image_info": corr_infos,
    }
    import json

    out_path = os.path.join(args.output_dir, f"exp4_glow_{int(time.time())}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== EXP4: glow ~ edge correlation ===")
    print(f"firing-rate glow ~ edge: {summary['glow_rate_vs_edge_pearson_mean']:+.3f} "
          f"({summary['glow_rate_vs_edge_frac_positive']:.0%} images positive)")
    print(f"informativeness glow ~ edge: {summary['glow_info_vs_edge_pearson_mean']:+.3f} "
          f"({summary['glow_info_vs_edge_frac_positive']:.0%} images positive)")
    print(f"PNGs: {glow_dir}/")
    print(f"Saved: {out_path}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="EXP4 retinotopic glow map")
    p.add_argument("--dataset-name", default="cifar10_grayscale")
    p.add_argument("--fovea-size", type=int, default=10)
    p.add_argument("--filters", type=int, default=4)
    p.add_argument("--scan-stride", type=int, default=1)
    p.add_argument("--settle", type=int, default=8)
    p.add_argument("--last", type=int, default=4)
    p.add_argument("--signal-gain", type=float, default=1.0)
    p.add_argument("--num-images", type=int, default=4)
    p.add_argument("--ablation", default="none")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="foveation_results")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
