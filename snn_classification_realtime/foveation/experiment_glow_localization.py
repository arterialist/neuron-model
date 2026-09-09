"""EXP10: does the informativeness glow localize the OBJECT, or just texture?

Caveat we must be honest about: correlating glow with EDGE energy cannot answer
this, because edges are high on textured backgrounds too. So glow~edge = +0.9
only proves "glow is an edge/contrast detector", not "glow finds the object".

CIFAR has no masks, so we use three mask-free discriminators, split by class into
clean-background (plane/car/ship/truck) vs cluttered-background (bird/cat/deer/
dog/frog/horse):

1. glow~edge per class     — expect uniformly high if glow == edge detector.
2. glow centre-of-mass distance from image centre (CIFAR objects are centre-
   biased). If glow sits centre for cluttered classes too -> object-ish; if it
   wanders to corners -> following background texture.
3. concentration (Gini) of glow vs the edge map BLURRED by the fovea footprint
   (fair comparison). If glow is MORE concentrated than blurred edges, it selects
   a subset of edges (the object's); if equal, it is just a blurred edge map.

Interpretation is collaborative — the numbers frame the question, they don't
settle "is edge-detection enough for the drive".
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from tqdm import tqdm
from scipy.ndimage import uniform_filter

from snn_classification_realtime.activity_dataset_builder.vision_datasets import (
    load_dataset_by_name,
)
from snn_classification_realtime.foveation.fovea import Fovea
from snn_classification_realtime.foveation.perception import (
    PerceptionNetwork,
    build_fovea_network_json,
)
from snn_classification_realtime.foveation import saliency as sal

CIFAR_NAMES = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
CLEAN = {0, 1, 8, 9}
CLUTTERED = {2, 3, 4, 5, 6, 7}


def gini(x: np.ndarray) -> float:
    x = np.sort(x.ravel().astype(np.float64))
    x = x - x.min() + 1e-9
    n = len(x)
    idx = np.arange(1, n + 1)
    return float((np.sum((2 * idx - n - 1) * x)) / (n * np.sum(x)))


def com_dist(m: np.ndarray, H: int, W: int) -> float:
    """Distance of the value-weighted centre of mass from image centre, /half-diag."""
    m = np.clip(m, 0, None)
    if m.sum() < 1e-9:
        return 0.0
    ys, xs = np.mgrid[0:H, 0:W]
    cy = (m * ys).sum() / m.sum()
    cx = (m * xs).sum() / m.sum()
    return float(np.hypot(cy - H / 2, cx - W / 2) / np.hypot(H / 2, W / 2))


def info_map(perception, fovea, image, positions, settle, last, H, W, fov):
    acc = np.zeros((H, W), np.float32)
    cnt = np.zeros((H, W), np.float32)
    for (y, x) in positions:
        perception.reset()
        fovea.set_position(y, x)
        sig = perception.patch_to_signals(fovea.crop(image))
        S_hist = []
        for t in range(settle):
            st = perception.step(sig)
            if t >= settle - last:
                S_hist.append(st.S)
        v = float(np.mean([np.std(s) for s in S_hist]))
        acc[y:y + fov, x:x + fov] += v
        cnt[y:y + fov, x:x + fov] += 1.0
    return np.divide(acc, cnt, out=np.zeros_like(acc), where=cnt > 0)


def run(args):
    torch.manual_seed(0)
    np.random.seed(0)
    ds_cfg = load_dataset_by_name(args.dataset_name, train=True)
    ds_cfg.signal_gain = args.signal_gain
    ds = ds_cfg.dataset
    img0, _ = ds[0]
    channels, H, W = img0.shape
    fov = args.fovea_size

    layers = [{"type": "conv", "kernel_size": 3, "stride": 1, "filters": args.filters}]
    net_path = os.path.join(args.output_dir, f"fovea_glow_{channels}x{fov}.json")
    os.makedirs(args.output_dir, exist_ok=True)
    build_fovea_network_json(net_path, channels=channels, size=fov, layers=layers, seed=args.seed)
    perception = PerceptionNetwork(net_path, ds_cfg, ablation=args.ablation)
    fovea = Fovea(image_h=H, image_w=W, size=fov)
    positions = fovea.grid_positions(stride=args.scan_stride)

    # collect per-class images
    by_label: dict[int, list[int]] = {i: [] for i in range(10)}
    for i in range(len(ds)):
        _, l = ds[i]
        l = int(l)
        if len(by_label[l]) < args.per_class:
            by_label[l].append(i)
        if all(len(by_label[c]) >= args.per_class for c in range(10)):
            break

    print(f"{perception.num_neurons} neurons | fov {fov} | {len(positions)} pos | "
          f"{args.per_class}/class | gain {args.signal_gain}")

    per_class = {}
    for c in tqdm(range(10), desc="classes"):
        rows = {"glow_edge": [], "glow_com": [], "edge_com": [],
                "glow_gini": [], "blur_edge_gini": [], "center_frac": []}
        for idx in by_label[c]:
            image, _ = ds[idx]
            glow = info_map(perception, fovea, image, positions, args.settle,
                            args.last, H, W, fov)
            edge = sal.edge_energy_map(image)
            blur_edge = uniform_filter(edge, size=fov)  # match fovea footprint
            wsal = sal.patch_saliency(image, positions, fov)
            wglow = np.array([glow[y + fov // 2, x + fov // 2] for (y, x) in positions])
            rows["glow_edge"].append(sal.safe_corr(wglow, wsal)[0])
            rows["glow_com"].append(com_dist(glow, H, W))
            rows["edge_com"].append(com_dist(edge, H, W))
            rows["glow_gini"].append(gini(glow))
            rows["blur_edge_gini"].append(gini(blur_edge))
            # fraction of glow mass in the centre 16x16
            cs = (H - 16) // 2
            center_mass = glow[cs:cs + 16, cs:cs + 16].sum()
            rows["center_frac"].append(float(center_mass / (glow.sum() + 1e-9)))
        per_class[c] = {k: float(np.mean(v)) for k, v in rows.items()}

    def group(labels, key):
        return float(np.mean([per_class[c][key] for c in labels]))

    summary = {
        "config": vars(args),
        "per_class": {CIFAR_NAMES[c]: per_class[c] for c in range(10)},
        "clean_vs_cluttered": {
            k: {"clean": group(CLEAN, k), "cluttered": group(CLUTTERED, k)}
            for k in ("glow_edge", "glow_com", "edge_com", "glow_gini",
                      "blur_edge_gini", "center_frac")
        },
    }
    out = os.path.join(args.output_dir, f"exp10_glow_localization_{int(time.time())}.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== EXP10: glow localization ===")
    print(f"{'class':<8}{'glow~edge':>10}{'glowCOM':>9}{'edgeCOM':>9}"
          f"{'glowGini':>9}{'blurEdgeGini':>13}{'centerFrac':>11}")
    for c in range(10):
        p = per_class[c]
        tag = "clean" if c in CLEAN else "clutt"
        print(f"{CIFAR_NAMES[c]:<8}{p['glow_edge']:>+10.2f}{p['glow_com']:>9.3f}"
              f"{p['edge_com']:>9.3f}{p['glow_gini']:>9.3f}{p['blur_edge_gini']:>13.3f}"
              f"{p['center_frac']:>11.3f}  {tag}")
    cvc = summary["clean_vs_cluttered"]
    print("\nclean vs cluttered:")
    for k in ("glow_edge", "glow_com", "edge_com", "glow_gini", "blur_edge_gini", "center_frac"):
        print(f"  {k:<16} clean {cvc[k]['clean']:+.3f} | cluttered {cvc[k]['cluttered']:+.3f}")
    print(f"Saved: {out}")
    return summary


def main():
    p = argparse.ArgumentParser(description="EXP10 glow object-vs-texture localization")
    p.add_argument("--dataset-name", default="cifar10_grayscale")
    p.add_argument("--fovea-size", type=int, default=10)
    p.add_argument("--filters", type=int, default=4)
    p.add_argument("--scan-stride", type=int, default=2)
    p.add_argument("--settle", type=int, default=6)
    p.add_argument("--last", type=int, default=3)
    p.add_argument("--signal-gain", type=float, default=1.0)
    p.add_argument("--per-class", type=int, default=4)
    p.add_argument("--ablation", default="none")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="foveation_results")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
