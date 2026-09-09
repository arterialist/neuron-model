"""Animated retinotopic glow: watch the object emerge as the fovea scans.

Renders an mp4 per image with 6 panels:
  1. input + moving fovea box
  2. edge energy (target)
  3. edge energy overlaid semi-transparently on the input
  4. informativeness glow, accumulating as the fovea visits positions
  5. firing-rate glow, accumulating
  6. informativeness glow overlaid semi-transparently on the input

Scan order is either raster (full grid) or random (a bounded random walk, which
previews the P2a diffusion controller: the eye wanders and the glow fills in
only where it has looked).
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, PillowWriter
from matplotlib.patches import Rectangle

from snn_classification_realtime.activity_dataset_builder.vision_datasets import (
    load_dataset_by_name,
)
from snn_classification_realtime.foveation.fovea import Fovea
from snn_classification_realtime.foveation.perception import (
    PerceptionNetwork,
    build_fovea_network_json,
)
from snn_classification_realtime.foveation import saliency as sal


def gray(image: torch.Tensor) -> np.ndarray:
    img = image.detach().cpu().numpy()
    return img.mean(0) if img.ndim == 3 else img


def measure(perception, fovea, image, y, x, settle, last):
    perception.reset()
    fovea.set_position(y, x)
    sig = perception.patch_to_signals(fovea.crop(image))
    rate = 0.0
    S_hist = []
    for t in range(settle):
        st = perception.step(sig)
        if t >= settle - last:
            rate += float(st.O.mean())
            S_hist.append(st.S)
    return rate / last, float(np.mean([np.std(s) for s in S_hist]))


def random_walk_positions(fovea: Fovea, num_steps: int, step_max: int, rng) -> list:
    """A bounded random walk over integer top-left positions from the center."""
    y, x = fovea.max_fy // 2, fovea.max_fx // 2
    seq = [(y, x)]
    for _ in range(num_steps):
        y = int(np.clip(y + rng.integers(-step_max, step_max + 1), 0, fovea.max_fy))
        x = int(np.clip(x + rng.integers(-step_max, step_max + 1), 0, fovea.max_fx))
        seq.append((y, x))
    return seq


def splat(acc, count, y, x, fov, val):
    acc[y:y + fov, x:x + fov] += val
    count[y:y + fov, x:x + fov] += 1.0


def show(m, cnt):
    return np.divide(m, cnt, out=np.zeros_like(m), where=cnt > 0)


def run(args):
    torch.manual_seed(0)
    np.random.seed(0)
    rng = np.random.default_rng(args.seed)
    ds_cfg = load_dataset_by_name(args.dataset_name, train=True)
    ds_cfg.signal_gain = args.signal_gain
    ds = ds_cfg.dataset
    img0, _ = ds[0]
    channels, H, W = img0.shape

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

    anim_dir = os.path.join(args.output_dir, "anim")
    os.makedirs(anim_dir, exist_ok=True)
    ext = "mp4" if args.writer == "ffmpeg" else "gif"

    n = min(args.num_images, len(ds))
    step = max(1, len(ds) // n)
    idxs = list(range(0, step * n, step))[:n]
    fov = args.fovea_size

    for img_idx in idxs:
        image, label = ds[img_idx]
        g = gray(image)
        edge = sal.edge_energy_map(image)
        edge_n = edge / (edge.max() + 1e-9)

        if args.scan_order == "random":
            num_steps = args.random_steps or (len(fovea.grid_positions(args.scan_stride)) * 2)
            positions = random_walk_positions(fovea, num_steps, args.step_max, rng)
        else:
            positions = fovea.grid_positions(stride=args.scan_stride)

        # Cache activity per unique position (random walk revisits).
        cache: dict[tuple[int, int], tuple[float, float]] = {}
        for (y, x) in set(positions):
            cache[(y, x)] = measure(perception, fovea, image, y, x, args.settle, args.last)
        rates = np.array([cache[p][0] for p in cache])
        infos = np.array([cache[p][1] for p in cache])
        rmax = rates.max() + 1e-9
        imax = infos.max() + 1e-9

        fig, ax = plt.subplots(1, 6, figsize=(19, 3.4))
        for a in ax:
            a.axis("off")
        ax[0].imshow(g, cmap="gray"); ax[0].set_title("input + fovea")
        im_edge = ax[1].imshow(np.zeros((H, W)), cmap="magma", vmin=0, vmax=1)
        ax[1].set_title("edge energy")
        ax[2].imshow(g, cmap="gray")
        im_edge_over = ax[2].imshow(np.zeros((H, W)), cmap="magma", alpha=0.0,
                                    vmin=0, vmax=1)
        ax[2].set_title("edge over input")
        im_info = ax[3].imshow(np.zeros((H, W)), cmap="inferno", vmin=0, vmax=1)
        ax[3].set_title("glow: informativeness")
        im_rate = ax[4].imshow(np.zeros((H, W)), cmap="inferno", vmin=0, vmax=1)
        ax[4].set_title("glow: firing rate")
        ax[5].imshow(g, cmap="gray")
        im_over = ax[5].imshow(np.zeros((H, W)), cmap="inferno", alpha=0.0, vmin=0, vmax=1)
        ax[5].set_title("glow over input")
        box = Rectangle((0, 0), fov, fov, fill=False, edgecolor="cyan", lw=1.5)
        ax[0].add_patch(box)
        fig.suptitle(
            f"label {int(label)} · fovea {fov}px · gain {args.signal_gain} · "
            f"scan={args.scan_order}"
        )
        fig.tight_layout()

        info_acc = np.zeros((H, W), np.float32); info_cnt = np.zeros((H, W), np.float32)
        rate_acc = np.zeros((H, W), np.float32); rate_cnt = np.zeros((H, W), np.float32)

        writer = (FFMpegWriter(fps=args.fps, bitrate=2600)
                  if args.writer == "ffmpeg" else PillowWriter(fps=args.fps))
        out = os.path.join(
            anim_dir, f"glow_{args.scan_order}_{img_idx}_label{int(label)}.{ext}"
        )
        with writer.saving(fig, out, dpi=args.dpi):
            for pi, (y, x) in enumerate(tqdm(positions, desc=f"img {img_idx}", leave=False)):
                r, iv = cache[(y, x)]
                splat(info_acc, info_cnt, y, x, fov, iv / imax)
                splat(rate_acc, rate_cnt, y, x, fov, r / rmax)
                box.set_xy((x, y))
                imap = show(info_acc, info_cnt)
                # Reveal edge energy only where the fovea has looked (in sync).
                visited = info_cnt > 0
                revealed_edge = edge_n * visited
                im_edge.set_data(revealed_edge)
                im_edge_over.set_data(revealed_edge)
                im_edge_over.set_alpha(0.6)
                im_info.set_data(imap)
                im_rate.set_data(show(rate_acc, rate_cnt))
                im_over.set_data(imap)
                im_over.set_alpha(0.6)
                if pi % args.frame_every == 0 or pi == len(positions) - 1:
                    writer.grab_frame()
            for _ in range(args.hold_frames):
                writer.grab_frame()
        plt.close(fig)
        print(f"Saved {out}")


def main():
    p = argparse.ArgumentParser(description="Animated glow scan")
    p.add_argument("--dataset-name", default="cifar10_grayscale")
    p.add_argument("--fovea-size", type=int, default=10)
    p.add_argument("--filters", type=int, default=4)
    p.add_argument("--scan-order", default="raster", choices=["raster", "random"])
    p.add_argument("--scan-stride", type=int, default=1)
    p.add_argument("--random-steps", type=int, default=0, help="0 = 2x grid size")
    p.add_argument("--step-max", type=int, default=3, help="max px per random step")
    p.add_argument("--settle", type=int, default=8)
    p.add_argument("--last", type=int, default=4)
    p.add_argument("--signal-gain", type=float, default=1.0)
    p.add_argument("--num-images", type=int, default=2)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--dpi", type=int, default=90)
    p.add_argument("--frame-every", type=int, default=2)
    p.add_argument("--hold-frames", type=int, default=15)
    p.add_argument("--writer", default="ffmpeg", choices=["ffmpeg", "pillow"])
    p.add_argument("--ablation", default="none")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="foveation_results")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
