"""Animated settling in the (<S>, <t_ref>) attractor space.

The PAULA paper visualizes memory as basins in (<S>, <t_ref>) space, one per
class. This animates several fixed inputs of different labels settling into that
space in real time: left = live spike raster (which neurons fire), right = the
phase-plane trajectory tracing out, one colored path per input. Watching the
paths diverge is the world-modeling premise made visible.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, PillowWriter

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
    meanS = np.empty(ticks); meanT = np.empty(ticks)
    O = np.empty((ticks, n), np.float32)
    for t in range(ticks):
        st = perception.step(sig)
        meanS[t] = np.mean(st.S)
        meanT[t] = np.mean(st.t_ref)
        O[t] = st.O
    return meanS, meanT, O


def run(args):
    torch.manual_seed(0); np.random.seed(0)
    ds_cfg = load_dataset_by_name(args.dataset_name, train=True)
    ds_cfg.signal_gain = args.signal_gain
    ds = ds_cfg.dataset
    img0, _ = ds[0]
    channels, H, W = img0.shape

    net_path = os.path.join(args.output_dir, f"fovea_deep_{channels}x{args.fovea_size}.json")
    os.makedirs(args.output_dir, exist_ok=True)
    build_fovea_network_json(net_path, channels=channels, size=args.fovea_size, seed=args.seed)
    perception = PerceptionNetwork(net_path, ds_cfg, ablation=args.ablation)
    fovea = Fovea(image_h=H, image_w=W, size=args.fovea_size)
    cy, cx = fovea.max_fy // 2, fovea.max_fx // 2

    # one image per label
    picks = []
    seen = set()
    for i in range(len(ds)):
        _, l = ds[i]; l = int(l)
        if l < args.num_labels and l not in seen:
            picks.append((i, l)); seen.add(l)
        if len(seen) >= args.num_labels:
            break

    trajs = []
    for idx, lbl in picks:
        image, _ = ds[idx]
        meanS, meanT, O = run_fixed(perception, fovea, image, cy, cx, args.ticks)
        trajs.append((lbl, meanS, meanT, O))
        print(f"label {lbl}: <S> {meanS[-1]:+.3f}  <t_ref> {meanT[-1]:.2f}")

    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    allS = np.concatenate([t[1] for t in trajs])
    allT = np.concatenate([t[2] for t in trajs])
    n = perception.num_neurons

    fig, (axr, axp) = plt.subplots(1, 2, figsize=(12, 5))
    axr.set_title("spike raster (label 0 shown)")
    axr.set_xlabel("tick"); axr.set_ylabel("neuron")
    axr.set_xlim(0, args.ticks); axr.set_ylim(0, n)
    axp.set_title("attractor space  (<S> vs <t_ref>)")
    axp.set_xlabel("<S>  (mean membrane)"); axp.set_ylabel("<t_ref>  (mean window)")
    axp.set_xlim(allS.min() - 0.05, allS.max() + 0.05)
    axp.set_ylim(allT.min() - 0.5, allT.max() + 0.5)
    lines = [axp.plot([], [], color=colors[t[0]], lw=1.6, label=f"label {t[0]}")[0]
             for t in trajs]
    dots = [axp.plot([], [], "o", color=colors[t[0]], ms=8)[0] for t in trajs]
    axp.legend(loc="upper right", fontsize=8)
    raster_scat = axr.scatter([], [], s=2, c="k")

    writer = (FFMpegWriter(fps=args.fps, bitrate=2400)
              if args.writer == "ffmpeg" else PillowWriter(fps=args.fps))
    ext = "mp4" if args.writer == "ffmpeg" else "gif"
    out = os.path.join(args.output_dir, "anim", f"attractor_settling.{ext}")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    raster_O = trajs[0][3]
    with writer.saving(fig, out, dpi=args.dpi):
        for t in range(1, args.ticks + 1, args.frame_every):
            for li, (lbl, meanS, meanT, O) in enumerate(trajs):
                lines[li].set_data(meanS[:t], meanT[:t])
                dots[li].set_data([meanS[t - 1]], [meanT[t - 1]])
            ys, xs = np.nonzero(raster_O[:t])
            raster_scat.set_offsets(np.c_[ys, xs] if len(ys) else np.empty((0, 2)))
            axp.set_title(f"attractor space  ·  tick {t}/{args.ticks}")
            writer.grab_frame()
        for _ in range(args.hold_frames):
            writer.grab_frame()
    plt.close(fig)
    print(f"Saved {out}")


def main():
    p = argparse.ArgumentParser(description="Animated attractor-space settling")
    p.add_argument("--dataset-name", default="cifar10_grayscale")
    p.add_argument("--fovea-size", type=int, default=16)
    p.add_argument("--ticks", type=int, default=400)
    p.add_argument("--num-labels", type=int, default=5)
    p.add_argument("--signal-gain", type=float, default=1.0)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--dpi", type=int, default=90)
    p.add_argument("--frame-every", type=int, default=2)
    p.add_argument("--hold-frames", type=int, default=20)
    p.add_argument("--writer", default="ffmpeg", choices=["ffmpeg", "pillow"])
    p.add_argument("--ablation", default="none")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="foveation_results")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
