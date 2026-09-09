"""Produce a mini-brain run video for visual intuition.

Warms the substrate on a short teacher-on stream, then records a fresh continuous
stream (moving fovea via klinotaxis) and renders the multi-panel video.
"""
from __future__ import annotations

import argparse
import numpy as np

from snn_classification_realtime.activity_dataset_builder.vision_datasets import (
    load_dataset_by_name,
)
from snn_classification_realtime.foveation.minibrain.core import MiniBrainConfig, MiniBrain
from snn_classification_realtime.foveation.minibrain.render_run import (
    record_run, render_video,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-name", default="cifar10_grayscale")
    p.add_argument("--warm", type=int, default=300, help="teacher-on warmup images")
    p.add_argument("--record", type=int, default=6, help="images to record in the video")
    p.add_argument("--dwell", type=int, default=40)
    p.add_argument("--tonic", type=float, default=0.15)
    p.add_argument("--gaze", default="klinotaxis", choices=["klinotaxis", "center"])
    p.add_argument("--fovea", type=int, default=8)
    p.add_argument("--grid", type=int, default=12)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="foveation_results/minibrain/run.mp4")
    args = p.parse_args()

    ds_cfg = load_dataset_by_name(args.dataset_name, train=True)
    ds = ds_cfg.dataset
    cfg = MiniBrainConfig(dataset_name=args.dataset_name, tonic_drive=args.tonic,
                          target_participation=0.05, dwell=args.dwell,
                          fovea=args.fovea, grid=args.grid, seed=args.seed)
    brain = MiniBrain(cfg, ds_cfg)

    rng = np.random.RandomState(args.seed)
    if args.warm > 0:
        for idx in rng.randint(0, len(ds), size=args.warm):
            img, y = ds[idx]
            brain.present(img, int(y), learn=True, teach=True)
        print(f"[warm] trained on {args.warm} images")

    ridx = rng.randint(0, len(ds), size=args.record)
    imgs = [ds[i][0] for i in ridx]; labs = [int(ds[i][1]) for i in ridx]
    rec = record_run(brain, imgs, labs, ticks_per_image=args.dwell,
                     teach=True, learn=False, gaze_mode=args.gaze, seed=args.seed)
    print(f"[record] {len(rec.ticks)} frames")
    out = render_video(rec, args.out, fps=args.fps)
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()
