"""Render a gallery of dynamics animations (one per architecture) for the findings
artifact, plus a representative still frame each. Visual-intuition deliverable."""
from __future__ import annotations
import argparse, os, subprocess
import numpy as np

from snn_classification_realtime.foveation.minibrain.core import MiniBrainConfig, MiniBrain
from snn_classification_realtime.foveation.minibrain.render_run import record_run, render_video
from snn_classification_realtime.foveation.minibrain.cluttered import make_cluttered_cfg
from snn_classification_realtime.activity_dataset_builder.vision_datasets import load_dataset_by_name


def warmup(brain, ds, n=500):
    per = max(1, brain.cfg.input_period); done = 0; i = 0
    brain.sub.set_learning(False)
    while done < n:
        sig = brain.sub.patch_to_signals(brain._encode(ds[i % 40][0]))
        for t in range(brain.cfg.dwell):
            brain.sub.step((sig if t % per == 0 else []) + brain._tonic); done += 1
            if done >= n:
                break
        i += 1
    brain.sub.set_learning(True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--arch", default="reservoir_retino",
                   choices=["reservoir_random", "reservoir_retino", "conv"])
    p.add_argument("--dataset", default="cifar10_grayscale")
    p.add_argument("--cluttered", type=int, default=0)
    p.add_argument("--images", type=int, default=16)
    p.add_argument("--ticks", type=int, default=60)
    p.add_argument("--out", default="foveation_results/minibrain/gallery")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    substrate = "conv" if a.arch == "conv" else "reservoir"
    wiring = "retinotopic" if a.arch == "reservoir_retino" else "random"
    cfg = MiniBrainConfig(dataset_name=a.dataset, dwell=a.ticks, substrate_type=substrate,
                          wiring=wiring, conv_bank="rich", readout="all",
                          output_dir=os.path.join(a.out, f"net_{a.tag}"), seed=0)
    if a.cluttered:
        cfg.fovea = 44; cfg.periph = 72; cfg.grid = 16
        dscfg = make_cluttered_cfg(a.dataset, 72, 4, 16, seed=999)
    else:
        dscfg = load_dataset_by_name(a.dataset, train=True)
    brain = MiniBrain(cfg, dscfg); ds = dscfg.dataset
    warmup(brain, ds, 500)
    idxs = list(range(a.images))
    imgs = [ds[i][0] for i in idxs]; labs = [int(ds[i][1]) for i in idxs]
    rec = record_run(brain, imgs, labs, ticks_per_image=a.ticks, teach=True, learn=False,
                     gaze_mode="klinotaxis", seed=0)
    path = os.path.join(a.out, f"{a.tag}.mp4")
    outp = render_video(rec, path, fps=12)
    # extract a representative still frame (~60% through) for the artifact
    frame = os.path.join(a.out, f"{a.tag}.png")
    try:
        import shutil
        ff = shutil.which("ffmpeg") or "ffmpeg"
        subprocess.run([ff, "-y", "-i", outp, "-vf",
                        "select=eq(n\\,%d)" % int(len(rec.ticks) * 0.6), "-vframes", "1", frame],
                       check=False, capture_output=True)
    except Exception as e:
        print("frame extract failed:", e)
    print(f"GALLERY {a.tag}: {outp}  frame={os.path.exists(frame)}", flush=True)


if __name__ == "__main__":
    main()
