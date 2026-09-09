"""Stage A -- the crux: does native ALERM teacher feedback make the PAULA
substrate representation more class-separable over continuous time?

Protocol (continuous, no reset between images):
  1. Stream labeled CIFAR images, one fixation each. Teacher (true label) injects
     m0/m1 -> native w_r/w_tref -> reward-gated LTP / stress LTD in the substrate.
     The online linear decoder trains as it goes.
  2. Periodically, measure the SUBSTRATE's own separability with a FRESH kNN probe
     on buffered (rep, label) -- this is independent of the online decoder, so a
     rise means the *substrate* (not just the decoder) is learning representations.
  3. Control arm: identical stream with the teacher SILENT (no neuromod) -- isolates
     what the teacher adds over passive drift.
  4. Teacher-removal test: after training, freeze learning + silence teacher, stream
     held-out images, report decoder accuracy on the trained-but-unsupervised substrate.

Measurement-only on neuron.py (legacy rule, native path). Exploration, not a benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from snn_classification_realtime.activity_dataset_builder.vision_datasets import (
    load_dataset_by_name,
)
from snn_classification_realtime.foveation.minibrain.core import (
    MiniBrainConfig, MiniBrain, knn_accuracy,
)


def train_arm(cfg, ds, order, probe_every, teach, log):
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    brain = MiniBrain(cfg, ds_cfg_for(cfg))
    buf_x, buf_y = [], []
    run_correct, run_n = 0, 0
    rec = {"probe_tick": [], "probe_knn": [], "online_acc": [], "reward": [], "eff": []}
    t0 = time.time()
    for i, idx in enumerate(order):
        img, y = ds[idx]; y = int(y)
        x, pred = brain.present(img, y, learn=True, teach=teach)
        buf_x.append(x); buf_y.append(y)
        run_correct += int(pred == y); run_n += 1
        if len(buf_x) > cfg_buf:
            buf_x.pop(0); buf_y.pop(0)
        if (i + 1) % probe_every == 0:
            knn = knn_accuracy(buf_x, buf_y, k=5) if len(set(buf_y)) > 1 else float("nan")
            oa = run_correct / max(1, run_n)
            m1 = brain._last_nm[1] if hasattr(brain, "_last_nm") else 0.0
            eff = brain.sub.mean_efficacy()
            rec["probe_tick"].append(i + 1); rec["probe_knn"].append(knn)
            rec["online_acc"].append(oa); rec["reward"].append(m1); rec["eff"].append(eff)
            log(f"  [{'TEACH' if teach else 'CTRL '}] {i+1:5d}/{len(order)} | "
                f"probe kNN {knn:.3f} | online acc {oa:.3f} | eff {eff:.2f} "
                f"| {time.time()-t0:.0f}s")
            run_correct, run_n = 0, 0
    return brain, rec, (buf_x, buf_y)


def removal_test(brain, ds, order, log):
    """Teacher silent + learning frozen: does the decoder still classify?"""
    correct = 0
    for idx in order:
        img, y = ds[idx]; y = int(y)
        _, pred = brain.present(img, int(y), learn=False, teach=False)
        correct += int(pred == y)
    acc = correct / max(1, len(order))
    log(f"  TEACHER-REMOVED decoder accuracy on held-out stream: {acc:.3f}")
    return acc


_DSC = {}
def ds_cfg_for(cfg):
    if cfg.dataset_name not in _DSC:
        d = load_dataset_by_name(cfg.dataset_name, train=True)
        _DSC[cfg.dataset_name] = d
    return _DSC[cfg.dataset_name]


cfg_buf = 800   # rolling buffer for the fresh-probe separability


def run(args):
    global cfg_buf
    cfg_buf = args.buffer
    cfg = MiniBrainConfig(
        dataset_name=args.dataset_name, connectivity=args.connectivity,
        w_tref_scale=args.w_tref_scale, w_r_scale=args.w_r_scale,
        dwell=args.dwell, reward_gain=args.reward_gain, stress_gain=args.stress_gain,
        teacher_mode=args.teacher_mode, decoder_lr=args.decoder_lr,
        seed=args.seed, output_dir=args.output_dir,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    logpath = os.path.join(args.output_dir, "stage_a.log")
    lf = open(logpath, "a")
    def log(m): print(m); lf.write(m + "\n"); lf.flush()

    ds = ds_cfg_for(cfg).dataset
    rng = np.random.RandomState(args.seed)
    N = args.samples
    order = rng.randint(0, len(ds), size=N)
    held = rng.randint(0, len(ds), size=args.held)

    log(f"\n=== Stage A | {cfg.dataset_name} | conn {cfg.connectivity} | "
        f"w_tref x{cfg.w_tref_scale} w_r x{cfg.w_r_scale} | dwell {cfg.dwell} | "
        f"N {N} ===")
    log("--- TEACHER ON ---")
    brain_t, rec_t, _ = train_arm(cfg, ds, order, args.probe_every, True, log)
    acc_t = removal_test(brain_t, ds, held, log)
    log("--- CONTROL (teacher silent) ---")
    brain_c, rec_c, _ = train_arm(cfg, ds, order, args.probe_every, False, log)
    acc_c = removal_test(brain_c, ds, held, log)

    _plot(args, cfg, rec_t, rec_c, acc_t, acc_c)
    summary = {"config": vars(args),
               "teacher_removed_acc": acc_t, "control_removed_acc": acc_c,
               "final_probe_knn_teach": rec_t["probe_knn"][-1] if rec_t["probe_knn"] else None,
               "final_probe_knn_ctrl": rec_c["probe_knn"][-1] if rec_c["probe_knn"] else None,
               "rec_teach": rec_t, "rec_ctrl": rec_c}
    js = os.path.join(args.output_dir, f"stage_a_{int(time.time())}.json")
    json.dump(summary, open(js, "w"), indent=2)
    log(f"\nVERDICT: substrate separability (fresh kNN)  teacher {summary['final_probe_knn_teach']}"
        f"  vs control {summary['final_probe_knn_ctrl']}")
    log(f"         teacher-removed decoder acc  teacher-trained {acc_t:.3f}  vs control {acc_c:.3f}")
    log(f"Saved {js}")


def _plot(args, cfg, rec_t, rec_c, acc_t, acc_c):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    xt, xc = rec_t["probe_tick"], rec_c["probe_tick"]
    ax[0].plot(xt, rec_t["probe_knn"], "-o", color="tab:green", ms=3, label="teacher")
    ax[0].plot(xc, rec_c["probe_knn"], "-o", color="tab:gray", ms=3, label="control")
    ax[0].axhline(0.1, color="k", ls=":", lw=1, label="chance")
    ax[0].set_title("substrate separability (fresh kNN probe)")
    ax[0].set_xlabel("images seen"); ax[0].set_ylabel("kNN accuracy"); ax[0].legend(fontsize=8)
    ax[1].plot(xt, rec_t["online_acc"], "-o", color="tab:green", ms=3, label="teacher")
    ax[1].plot(xc, rec_c["online_acc"], "-o", color="tab:gray", ms=3, label="control")
    ax[1].axhline(0.1, color="k", ls=":", lw=1)
    ax[1].set_title("online decoder accuracy"); ax[1].set_xlabel("images seen")
    ax[1].set_ylabel("running acc"); ax[1].legend(fontsize=8)
    ax[2].plot(xt, rec_t["eff"], "-o", color="tab:green", ms=3, label="teacher")
    ax[2].plot(xc, rec_c["eff"], "-o", color="tab:gray", ms=3, label="control")
    ax[2].set_title("mean synaptic efficacy"); ax[2].set_xlabel("images seen")
    ax[2].set_ylabel("mean u_i.info"); ax[2].legend(fontsize=8)
    fig.suptitle(f"Mini-brain Stage A: {cfg.dataset_name}, conn {cfg.connectivity}, "
                 f"w_tref x{cfg.w_tref_scale}  |  teacher-removed acc: "
                 f"teach {acc_t:.2f} vs ctrl {acc_c:.2f}", fontsize=11)
    fig.tight_layout()
    png = os.path.join(args.output_dir, "stage_a.png")
    fig.savefig(png, dpi=120); plt.close(fig)
    print(f"Saved {png}")


def main():
    p = argparse.ArgumentParser(description="Mini-brain Stage A (crux mechanism)")
    p.add_argument("--dataset-name", default="cifar10_grayscale")
    p.add_argument("--connectivity", type=float, default=0.2)
    p.add_argument("--w-tref-scale", type=float, default=1.0)
    p.add_argument("--w-r-scale", type=float, default=1.0)
    p.add_argument("--dwell", type=int, default=40)
    p.add_argument("--reward-gain", type=float, default=1.0)
    p.add_argument("--stress-gain", type=float, default=1.0)
    p.add_argument("--teacher-mode", default="graded", choices=["graded", "binary"])
    p.add_argument("--decoder-lr", type=float, default=0.05)
    p.add_argument("--samples", type=int, default=2000)
    p.add_argument("--held", type=int, default=400)
    p.add_argument("--buffer", type=int, default=800)
    p.add_argument("--probe-every", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="foveation_results/minibrain/stage_a")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
