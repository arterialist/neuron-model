"""EXP13 — efficacy saturation as a familiarity/novelty memory.

Motivated by a viewer observation: hold the gaze on a patch and mean synaptic
efficacy grows exponentially then plateaus (~1k+ ticks); move to a new patch and
it grows again; RETURN to a plateaued patch and it stays flat. Reproducible for
any patch. That is the signature of a content-specific, persistent familiarity
memory living in the SLOW (plastic weight) state — invisible to activity probes.

This quantifies it. One retina + perception net, learning ON, run through a
scripted gaze schedule [A, B, A, C, B] holding each patch for T_hold ticks.
We measure, per segment:
  - growth curve of mean efficacy (exponential -> plateau),
  - novelty response = efficacy slope in the first `window` ticks,
  - retention  = novelty slope on RETURN to A vs its FIRST visit
                 (near-zero on return => the trace was retained),
  - content-specificity = overlap of the synapses that grew for A vs B
                 (low overlap => each patch writes its own trace).

Saves compact JSON + a PNG of the growth curves.
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
from snn_classification_realtime.foveation.retina import Retina
from snn_classification_realtime.foveation.perception import (
    PerceptionNetwork, build_fovea_network_json,
)


def early_slope(curve, window):
    w = min(window, len(curve) - 1)
    if w <= 0:
        return 0.0
    return float((curve[w] - curve[0]) / w)


def grew_mask(delta, frac=0.5):
    """Top-`frac` most-grown synapses (a robust 'this patch wrote here' mask)."""
    thr = np.quantile(delta, 1 - frac)
    return delta >= max(thr, 1e-9)


def run(args):
    torch.manual_seed(0); np.random.seed(0)
    ds_cfg = load_dataset_by_name(args.dataset_name, train=True)
    ds_cfg.signal_gain = args.gain
    ds = ds_cfg.dataset
    image, label = ds[args.img_index]
    C, H, W = image.shape

    retina = Retina(H, W, grid=args.grid, fovea_extent=args.fovea_extent)
    chans = C * retina.out_channels_factor
    net_path = os.path.join(args.output_dir, f"famil_net_{chans}x{args.grid}.json")
    os.makedirs(args.output_dir, exist_ok=True)
    build_fovea_network_json(net_path, channels=chans, size=args.grid,
                             layers=[{"type": "conv", "kernel_size": 4, "stride": 2,
                                      "filters": args.filters}], seed=args.seed)
    perc = PerceptionNetwork(net_path, ds_cfg)  # learning ON
    perc.set_plasticity(mode=args.plasticity_mode, lr_error=args.lr_error,
                        weight_decay_tau=args.weight_decay_tau,
                        weight_baseline=args.weight_baseline)
    eidx = perc.efficacy_index()

    # Well-separated gaze centers named A,B,C.
    patches = {
        "A": (H * 0.30, W * 0.30),
        "B": (H * 0.30, W * 0.70),
        "C": (H * 0.70, W * 0.50),
    }
    schedule = ["A", "B", "A", "C", "B"]

    # Per-patch driven-synapse mask: positions of the synapses each patch drives
    # most strongly (top `driven_frac`). This isolates each patch's own trace
    # from the global-saturation creep.
    driven = {}
    for name, (cy, cx) in patches.items():
        retina.set_center(cy, cx)
        sig = perc.patch_to_signals(retina.render(image))
        strengths = np.zeros(perc.efficacy_vector().shape[0], np.float32)
        for (nid, sid, s) in sig:
            if (nid, sid) in eidx:
                strengths[eidx[(nid, sid)]] = abs(s)
        thr = np.quantile(strengths[strengths > 0], 1 - args.driven_frac) if (strengths > 0).any() else 0
        driven[name] = strengths >= max(thr, 1e-6)

    perc.reset()
    seg_bounds = []                       # (name, start, end)
    global_curve = []                     # global mean efficacy per sample
    patch_curve = {p: [] for p in patches}  # per-patch driven-synapse mean per sample
    sample_ticks = []
    seg_delta = {}
    tick = 0
    for name in schedule:
        cy, cx = patches[name]
        retina.set_center(cy, cx)
        sig = perc.patch_to_signals(retina.render(image))
        e0 = perc.efficacy_vector()
        start = tick
        for _ in range(args.hold):
            perc.step(sig)
            if tick % args.sample_every == 0:
                ev = perc.efficacy_vector()
                sample_ticks.append(tick)
                global_curve.append(float(ev.mean()))
                for p in patches:
                    patch_curve[p].append(float(ev[driven[p]].mean()))
            tick += 1
        e1 = perc.efficacy_vector()
        seg_bounds.append((name, start, tick))
        d = np.clip(e1 - e0, 0, None)
        seg_delta.setdefault(name, np.zeros_like(d))
        seg_delta[name] = seg_delta[name] + d

    st = np.array(sample_ticks)

    def seg_rise(patch, seg_name):
        """Mean rise of patch's driven efficacy during segments of seg_name."""
        rises = []
        arr = np.array(patch_curve[patch])
        for (n, s, e) in seg_bounds:
            if n != seg_name:
                continue
            m = (st >= s) & (st < e)
            if m.sum() >= 2:
                rises.append(arr[m][-1] - arr[m][0])
        return float(np.mean(rises)) if rises else 0.0

    # A's own trace should rise during A-segments and stay ~flat during B/C.
    A_on_A = seg_rise("A", "A")
    A_off = 0.5 * (seg_rise("A", "B") + seg_rise("A", "C"))
    specificity = 1.0 - (abs(A_off) / (abs(A_on_A) + 1e-9))  # ~1 => writes only on A
    mA, mB = grew_mask(seg_delta["A"]), grew_mask(seg_delta["B"])
    jacc = float((mA & mB).sum() / ((mA | mB).sum() + 1e-9))
    corrAB = float(np.corrcoef(seg_delta["A"], seg_delta["B"])[0, 1])

    metrics = {
        "img_index": args.img_index, "label": int(label), "schedule": schedule,
        "A_rise_during_A": A_on_A, "A_rise_during_BC": A_off,
        "A_trace_specificity": specificity,   # ~1 => A's trace written only when viewing A
        "AB_grown_synapse_jaccard": jacc, "AB_delta_correlation": corrAB,
        "global_plateau": float(global_curve[-1]),
        "patch_final": {p: float(patch_curve[p][-1]) for p in patches},
    }

    # plot: per-patch driven-synapse efficacy across the schedule
    fig, ax = plt.subplots(figsize=(12, 4.8))
    colors = {"A": "tab:blue", "B": "tab:orange", "C": "tab:green"}
    for p in patches:
        ax.plot(st, patch_curve[p], color=colors[p], lw=2.2, label=f"patch {p} driven-syn")
    ax.plot(st, global_curve, color="gray", lw=1.2, ls=":", label="global mean")
    for (n, s, e) in seg_bounds:
        ax.axvspan(s, e, color=colors[n], alpha=0.07)
        ax.text((s + e) / 2, ax.get_ylim()[1], f"view {n}", color=colors[n],
                fontsize=10, ha="center", va="top")
    ax.set_xlabel("global tick"); ax.set_ylabel("mean efficacy of driven synapses")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title(f"Content-specific familiarity memory (img {args.img_index}, label {label})\n"
                 f"A's trace rises {A_on_A:.2f} when viewing A vs {A_off:.2f} when viewing B/C "
                 f"→ specificity {specificity:.2f}  |  A/B grown-syn Jaccard {jacc:.2f}")
    fig.tight_layout()
    png = os.path.join(args.output_dir, f"exp13_familiarity_{args.img_index}.png")
    fig.savefig(png, dpi=110); plt.close(fig)

    out = os.path.join(args.output_dir, f"exp13_familiarity_{int(time.time())}.json")
    json.dump(metrics, open(out, "w"), indent=2)
    print("=== EXP13 content-specific familiarity memory ===")
    print(f"A's driven-synapse trace rises {A_on_A:.3f} while VIEWING A "
          f"vs {A_off:.3f} while viewing B/C  => specificity {specificity:.2f} "
          f"(1 = written only when A is on screen)")
    print(f"A vs B grown-synapse Jaccard {jacc:.2f} | delta-corr {corrAB:.2f} "
          f"(low => each patch writes its own trace)")
    print(f"per-patch final driven efficacy {metrics['patch_final']} | "
          f"global {metrics['global_plateau']:.3f}")
    print(f"Saved {png}\nSaved {out}")
    return metrics


def main():
    p = argparse.ArgumentParser(description="EXP13 familiarity/novelty memory")
    p.add_argument("--dataset-name", default="cifar10_grayscale")
    p.add_argument("--img-index", type=int, default=0)
    p.add_argument("--grid", type=int, default=16)
    p.add_argument("--fovea-extent", type=int, default=8)
    p.add_argument("--filters", type=int, default=4)
    p.add_argument("--gain", type=float, default=2.0)
    p.add_argument("--hold", type=int, default=1500, help="ticks per segment")
    p.add_argument("--sample-every", type=int, default=10)
    p.add_argument("--window", type=int, default=200, help="ticks for novelty slope")
    p.add_argument("--driven-frac", type=float, default=0.3,
                   help="top fraction of driven synapses to track per patch")
    p.add_argument("--plasticity-mode", default="legacy_multiplicative",
                   choices=["legacy_multiplicative", "error_correcting"])
    p.add_argument("--lr-error", type=float, default=0.05)
    p.add_argument("--weight-decay-tau", type=float, default=0.0)
    p.add_argument("--weight-baseline", type=float, default=1.0)
    p.add_argument("--ablation", default="none")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="foveation_results")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
