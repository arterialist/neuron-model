"""t_ref characterization on VARIABLE inputs over LONG timescales (measurement only).

Question (raised by the user): does the native ALERM reward coupling
  M_vector --(w_tref)--> t_ref --> direction(=+1 if Δt<=t_ref) --> LTP/LTD
actually gate plasticity, or is t_ref pinned so high (>> inter-spike interval)
that every update is causal regardless of reward? My earlier claim that it is
"drowned out" rested only on a FIXED image held 40 ticks — the wrong regime for a
homeostatic variable. This measures it properly: a stream of DIFFERENT images,
switched every `switch_every` ticks, over thousands of ticks.

NO MODEL CHANGES. The perception net runs at pure defaults (legacy multiplicative
rule, nm_plasticity_kappa=0); we only broadcast m0/m1 and observe.

Per tick we log, across neurons:
  - mean t_ref (and its ceiling), mean M0/M1
  - participation (spike fraction)
  - causal_fraction = fraction of neurons with (current_tick - t_last_fire) <= t_ref
    == the fraction of plastic updates that are LTP. THIS is what reward must move.
  - mean Δt (spike recency)

Three arms: baseline (no modulator), reward (broadcast m1 each tick), stress
(broadcast m0 each tick). If reward raises causal_fraction and stress lowers it,
the native path works on variable input; if causal_fraction ≈ 1 everywhere, it is
saturated out and only excitability (w_r) remains as a lever.
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
from snn_classification_realtime.foveation.fovea import Fovea
from snn_classification_realtime.foveation.perception import (
    PerceptionNetwork, build_fovea_network_json,
)


def measure(perc, whole, images, T, switch_every, m0, m1):
    ids = perc._ids
    neurons = perc.sim.network.neurons
    ceil = float(neurons[ids[0]].upper_t_ref_bound)
    lower = float(neurons[ids[0]].lower_t_ref_bound)
    rec = {k: [] for k in ("tref", "M0", "M1", "part", "causal", "dt", "favg")}
    perc.reset()
    img_i = -1
    sig = None
    for t in range(T):
        if t % switch_every == 0:
            img_i = (img_i + 1) % len(images)
            sig = perc.patch_to_signals(whole.crop(images[img_i]))
        if m0 > 0 or m1 > 0:
            perc.broadcast_neuromod(m0, m1)
        st = perc.step(sig)
        ct = perc.sim.current_tick
        trefs = np.empty(len(ids), np.float32)
        dts = np.empty(len(ids), np.float32)
        favg = np.empty(len(ids), np.float32)
        M0 = np.empty(len(ids), np.float32); M1 = np.empty(len(ids), np.float32)
        for j, nid in enumerate(ids):
            n = neurons[nid]
            trefs[j] = n.t_ref
            favg[j] = n.F_avg
            tl = n.t_last_fire
            dts[j] = (ct - tl) if np.isfinite(tl) else 1e9
            M0[j] = n.M_vector[0]; M1[j] = n.M_vector[1] if len(n.M_vector) > 1 else 0.0
        rec["tref"].append(float(trefs.mean()))
        rec["favg"].append(float(favg.mean()))
        rec["M0"].append(float(M0.mean())); rec["M1"].append(float(M1.mean()))
        rec["part"].append(float(st.O.mean()))
        rec["causal"].append(float((dts <= trefs).mean()))
        rec["dt"].append(float(np.median(dts[dts < 1e8])) if (dts < 1e8).any() else 0.0)
    return {k: np.array(v) for k, v in rec.items()}, ceil, lower


def run(args):
    torch.manual_seed(0); np.random.seed(0)
    ds_cfg = load_dataset_by_name(args.dataset_name, train=True)
    ds_cfg.signal_gain = args.gain
    ds = ds_cfg.dataset
    img0, _ = ds[0]; C, H, W = img0.shape
    whole = Fovea(H, W, size=H)
    images = [ds[i][0] for i in range(args.num_images)]

    net_path = os.path.join(args.output_dir, f"tref_net_{C}x{H}.json")
    os.makedirs(args.output_dir, exist_ok=True)
    build_fovea_network_json(net_path, channels=C, size=H,
                             layers=[{"type": "conv", "kernel_size": 4,
                                      "stride": 2, "filters": args.filters}],
                             seed=args.seed)
    # PURE DEFAULTS: no set_plasticity call -> legacy rule, no neuromod gating.
    perc = PerceptionNetwork(net_path, ds_cfg)
    # Runtime param override (NOT a model edit): shorten the F_avg EMA so the
    # homeostat equilibrates faster. beta_avg=0.995 -> tau~200 (vs 0.999 -> ~1000);
    # same steady state, reached ~5x sooner (user's suggestion).
    if args.beta_avg > 0:
        for _n in perc.sim.network.neurons.values():
            _n.params.beta_avg = float(args.beta_avg)

    arms = {"baseline": (0.0, 0.0), "reward_m1": (0.0, args.m), "stress_m0": (args.m, 0.0)}
    out = {}
    ceil = lower = None
    for name, (m0, m1) in arms.items():
        rec, ceil, lower = measure(perc, whole, images, args.ticks, args.switch_every, m0, m1)
        tail = slice(-args.ticks // 4, None)  # steady-state tail (last 1/4)
        out[name] = {
            "tref_mean_tail": float(rec["tref"][tail].mean()),
            "favg_tail": float(rec["favg"][tail].mean()),
            "M0_tail": float(rec["M0"][tail].mean()),
            "M1_tail": float(rec["M1"][tail].mean()),
            "participation_tail": float(rec["part"][tail].mean()),
            "causal_fraction_tail": float(rec["causal"][tail].mean()),
            "median_dt_tail": float(rec["dt"][tail].mean()),
            "_rec": rec,
        }
        print(f"[{name}] t_ref {out[name]['tref_mean_tail']:.1f} (ceil {ceil:.0f}/floor {lower:.0f}) "
              f"| F_avg {out[name]['favg_tail']:.4f} "
              f"| M0 {out[name]['M0_tail']:.3f} M1 {out[name]['M1_tail']:.3f} "
              f"| part {out[name]['participation_tail']:.3f} "
              f"| median Δt {out[name]['median_dt_tail']:.1f} "
              f"| CAUSAL FRAC {out[name]['causal_fraction_tail']:.3f}")

    # plot: F_avg (equilibration), t_ref, causal fraction
    fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    colors = {"baseline": "gray", "reward_m1": "tab:green", "stress_m0": "tab:red"}
    for name in arms:
        r = out[name]["_rec"]
        ax[0].plot(r["favg"], color=colors[name], lw=1.3, label=name)
        ax[1].plot(r["tref"], color=colors[name], lw=1.3, label=name)
        ax[2].plot(r["causal"], color=colors[name], lw=1.3, label=name)
    ax[0].axhline(1.0 / perc.sim.network.neurons[perc._ids[0]].params.c, color="k",
                  ls=":", lw=1, label="F_avg where t_ref hits floor (1/c)")
    ax[0].set_ylabel("mean F_avg (EMA)"); ax[0].legend(fontsize=8)
    ax[0].set_title(f"t_ref homeostat under variable input, LONG timescale "
                    f"({args.num_images} imgs, switch {args.switch_every}t, m={args.m}, "
                    f"F_avg EMA τ≈1000)")
    ax[1].axhline(ceil, color="k", ls=":", lw=1, label=f"ceiling {ceil:.0f}")
    ax[1].axhline(lower, color="k", ls="--", lw=1, label=f"floor {lower:.0f}")
    ax[1].set_ylabel("mean t_ref"); ax[1].legend(fontsize=8)
    ax[2].set_ylabel("causal fraction (LTP)"); ax[2].set_xlabel("tick")
    ax[2].set_ylim(-0.02, 1.02); ax[2].legend(fontsize=8)
    fig.tight_layout()
    png = os.path.join(args.output_dir, "exp_tref_char.png")
    fig.savefig(png, dpi=110); plt.close(fig)

    # verdict
    cb = out["baseline"]["causal_fraction_tail"]
    cr = out["reward_m1"]["causal_fraction_tail"]
    cs = out["stress_m0"]["causal_fraction_tail"]
    summary = {"config": vars(args), "t_ref_ceiling": ceil, "t_ref_floor": lower,
               "arms": {k: {kk: vv for kk, vv in v.items() if kk != "_rec"}
                        for k, v in out.items()},
               "reward_minus_stress_causal": cr - cs}
    js = os.path.join(args.output_dir, f"exp_tref_char_{int(time.time())}.json")
    json.dump(summary, open(js, "w"), indent=2)
    print(f"\nVERDICT: causal fraction  baseline {cb:.3f} | reward {cr:.3f} | "
          f"stress {cs:.3f}  (reward-stress = {cr-cs:+.3f})")
    print("  -> native w_tref reward path is EFFECTIVE if reward-stress is clearly "
          ">0; DROWNED OUT if all ~equal (esp. near 1.0).")
    print(f"Saved {png}\nSaved {js}")
    return summary


def main():
    p = argparse.ArgumentParser(description="t_ref characterization (measurement only)")
    p.add_argument("--dataset-name", default="cifar10_grayscale")
    p.add_argument("--filters", type=int, default=2)
    p.add_argument("--gain", type=float, default=3.0)
    p.add_argument("--ticks", type=int, default=6000)  # >> F_avg EMA tau(~1000)
    p.add_argument("--switch-every", type=int, default=25)
    p.add_argument("--num-images", type=int, default=20)
    p.add_argument("--m", type=float, default=1.0, help="modulator broadcast strength")
    p.add_argument("--beta-avg", type=float, default=0.0,
                   help="runtime F_avg EMA override (0=leave default 0.999); "
                        "0.995 -> tau~200 for faster equilibration")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="foveation_results")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
