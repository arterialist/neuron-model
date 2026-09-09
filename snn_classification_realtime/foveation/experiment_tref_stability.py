"""Sparse vs dense MLP: does t_ref stabilize under constant drive? (measurement only)

USER CLAIM UNDER TEST (PAULA paper, ~15k+ tick scale): in a DENSE MLP the
metaplastic learning window t_ref fails to stabilize, whereas in a SPARSE MLP it
settles to equilibrium.

MECHANISTIC FRAME (verified by reading neuron.py, not assumed):
  * t_ref is used at exactly ONE place -- the plasticity direction test
    `direction = +1 if (tick - t_last_fire) <= t_ref else -1`. It NEVER feeds
    back into firing (firing = S >= threshold and dt >= c). So t_ref is a *pure
    passive readout* of the firing-rate EMA F_avg via
        t_ref = clip(ceil - (ceil-floor)*clip(F_avg*c,0,1), floor, ceil)
    with ceil = c*num_inputs, floor = 2c. "t_ref stabilizes" is therefore
    EXACTLY "population firing rate F_avg reaches a fixed point."
  * F_avg is an EMA with beta_avg (default 0.999 -> tau~1000 ticks). For it to
    fail to settle over 15k ticks the firing dynamics must be non-stationary on
    timescales comparable to tau (slow drift / mode switching / wandering), not
    just fast oscillation (which the EMA would average out).

NO MODEL CHANGES. Pure defaults (legacy rule). We build MLPs at several
connectivity levels, drive each with a CONSTANT input for many ticks, and record
the t_ref / F_avg / participation trajectories. Artifact = raw .npz + JSON +
static PNG + animated histogram (dense vs sparse side by side).
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
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

from snn_classification_realtime.activity_dataset_builder.vision_datasets import (
    load_dataset_by_name,
)
from snn_classification_realtime.network_builder_direct import (
    build_network_config_direct,
)
from snn_classification_realtime.activity_dataset_builder.drive_calibration import (
    estimate_input_drive,
)
from snn_classification_realtime.foveation.perception import PerceptionNetwork


def autoscale_gain(perc, ds_cfg, target_ratio, probe_n=10, verbose=True):
    """Set ds_cfg.signal_gain so the INPUT layer's mean drive/threshold ratio
    hits `target_ratio` (image-driven band ~1.0-2.5; >2.54 saturates and carries
    no image information). Mirrors build_activity_dataset.py --auto-gain. Returns
    the DriveReport (or None if disabled with target_ratio<=0)."""
    if target_ratio is None or target_ratio <= 0:
        return None
    idx = list(range(min(probe_n, len(ds_cfg.dataset))))
    rep = estimate_input_drive(perc.sim, perc.input_layer_ids,
                               perc.input_synapses_per_neuron, ds_cfg, idx,
                               target_ratio=target_ratio)
    ds_cfg.signal_gain = float(rep.suggested_gain)
    if verbose:
        print(f"  [auto-gain] input ratio {rep.ratio_mean:.2f} ({rep.verdict}, "
              f"{rep.frac_saturated:.0%} sat) -> gain {ds_cfg.signal_gain:.4g} "
              f"(target {target_ratio})")
    return rep


def build_mlp_json(path, *, channels, size, input_size, hidden, sizes,
                   connectivity, seed):
    """Build a dense-first MLP: input_size input neurons + `hidden` dense layers
    of width `sizes`, every dense layer wired at Bernoulli prob `connectivity`.
    num_synapses per dense neuron defaults to prev-layer width, so the t_ref
    ceiling (c*num_inputs) is matched across connectivity arms -- the ONLY thing
    that changes is how many of those synapses are actually driven."""
    import random
    random.seed(seed); np.random.seed(seed)
    layers = [{"type": "dense", "size": sizes, "connectivity": connectivity}
              for _ in range(hidden)]
    cfg = {"dataset": "cifar10_grayscale" if channels == 1 else "cifar10",
           "input_size": input_size, "layers": layers}
    config_out = build_network_config_direct(cfg, input_shape=(channels, size, size))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(config_out, f)
    return path


def run_arm(perc, signals, T, beta_avg, switch_every=0):
    """Drive `perc` for T ticks; record trajectories.

    `signals` is either a single signal list (constant drive, switch_every=0) or
    a list of signal lists cycled every `switch_every` ticks (streaming drive --
    the realistic regime where F_avg is NOT railed at the firing cap, so a dense
    net can actually destabilize if the claim holds).

    Because upper_t_ref_bound = c*num_inputs is PER-NEURON (input neurons carry
    many more synapses than hidden neurons), raw mean t_ref is not comparable
    across arms. We record a per-neuron NORMALIZED SATURATION
        sat_i = (t_ref_i - floor_i) / (ceil_i - floor_i)  in [0,1]
    (0 = pinned at floor / max firing, 1 = pinned at ceiling / silent). This is
    the apples-to-apples stability signal. We also sample mean synaptic efficacy
    (weight growth) -- the legacy multiplicative rule saturates weights over
    thousands of ticks, the suspected driver of any late destabilization."""
    if beta_avg > 0:
        for n in perc.sim.network.neurons.values():
            n.params.beta_avg = float(beta_avg)
    ids = perc._ids
    neurons = perc.sim.network.neurons
    layer_of = perc.layer_of_pos
    ceil_v = np.array([float(neurons[i].upper_t_ref_bound) for i in ids], np.float32)
    floor_v = np.array([float(neurons[i].lower_t_ref_bound) for i in ids], np.float32)
    rng = np.maximum(ceil_v - floor_v, 1e-6)
    perc.reset()
    rec = {k: np.empty(T, np.float32) for k in
           ("tref_mean", "tref_std", "favg_mean", "part", "sat_mean", "sat_std")}
    stream = isinstance(signals[0], list) if signals else False
    snap_every = max(1, T // 240)
    snaps, snap_ticks, eff = [], [], []
    cur = -1
    for t in range(T):
        if switch_every > 0 and stream and t % switch_every == 0:
            cur = (cur + 1) % len(signals)
            signal = signals[cur]
        elif t == 0:
            signal = signals[0] if stream else signals
        st = perc.step(signal)
        trefs = st.t_ref
        sat = (trefs - floor_v) / rng
        rec["tref_mean"][t] = trefs.mean()
        rec["tref_std"][t] = trefs.std()
        rec["sat_mean"][t] = sat.mean()
        rec["sat_std"][t] = sat.std()
        rec["favg_mean"][t] = st.F_avg.mean()
        rec["part"][t] = st.O.mean()
        if t % snap_every == 0:
            snaps.append(sat.copy()); snap_ticks.append(t)
            eff.append(perc.mean_efficacy())
    rec["eff"] = np.array(eff, np.float32)
    return (rec, np.array(snap_ticks), np.array(snaps), ceil_v, floor_v, layer_of)


def stabilization_metrics(series, band=0.01, tail_frac=0.4):
    """Quantify whether `series` (normalized saturation in [0,1]) has settled.
    `band` is the absolute +-tolerance for the settle-time test."""
    series = np.asarray(series, np.float64)
    T = len(series)
    t0 = int(T * (1 - tail_frac))
    tail = series[t0:]
    x = np.arange(len(tail), dtype=np.float64)
    slope_per_kt = float(np.polyfit(x, tail, 1)[0] * 1000.0)  # per 1000 ticks
    tail_std = float(tail.std())
    tail_range = float(tail.max() - tail.min())
    final = float(series[-1])
    outside = np.where(np.abs(series - final) > band)[0]
    settle_tick = int(outside[-1] + 1) if len(outside) else 0
    return {"tail_slope_per_1000t": slope_per_kt, "tail_std": tail_std,
            "tail_range": tail_range, "final_sat": final,
            "settle_tick": settle_tick, "settle_frac": settle_tick / T}


def run(args):
    torch.manual_seed(0); np.random.seed(0)
    ds_cfg = load_dataset_by_name(args.dataset_name, train=True)
    ds_cfg.signal_gain = args.gain
    ds = ds_cfg.dataset
    img0, _ = ds[args.img_index]
    C, H, W = img0.shape
    imgs = [ds[(args.img_index + i) % len(ds)][0] for i in range(args.num_images)]
    os.makedirs(args.output_dir, exist_ok=True)
    regime = f"stream(switch{args.switch_every}, {args.num_images}img)" \
        if args.switch_every > 0 else "constant"

    conns = [float(x) for x in args.connectivity.split(",")]
    arms = {}
    for p in conns:
        tag = f"c{p:g}"
        net_path = os.path.join(args.output_dir, f"mlp_{tag}_{C}x{H}.json")
        build_mlp_json(net_path, channels=C, size=H, input_size=args.input_size,
                       hidden=args.hidden, sizes=args.width, connectivity=p,
                       seed=args.seed)
        perc = PerceptionNetwork(net_path, ds_cfg)
        autoscale_gain(perc, ds_cfg, args.auto_gain)
        if args.switch_every > 0:
            signals = [perc.patch_to_signals(im) for im in imgs]
        else:
            signals = perc.patch_to_signals(img0)
        t_start = time.time()
        rec, snap_ticks, snaps, ceil_v, floor_v, layer_of = run_arm(
            perc, signals, args.ticks, args.beta_avg, args.switch_every)
        m = stabilization_metrics(rec["sat_mean"])
        arms[tag] = {"connectivity": p, "n_neurons": perc.num_neurons,
                     "ceil_med": float(np.median(ceil_v)), "floor": float(floor_v[0]),
                     "metrics": m, "rec": rec, "snap_ticks": snap_ticks,
                     "snaps": snaps, "layer_of": layer_of}
        print(f"[{tag}] conn={p:g} n={perc.num_neurons} ceil(med)={np.median(ceil_v):.0f} "
              f"| final sat {m['final_sat']:.3f} | tail slope {m['tail_slope_per_1000t']:+.4f}/kt "
              f"| tail std {m['tail_std']:.4f} | settle {m['settle_frac']*100:.0f}% "
              f"| part {rec['part'][-args.ticks//4:].mean():.3f} "
              f"| eff {rec['eff'][0]:.2f}->{rec['eff'][-1]:.2f} | {time.time()-t_start:.1f}s")

    _save_raw(args, arms)
    _plot_static(args, arms, regime)
    _animate(args, arms, regime)
    _save_summary(args, arms, conns, regime)
    return arms


def _save_raw(args, arms):
    payload = {}
    for tag, a in arms.items():
        for k, v in a["rec"].items():
            payload[f"{tag}__{k}"] = v
        payload[f"{tag}__snap_ticks"] = a["snap_ticks"]
        payload[f"{tag}__snaps"] = a["snaps"]
        payload[f"{tag}__layer_of"] = a["layer_of"]
    np.savez_compressed(os.path.join(args.output_dir, "tref_stability_raw.npz"),
                        **payload)


def _plot_static(args, arms, regime=""):
    cmap = plt.get_cmap("viridis")
    ps = [a["connectivity"] for a in arms.values()]
    norm = plt.Normalize(min(ps), max(ps))
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    w = max(50, args.ticks // 100)
    for tag, a in arms.items():
        col = cmap(norm(a["connectivity"]))
        r = a["rec"]
        lbl = f"conn={a['connectivity']:g}"
        ax[0, 0].plot(r["sat_mean"], color=col, lw=1.3, label=lbl)
        ax[0, 1].plot(a["snap_ticks"], r["eff"], color=col, lw=1.4, label=lbl)
        ax[1, 0].plot(r["part"], color=col, lw=0.8, alpha=0.85, label=lbl)
        rollstd = np.array([r["sat_mean"][i:i + w].std()
                            for i in range(len(r["sat_mean"]) - w)])
        ax[1, 1].plot(rollstd, color=col, lw=1.3, label=lbl)
    ax[0, 0].set_title("mean normalized t_ref saturation (0=floor/max-fire, 1=ceil/silent)")
    ax[0, 0].set_ylabel("mean sat"); ax[0, 0].set_ylim(-0.02, 1.02)
    ax[0, 1].set_title("mean synaptic efficacy (weight growth -- legacy rule)")
    ax[0, 1].set_ylabel("mean u_i.info")
    ax[1, 0].axhline(1.0 / 10, color="k", ls=":", lw=1, label="1/c firing cap")
    ax[1, 0].set_title("participation (spike fraction)"); ax[1, 0].set_ylabel("fraction firing")
    ax[1, 1].set_title(f"rolling std of mean sat (win={w}) -- stationarity")
    ax[1, 1].set_ylabel("rolling std")
    for a in ax.flat:
        a.set_xlabel("tick"); a.legend(fontsize=7)
    fig.suptitle(f"t_ref stabilization: sparse vs dense MLP under CONSTANT input "
                 f"({args.hidden} dense layers x {args.width}, {args.ticks} ticks, "
                 f"beta_avg={args.beta_avg or 0.999}, input={regime})", fontsize=12)
    fig.tight_layout()
    png = os.path.join(args.output_dir, "tref_stability_static.png")
    fig.savefig(png, dpi=120); plt.close(fig)
    print(f"Saved {png}")


def _animate(args, arms, regime=""):
    """Side-by-side: extreme-sparse vs dense, evolving per-neuron t_ref histogram."""
    tags = list(arms.keys())
    sparse_tag, dense_tag = tags[0], tags[-1]
    A, B = arms[sparse_tag], arms[dense_tag]
    nframes = min(len(A["snaps"]), len(B["snaps"]))
    bins = np.linspace(0, 1, 40)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.suptitle("Per-neuron t_ref saturation over time (0=floor, 1=ceil; constant input)")

    def draw(i):
        for k, (tag, a) in enumerate(((sparse_tag, A), (dense_tag, B))):
            ax[k].clear()
            ax[k].hist(a["snaps"][i], bins=bins, color="tab:blue" if k == 0 else "tab:red",
                       alpha=0.8)
            ax[k].axvline(a["snaps"][i].mean(), color="k", lw=2)
            ax[k].set_xlim(0, 1)
            ax[k].set_title(f"conn={a['connectivity']:g}  tick={int(a['snap_ticks'][i])}\n"
                            f"mean sat={a['snaps'][i].mean():.3f}")
            ax[k].set_xlabel("t_ref saturation")
        ax[0].set_ylabel("neuron count")

    anim = FuncAnimation(fig, draw, frames=nframes, interval=60)
    base = os.path.join(args.output_dir, "tref_stability_anim")
    try:
        anim.save(base + ".mp4", writer=FFMpegWriter(fps=20, bitrate=2400))
        print(f"Saved {base}.mp4")
    except Exception as e:
        print(f"ffmpeg unavailable ({e}); writing gif")
        anim.save(base + ".gif", writer=PillowWriter(fps=15))
        print(f"Saved {base}.gif")
    plt.close(fig)


def _save_summary(args, arms, conns, regime=""):
    summary = {"config": vars(args), "connectivities": conns, "regime": regime,
               "arms": {tag: {"connectivity": a["connectivity"],
                              "n_neurons": a["n_neurons"], "ceil_med": a["ceil_med"],
                              "floor": a["floor"], "metrics": a["metrics"],
                              "eff_start": float(a["rec"]["eff"][0]),
                              "eff_end": float(a["rec"]["eff"][-1]),
                              "part_tail": float(a["rec"]["part"][-len(a["rec"]["part"])//4:].mean())}
                        for tag, a in arms.items()}}
    js = os.path.join(args.output_dir, f"tref_stability_{int(time.time())}.json")
    json.dump(summary, open(js, "w"), indent=2)
    print(f"Saved {js}")
    # verdict line
    srt = sorted(arms.values(), key=lambda a: a["connectivity"])
    print("\nVERDICT (tail slope -> 0 = stabilized; larger |slope|/std = fails):")
    for a in srt:
        m = a["metrics"]
        print(f"  conn={a['connectivity']:g}: final_sat={m['final_sat']:.3f} "
              f"|slope|={abs(m['tail_slope_per_1000t']):.4f}/kt "
              f"tail_std={m['tail_std']:.4f} settle@{m['settle_frac']*100:.0f}%")


def main():
    p = argparse.ArgumentParser(description="sparse vs dense t_ref stabilization")
    p.add_argument("--dataset-name", default="cifar10_grayscale")
    p.add_argument("--connectivity", default="0.1,0.25,0.5,1.0")
    p.add_argument("--input-size", type=int, default=64)
    p.add_argument("--hidden", type=int, default=3)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--gain", type=float, default=3.0)
    p.add_argument("--auto-gain", type=float, default=1.8,
                   help="auto-calibrate input drive/threshold ratio to this target "
                        "(image-driven band ~1.0-2.5; 0 disables, uses --gain)")
    p.add_argument("--ticks", type=int, default=16000)
    p.add_argument("--img-index", type=int, default=0)
    p.add_argument("--switch-every", type=int, default=0,
                   help="0 = constant input; >0 = stream, switch image every N ticks")
    p.add_argument("--num-images", type=int, default=1,
                   help="number of distinct images to cycle when streaming")
    p.add_argument("--beta-avg", type=float, default=0.0,
                   help="0 = model default 0.999 (tau~1000); 0.995 -> tau~200 fast proxy")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="foveation_results/tref_stability")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
