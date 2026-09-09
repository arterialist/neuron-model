"""Task B -- Adaptation & savings: does a pre-adapted net re-enter its dynamical
regime FASTER than a fresh net, given the same input? (measurement only)

USER QUESTION (verbatim intent): (1) do the dynamical properties of a network
constantly exposed to a certain input change on long timescales and stabilize /
reach equilibrium? (2) does a FRESH network enter the same dynamical regime given
the same input SLOWER than one that had time to adapt?

MECHANISTIC FRAME (verified, not assumed):
  * reset_simulation() zeros the FAST state (S, F_avg->0, t_ref->ceiling,
    M_vector, r/b) but PRESERVES the SLOW state -- synaptic weights u_i.info.
    So "adaptation" lives entirely in the weights.
  * t_ref is a passive readout of the firing-rate EMA F_avg (see
    experiment_tref_stability). F_avg relaxes toward the *current* firing rate at
    the EMA rate (tau ~ 1/(1-beta_avg)). But the firing rate itself depends on
    the weights, which are ALSO moving in a fresh net.
  * PREDICTION: cold net -> two coupled slow processes (weights grow AND F_avg
    chases the rising target) => equilibrium only after weights settle (>> tau).
    warm net -> weights already settled (stationary target) => F_avg settles at
    the pure EMA rate (~tau). Hence warm settles FASTER. This is a "savings"
    effect: the slow weights are a long-term memory of the regime.

DESIGN (per connectivity arm, fixed input image X, another image Y for control):
  COLD_X : fresh net driven by X for T ticks   (baseline speed; also ADAPTS net)
  WARM_X : same net, reset state (weights kept), driven by X for T ticks (savings)
  WARM_Y : same X-adapted net, reset, driven by Y  (is the savings X-SPECIFIC?)
  COLD_Y : a SECOND fresh net driven by Y         (baseline speed for Y)
Savings = COLD_X_settle - WARM_X_settle. If WARM_Y ~ COLD_Y the savings is
input-specific (a real memory of X); if WARM_Y is also fast it is generic
("bigger weights -> faster" regardless of content).

NO MODEL CHANGES. Pure defaults (legacy rule).
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
from snn_classification_realtime.foveation.perception import PerceptionNetwork
from snn_classification_realtime.foveation.experiment_tref_stability import (
    build_mlp_json, run_arm, autoscale_gain,
)


def settle_tick(series, equilibrium, band):
    """First tick after which |series - equilibrium| stays <= band forever."""
    series = np.asarray(series, np.float64)
    outside = np.where(np.abs(series - equilibrium) > band)[0]
    return int(outside[-1] + 1) if len(outside) else 0


def analyse(rec_cold, rec_warm, band=0.02):
    """Common equilibrium = warm tail (weights already settled there). Measure how
    long each trajectory takes to reach & stay within `band` of it."""
    eq = float(np.mean(rec_warm["sat_mean"][-len(rec_warm["sat_mean"]) // 5:]))
    T = len(rec_cold["sat_mean"])
    cold_s = settle_tick(rec_cold["sat_mean"], eq, band)
    warm_s = settle_tick(rec_warm["sat_mean"], eq, band)
    # weight settle (eff sampled at snap resolution) -- when does mean efficacy plateau
    return {"equilibrium_sat": eq, "cold_settle": cold_s, "warm_settle": warm_s,
            "savings_ticks": cold_s - warm_s,
            "savings_ratio": (cold_s / warm_s) if warm_s > 0 else float("inf"),
            "T": T,
            "eff_cold_start": float(rec_cold["eff"][0]),
            "eff_cold_end": float(rec_cold["eff"][-1]),
            "eff_warm_start": float(rec_warm["eff"][0]),
            "eff_warm_end": float(rec_warm["eff"][-1])}


def load_weights(perc, vec):
    """Assign u_i.info across all synapses in efficacy_vector() order. Lets us
    snapshot the slow weight-memory from an adapting net and inject it into a
    fresh probe net without perturbing the adapting trajectory."""
    i = 0
    for nid in perc._ids:
        nrn = perc.sim.network.neurons[nid]
        for sid in sorted(nrn.postsynaptic_points.keys()):
            nrn.postsynaptic_points[sid].u_i.info = float(vec[i]); i += 1


def run_compare(args):
    torch.manual_seed(0); np.random.seed(0)
    ds_cfg = load_dataset_by_name(args.dataset_name, train=True)
    ds_cfg.signal_gain = args.gain
    ds = ds_cfg.dataset
    imgX = ds[args.img_x][0]
    imgY = ds[args.img_y][0]
    C, H, W = imgX.shape
    os.makedirs(args.output_dir, exist_ok=True)
    conns = [float(x) for x in args.connectivity.split(",")]
    out = {}

    for p in conns:
        tag = f"c{p:g}"
        # --- net A: cold on X, then warm on X, then warm on Y (X-adapted) ---
        netA = os.path.join(args.output_dir, f"mlpA_{tag}.json")
        build_mlp_json(netA, channels=C, size=H, input_size=args.input_size,
                       hidden=args.hidden, sizes=args.width, connectivity=p,
                       seed=args.seed)
        percA = PerceptionNetwork(netA, ds_cfg)
        autoscale_gain(percA, ds_cfg, args.auto_gain)
        sigX = percA.patch_to_signals(imgX)
        sigY = percA.patch_to_signals(imgY)
        t0 = time.time()
        rec_coldX, *_ = run_arm(percA, sigX, args.ticks, args.beta_avg)   # cold + adapt
        # optional extra adaptation so weights are fully settled before warm probes
        if args.adapt_extra > 0:
            run_arm_keep(percA, sigX, args.adapt_extra, args.beta_avg)
        rec_warmX, *_ = run_arm(percA, sigX, args.ticks, args.beta_avg)    # warm, same input
        rec_warmY, *_ = run_arm(percA, sigY, args.ticks, args.beta_avg)    # warm, novel input

        # --- net B: cold on Y (baseline for Y) ---
        netB = os.path.join(args.output_dir, f"mlpB_{tag}.json")
        build_mlp_json(netB, channels=C, size=H, input_size=args.input_size,
                       hidden=args.hidden, sizes=args.width, connectivity=p,
                       seed=args.seed)
        percB = PerceptionNetwork(netB, ds_cfg)
        rec_coldY, *_ = run_arm(percB, percB.patch_to_signals(imgY),
                                args.ticks, args.beta_avg)

        aX = analyse(rec_coldX, rec_warmX, args.band)
        aY = analyse(rec_coldY, rec_warmY, args.band)
        out[tag] = {"connectivity": p, "savings_X": aX, "specificity_Y": aY,
                    "_rec": {"coldX": rec_coldX, "warmX": rec_warmX,
                             "warmY": rec_warmY, "coldY": rec_coldY}}
        print(f"[{tag}] conn={p:g} | COLD_X settle {aX['cold_settle']} -> WARM_X settle "
              f"{aX['warm_settle']}  (savings {aX['savings_ticks']:+d}t, "
              f"x{aX['savings_ratio']:.2f}) | eff X {aX['eff_cold_start']:.2f}->"
              f"{aX['eff_cold_end']:.2f} | COLD_Y {aY['cold_settle']} vs WARM_Y "
              f"{aY['warm_settle']} | {time.time()-t0:.1f}s")

    _plot_compare(args, out)
    _save_compare(args, out, conns)
    return out


def run_formation(args):
    """HEADLINE (Task B): how many ticks of exposure are needed to form a
    'meaningful memory'? Sweep the ADAPTATION duration T_adapt; for each, snapshot
    the weights and measure (a) how much they have grown (eff), and (b) how fast a
    reset net loaded with those weights re-enters equilibrium (warm settle). The
    knee of warm_settle(T_adapt) -- equivalently of eff(T_adapt) -- IS the
    memory-formation timescale in ticks."""
    torch.manual_seed(0); np.random.seed(0)
    ds_cfg = load_dataset_by_name(args.dataset_name, train=True)
    ds_cfg.signal_gain = args.gain
    ds = ds_cfg.dataset
    imgX = ds[args.img_x][0]
    C, H, W = imgX.shape
    os.makedirs(args.output_dir, exist_ok=True)
    p = float(args.connectivity.split(",")[0])

    cps = sorted(set(int(x) for x in args.adapt_ticks.split(",")))
    n_seeds = max(1, args.form_seeds)
    # The tick dynamics are DETERMINISTIC (no per-tick RNG) but sensitively
    # dependent on the weight configuration, so a single settle-time trace is
    # jumpy. We average warm_settle over n_seeds independent nets to get a smooth
    # functional curve; eff(T_adapt) (the weight-memory magnitude) is already
    # smooth and is the primary, deterministic answer.
    t0 = time.time()
    ws_by_seed = np.zeros((n_seeds, len(cps)))
    eff_by_seed = np.zeros((n_seeds, len(cps)))
    for si in range(n_seeds):
        seed = args.seed + si
        net = os.path.join(args.output_dir, f"mlp_form_s{si}.json")
        build_mlp_json(net, channels=C, size=H, input_size=args.input_size,
                       hidden=args.hidden, sizes=args.width, connectivity=p, seed=seed)
        adapt = PerceptionNetwork(net, ds_cfg)
        if si == 0:
            autoscale_gain(adapt, ds_cfg, args.auto_gain)
        sigX = adapt.patch_to_signals(imgX)
        if args.beta_avg > 0:
            for n in adapt.sim.network.neurons.values():
                n.params.beta_avg = float(args.beta_avg)
        snaps, t = {}, 0
        for cp in cps:
            while t < cp:
                adapt.step(sigX); t += 1
            snaps[cp] = adapt.efficacy_vector().copy()
            eff_by_seed[si, cps.index(cp)] = float(snaps[cp].mean())
        probe = PerceptionNetwork(net, ds_cfg)
        sigXp = probe.patch_to_signals(imgX)
        traces = []
        for cp in cps:
            load_weights(probe, snaps[cp])
            rec, *_ = run_arm(probe, sigXp, args.probe_ticks, args.beta_avg)
            traces.append(rec["sat_mean"].copy())
        ref = float(np.mean(traces[-1][-args.probe_ticks // 5:]))
        for k, tr in enumerate(traces):
            ws_by_seed[si, k] = settle_tick(tr, ref, args.band)
    warm = ws_by_seed.mean(axis=0)
    warm_sd = ws_by_seed.std(axis=0)
    effm = eff_by_seed.mean(axis=0)
    rows = [{"T_adapt": cps[k], "eff": float(effm[k]),
             "warm_settle": float(warm[k]), "warm_settle_sd": float(warm_sd[k])}
            for k in range(len(cps))]
    for r in rows:
        print(f"  T_adapt={r['T_adapt']:>6} | eff={r['eff']:.3f} "
              f"| warm_settle={r['warm_settle']:>7.0f} +-{r['warm_settle_sd']:.0f} "
              f"(n={n_seeds})")
    cold = rows[0]["warm_settle"]
    floor = min(r["warm_settle"] for r in rows)
    # memory-formation tick: smallest T_adapt whose warm_settle drops halfway
    # from the cold baseline to the fast floor
    half = cold - 0.5 * (cold - floor)
    formed = next((r["T_adapt"] for r in rows if r["warm_settle"] <= half), cps[-1])
    print(f"\nMEMORY-FORMATION TICK (warm_settle halfway to floor): ~{formed} ticks "
          f"[cold {cold}t -> floor {floor}t]  ({time.time()-t0:.1f}s)")
    _plot_formation(args, rows, p, formed, cold, floor)
    js = os.path.join(args.output_dir, f"formation_{int(time.time())}.json")
    json.dump({"config": vars(args), "connectivity": p,
               "formation_tick": formed, "cold_settle": cold, "floor_settle": floor,
               "rows": [{k: v for k, v in r.items() if k != "sat_trace"} for r in rows]},
              open(js, "w"), indent=2)
    np.savez_compressed(os.path.join(args.output_dir, "formation_raw.npz"),
                        T_adapt=np.array([r["T_adapt"] for r in rows]),
                        eff=np.array([r["eff"] for r in rows]),
                        warm_settle=np.array([r["warm_settle"] for r in rows]),
                        warm_settle_sd=np.array([r["warm_settle_sd"] for r in rows]))
    print(f"Saved {js}")
    return rows


def run_timescale(args):
    """SECONDARY (what makes the effect strong): savings is governed by the RATIO
    of two timescales -- tau_W (weight growth, set by eta_post) and tau_F (F_avg
    EMA, set by beta_avg). Predict savings_ratio ~ max(1, tau_W/tau_F). Sweep both
    knobs, measure empirical tau_W (from eff) and savings, test the collapse."""
    torch.manual_seed(0); np.random.seed(0)
    ds_cfg = load_dataset_by_name(args.dataset_name, train=True)
    ds_cfg.signal_gain = args.gain
    ds = ds_cfg.dataset
    imgX = ds[args.img_x][0]
    C, H, W = imgX.shape
    os.makedirs(args.output_dir, exist_ok=True)
    p = float(args.connectivity.split(",")[0])
    betas = [float(x) for x in args.betas.split(",")]
    etas = [float(x) for x in args.etas.split(",")]  # multipliers of default eta_post

    net = os.path.join(args.output_dir, "mlp_ts.json")
    build_mlp_json(net, channels=C, size=H, input_size=args.input_size,
                   hidden=args.hidden, sizes=args.width, connectivity=p, seed=args.seed)
    autoscale_gain(PerceptionNetwork(net, ds_cfg), ds_cfg, args.auto_gain)
    grid = []
    t0 = time.time()
    for beta in betas:
        tau_F = 1.0 / (1.0 - beta)
        for em in etas:
            perc = PerceptionNetwork(net, ds_cfg)
            for n in perc.sim.network.neurons.values():
                n.params.beta_avg = beta
                n.params.eta_post = 0.01 * em
            sig = perc.patch_to_signals(imgX)
            rec_cold, *_ = run_arm(perc, sig, args.ticks, 0.0)  # beta already set
            rec_warm, *_ = run_arm(perc, sig, args.ticks, 0.0)
            a = analyse(rec_cold, rec_warm, args.band)
            # empirical tau_W: ticks for eff to cover (1-1/e) of its total rise
            eff = rec_cold["eff"]
            rise = eff[-1] - eff[0]
            tau_W = args.ticks
            if rise > 1e-6:
                thr = eff[0] + (1 - 1 / np.e) * rise
                snap_dt = max(1, args.ticks // len(eff))
                idx = np.argmax(eff >= thr)
                tau_W = float(idx * snap_dt)
            grid.append({"beta": beta, "eta_mult": em, "tau_F": tau_F, "tau_W": tau_W,
                         "cold_settle": a["cold_settle"], "warm_settle": a["warm_settle"],
                         "savings_ratio": a["savings_ratio"]})
            print(f"  beta={beta:.3f}(tauF={tau_F:.0f}) eta_x{em:g} | tau_W~{tau_W:.0f} "
                  f"| cold {a['cold_settle']}t warm {a['warm_settle']}t "
                  f"| savings x{a['savings_ratio']:.2f} | ratio tauW/tauF={tau_W/tau_F:.2f}")
    _plot_timescale(args, grid, betas, etas)
    js = os.path.join(args.output_dir, f"timescale_{int(time.time())}.json")
    json.dump({"config": vars(args), "connectivity": p, "grid": grid},
              open(js, "w"), indent=2)
    print(f"Saved {js} ({time.time()-t0:.1f}s)")
    return grid


def _plot_formation(args, rows, p, formed, cold, floor):
    Ta = np.array([r["T_adapt"] for r in rows])
    ws = np.array([r["warm_settle"] for r in rows])
    wsd = np.array([r.get("warm_settle_sd", 0.0) for r in rows])
    ef = np.array([r["eff"] for r in rows])
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
    ax[0].errorbar(Ta, ws, yerr=wsd, fmt="o-", color="tab:green", lw=1.8,
                   capsize=3, ecolor="tab:green", elinewidth=0.8)
    ax[0].axhline(floor, color="gray", ls="--", lw=1, label=f"fast floor ~{floor}t")
    ax[0].axhline(cold, color="tab:red", ls=":", lw=1, label=f"cold baseline {cold}t")
    ax[0].axvline(formed, color="k", ls="-.", lw=1.4, label=f"memory formed ~{formed}t")
    ax[0].set_xscale("symlog"); ax[0].set_xlabel("adaptation duration T_adapt (ticks)")
    ax[0].set_ylabel("warm re-entry settle time (ticks)")
    ax[0].set_title("How fast a pre-adapted net re-enters equilibrium\nvs how long it was exposed")
    ax[0].legend(fontsize=8)
    axb = ax[1]; axb.plot(Ta, ef, "o-", color="tab:purple", lw=1.8)
    axb.axvline(formed, color="k", ls="-.", lw=1.4)
    axb.set_xscale("symlog"); axb.set_xlabel("adaptation duration T_adapt (ticks)")
    axb.set_ylabel("mean synaptic efficacy at reset (weight memory)")
    axb.set_title("Weight-memory growth vs exposure\n(the physical substrate of the savings)")
    fig.suptitle(f"Task B -- memory formation: conn={p:g}, gain={args.gain}, "
                 f"beta_avg={args.beta_avg or 0.999}", fontsize=12)
    fig.tight_layout()
    png = os.path.join(args.output_dir, "formation.png")
    fig.savefig(png, dpi=120); plt.close(fig)
    print(f"Saved {png}")


def _plot_timescale(args, grid, betas, etas):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    # collapse: savings_ratio vs tau_W/tau_F
    x = np.array([g["tau_W"] / g["tau_F"] for g in grid])
    y = np.array([g["savings_ratio"] for g in grid])
    ax[0].loglog(x, y, "o", color="tab:blue", ms=7)
    lo, hi = max(1e-2, x.min()), max(x.max(), 1.0)
    xs = np.linspace(lo, hi, 50)
    ax[0].plot(xs, np.maximum(1.0, xs), "k--", lw=1, label="predicted max(1, tauW/tauF)")
    ax[0].set_xlabel("tau_W / tau_F  (weight-growth vs F_avg-EMA time)")
    ax[0].set_ylabel("measured savings ratio (cold/warm settle)")
    ax[0].set_title("Savings collapses onto the timescale ratio")
    ax[0].legend(fontsize=8)
    # heatmap savings over (beta, eta)
    S = np.full((len(betas), len(etas)), np.nan)
    bi = {b: i for i, b in enumerate(betas)}; ei = {e: j for j, e in enumerate(etas)}
    for g in grid:
        S[bi[g["beta"]], ei[g["eta_mult"]]] = g["savings_ratio"]
    im = ax[1].imshow(S, origin="lower", aspect="auto", cmap="viridis")
    ax[1].set_xticks(range(len(etas))); ax[1].set_xticklabels([f"x{e:g}" for e in etas])
    ax[1].set_yticks(range(len(betas))); ax[1].set_yticklabels([f"{b:g}" for b in betas])
    ax[1].set_xlabel("eta_post multiplier (smaller tau_W ->)")
    ax[1].set_ylabel("beta_avg (larger tau_F ^)")
    ax[1].set_title("savings ratio(beta, eta)")
    for i in range(len(betas)):
        for j in range(len(etas)):
            if not np.isnan(S[i, j]):
                ax[1].text(j, i, f"{S[i, j]:.1f}", ha="center", va="center",
                           color="w", fontsize=8)
    fig.colorbar(im, ax=ax[1], label="savings ratio")
    fig.suptitle(f"What timescale makes savings strong (conn={float(args.connectivity.split(',')[0]):g}, "
                 f"gain={args.gain})", fontsize=12)
    fig.tight_layout()
    png = os.path.join(args.output_dir, "timescale.png")
    fig.savefig(png, dpi=120); plt.close(fig)
    print(f"Saved {png}")


def run_arm_keep(perc, signal, T, beta_avg):
    """Advance T ticks WITHOUT the leading reset (pure extra adaptation).
    We still preserve weights; we just keep integrating from current state."""
    if beta_avg > 0:
        for n in perc.sim.network.neurons.values():
            n.params.beta_avg = float(beta_avg)
    for _ in range(T):
        perc.step(signal)


def _plot_compare(args, out):
    ncol = len(out)
    fig, ax = plt.subplots(2, ncol, figsize=(5.2 * ncol, 8), squeeze=False)
    for j, (tag, a) in enumerate(out.items()):
        r = a["_rec"]
        eq = a["savings_X"]["equilibrium_sat"]
        ax[0, j].plot(r["coldX"]["sat_mean"], color="tab:red", lw=1.4, label="COLD_X (fresh)")
        ax[0, j].plot(r["warmX"]["sat_mean"], color="tab:green", lw=1.4, label="WARM_X (adapted)")
        ax[0, j].axhline(eq, color="k", ls=":", lw=1, label="equilibrium")
        ax[0, j].axhline(eq + args.band, color="gray", ls="--", lw=0.6)
        ax[0, j].axhline(eq - args.band, color="gray", ls="--", lw=0.6)
        ax[0, j].axvline(a["savings_X"]["cold_settle"], color="tab:red", ls=":", lw=1)
        ax[0, j].axvline(a["savings_X"]["warm_settle"], color="tab:green", ls=":", lw=1)
        ax[0, j].set_title(f"conn={a['connectivity']:g}  savings x{a['savings_X']['savings_ratio']:.2f}\n"
                           f"cold {a['savings_X']['cold_settle']}t -> warm {a['savings_X']['warm_settle']}t")
        ax[0, j].set_ylabel("mean t_ref saturation"); ax[0, j].set_xlabel("tick")
        ax[0, j].legend(fontsize=7)
        # specificity: cold_Y vs warm_Y (X-adapted on novel Y)
        ax[1, j].plot(r["coldY"]["sat_mean"], color="tab:blue", lw=1.4, label="COLD_Y (fresh)")
        ax[1, j].plot(r["warmY"]["sat_mean"], color="tab:orange", lw=1.4, label="WARM_Y (X-adapted)")
        ax[1, j].set_title(f"specificity: novel input Y\ncold_Y {a['specificity_Y']['cold_settle']}t "
                           f"vs warm_Y {a['specificity_Y']['warm_settle']}t")
        ax[1, j].set_ylabel("mean t_ref saturation"); ax[1, j].set_xlabel("tick")
        ax[1, j].legend(fontsize=7)
    fig.suptitle(f"Task B: savings -- fresh vs pre-adapted net entering the same regime "
                 f"(gain={args.gain}, {args.hidden}x{args.width}, beta_avg={args.beta_avg or 0.999})",
                 fontsize=12)
    fig.tight_layout()
    png = os.path.join(args.output_dir, "savings.png")
    fig.savefig(png, dpi=120); plt.close(fig)
    print(f"Saved {png}")


def _save_compare(args, out, conns):
    payload = {}
    for tag, a in out.items():
        for run_name, r in a["_rec"].items():
            for k in ("sat_mean", "eff", "part", "favg_mean"):
                payload[f"{tag}__{run_name}__{k}"] = r[k]
    np.savez_compressed(os.path.join(args.output_dir, "savings_raw.npz"), **payload)
    summary = {"config": vars(args),
               "arms": {tag: {"connectivity": a["connectivity"],
                              "savings_X": a["savings_X"],
                              "specificity_Y": a["specificity_Y"]}
                        for tag, a in out.items()}}
    js = os.path.join(args.output_dir, f"savings_{int(time.time())}.json")
    json.dump(summary, open(js, "w"), indent=2)
    print(f"Saved {js}")
    print("\nVERDICT (savings_ratio > 1 => pre-adapted net re-enters regime FASTER):")
    for tag, a in out.items():
        sx, sy = a["savings_X"], a["specificity_Y"]
        print(f"  conn={a['connectivity']:g}: X savings x{sx['savings_ratio']:.2f} "
              f"({sx['cold_settle']}->{sx['warm_settle']}t) | "
              f"Y: cold {sy['cold_settle']}t vs warm {sy['warm_settle']}t "
              f"(X-adaptation {'helps Y too (generic)' if sy['warm_settle'] < 0.8*sy['cold_settle'] else 'X-specific'})")


def main():
    p = argparse.ArgumentParser(description="Task B: adaptation & savings")
    p.add_argument("--mode", default="formation",
                   choices=["formation", "compare", "timescale"],
                   help="formation=memory-formation sweep (headline); "
                        "compare=cold-vs-warm + specificity; "
                        "timescale=beta/eta ratio sweep")
    p.add_argument("--dataset-name", default="cifar10_grayscale")
    p.add_argument("--connectivity", default="0.1,0.25")
    p.add_argument("--input-size", type=int, default=64)
    p.add_argument("--hidden", type=int, default=3)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--gain", type=float, default=1.0)
    p.add_argument("--auto-gain", type=float, default=1.8,
                   help="auto-calibrate input drive/threshold ratio to this target "
                        "(image-driven band ~1.0-2.5; 0 disables)")
    p.add_argument("--ticks", type=int, default=8000)
    p.add_argument("--adapt-extra", type=int, default=0,
                   help="extra adaptation ticks (no reset) between cold and warm [compare]")
    p.add_argument("--band", type=float, default=0.02,
                   help="absolute sat tolerance defining 'in equilibrium'")
    p.add_argument("--img-x", type=int, default=0)
    p.add_argument("--img-y", type=int, default=7)
    p.add_argument("--beta-avg", type=float, default=0.0,
                   help="0=model default 0.999 (tau~1000); 0.995->tau~200 fast proxy")
    # formation mode
    p.add_argument("--adapt-ticks", default="0,50,100,250,500,1000,2000,4000,8000,16000",
                   help="[formation] adaptation durations to sweep")
    p.add_argument("--probe-ticks", type=int, default=4000,
                   help="[formation] ticks per warm re-entry probe")
    p.add_argument("--form-seeds", type=int, default=3,
                   help="[formation] independent nets to average warm_settle over")
    # timescale mode
    p.add_argument("--betas", default="0.99,0.995,0.998,0.999",
                   help="[timescale] beta_avg values (set tau_F)")
    p.add_argument("--etas", default="0.5,1,2,4",
                   help="[timescale] eta_post multipliers (set tau_W)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="foveation_results/savings")
    args = p.parse_args()
    if args.mode == "formation":
        run_formation(args)
    elif args.mode == "timescale":
        run_timescale(args)
    else:
        run_compare(args)


if __name__ == "__main__":
    main()
