"""Does a LEARNED multi-glimpse gaze beat a single glimpse (and random gaze)?

Frozen conv substrate + trained decoder. Three arms, online decoder acc over time:
  static  : 1 central glimpse (the single-glimpse baseline, ~0.22 gray)
  random  : n saccades, policy frozen at random (multi-glimpse, no learning where)
  learned : n saccades, REINFORCE gaze policy trained on reward (learn where to look)

If learned > random > static, active vision + learning-where-to-look pays off.
"""
from __future__ import annotations

import argparse, os, time, json, glob
import numpy as np

from snn_classification_realtime.activity_dataset_builder.vision_datasets import (
    load_dataset_by_name,
)
from snn_classification_realtime.foveation.minibrain.core import MiniBrainConfig, MiniBrain
from snn_classification_realtime.foveation.minibrain.active_vision import (
    GazePolicy, present_active,
)


def run_arm(cfg, make_ds, order, probe_every, mode, n_sacc, ticks, log, max_step=8.0):
    ds_cfg = make_ds()
    brain = MiniBrain(cfg, ds_cfg)
    ds = ds_cfg.dataset
    dim = brain.dim + 2                        # readout + gaze proprioception
    policy = GazePolicy(dim, seed=cfg.seed, max_step=max_step)
    learn_gaze = (mode == "learned")
    explore = mode in ("random", "learned")
    is_oracle = (mode == "oracle")            # fixate the KNOWN object location
    # static: 1 glimpse x `ticks`. staticlong/oracle: 1 glimpse x (n_sacc*ticks).
    if mode in ("static", "staticlong", "oracle"):
        ns = 1
        ticks = ticks * n_sacc if mode in ("staticlong", "oracle") else ticks
    else:
        ns = n_sacc
    traj_t, traj_oa = [], []
    correct = 0
    for i, idx in enumerate(order):
        img, y = ds[idx]; y = int(y)
        start_pos = ds.object_center(idx) if is_oracle else None
        pred, r, _ = present_active(brain, img, y, policy, n_saccades=ns,
                                    ticks_per_sacc=ticks, learn_gaze=learn_gaze,
                                    train_decoder=True, explore=explore, start_pos=start_pos)
        correct += int(pred == y)
        if (i + 1) % probe_every == 0:
            oa = correct / probe_every; correct = 0
            traj_t.append(i + 1); traj_oa.append(oa)
            log(f"    [{mode:7s}] {i+1:5d} online {oa:.3f} "
                f"|W_g| {np.linalg.norm(policy.W):.3f} baseline {policy.baseline:.3f}")
    return traj_t, traj_oa


def aggregate(args):
    """Load all per-(mode,seed) JSONs and plot mean +/- std per mode."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    pat = os.path.join(args.out, f"{args.dataset_name}_{args.readout}_*_seed*.json")
    runs = [json.load(open(f)) for f in glob.glob(pat)]
    if not runs:
        print(f"no result JSONs matching {pat}"); return
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    summary = {}
    for m, c in [("static", "tab:gray"), ("staticlong", "tab:brown"),
                 ("random", "tab:orange"), ("learned", "tab:green")]:
        rs = [r for r in runs if r["mode"] == m]
        if not rs:
            continue
        t = np.array(rs[0]["traj_t"])
        Y = np.array([r["traj_oa"] for r in rs if len(r["traj_oa"]) == len(t)])
        mu, sd = Y.mean(0), Y.std(0)
        ax.plot(t, mu, "-o", ms=3, color=c, label=f"{m} (n={len(Y)})")
        ax.fill_between(t, mu - sd, mu + sd, color=c, alpha=0.18)
        summary[m] = (float(mu[-1]), float(sd[-1]))
    ax.axhline(0.1, color="k", ls=":", lw=1, label="chance")
    ax.set_title(f"active vision: online decoder acc ({args.dataset_name}, readout {args.readout})")
    ax.set_xlabel("images seen"); ax.set_ylabel("online decoder acc"); ax.legend(fontsize=9)
    fig.tight_layout()
    png = os.path.join(args.out, f"{args.dataset_name}_{args.readout}_gaze.png")
    fig.savefig(png, dpi=120)
    print("FINAL (mean +/- std):  " +
          "  ".join(f"{m} {v[0]:.3f}±{v[1]:.3f}" for m, v in summary.items()))
    print(f"saved {png}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-name", default="cifar10_grayscale")
    p.add_argument("--images", type=int, default=1500)
    p.add_argument("--saccades", type=int, default=5)
    p.add_argument("--ticks-per-sacc", type=int, default=12)
    p.add_argument("--readout", default="all", choices=["input", "pool", "all"])
    p.add_argument("--canvas", type=int, default=0, help=">0 = cluttered canvas of this size (needs gaze)")
    p.add_argument("--distractors", type=int, default=4)
    p.add_argument("--distractor-size", type=int, default=16)
    p.add_argument("--fovea", type=int, default=8)
    p.add_argument("--grid", type=int, default=12)
    p.add_argument("--max-step", type=float, default=8.0)
    p.add_argument("--probe-every", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mode", default="all",
                   choices=["static", "staticlong", "oracle", "random", "learned", "all"])
    p.add_argument("--aggregate", action="store_true", help="just plot from existing JSONs")
    p.add_argument("--out", default="foveation_results/minibrain/gaze")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.aggregate:
        aggregate(args); return

    tag = f"{args.mode}_seed{args.seed}"
    lf = open(os.path.join(args.out, f"{args.dataset_name}_{tag}.log"), "a")
    def log(m): print(m, flush=True); lf.write(m + "\n"); lf.flush()

    if args.canvas > 0:
        from snn_classification_realtime.foveation.minibrain.cluttered import make_cluttered_cfg
        make_ds = lambda: make_cluttered_cfg(args.dataset_name, args.canvas,
                                             args.distractors, args.distractor_size, seed=999)
        periph = args.canvas          # periphery = wide low-res view of the WHOLE canvas
    else:
        make_ds = lambda: load_dataset_by_name(args.dataset_name, train=True)
        periph = 24
    ds = make_ds().dataset
    rng = np.random.RandomState(args.seed)
    order = rng.randint(0, len(ds), size=args.images)
    # unique net dir per process so parallel runs don't clobber the shared reservoir json
    net_dir = os.path.join(args.out, f"net_{args.mode}_seed{args.seed}_{os.getpid()}")
    cfg = MiniBrainConfig(dataset_name=args.dataset_name, retinal_conv=True,
                          tonic_drive=0.15, target_participation=0.05,
                          freeze_reservoir=True, readout=args.readout,
                          fovea=args.fovea, periph=periph, grid=args.grid,
                          dwell=args.ticks_per_sacc, seed=args.seed, output_dir=net_dir)
    modes = (["static", "staticlong", "random", "learned"]
             if args.mode == "all" else [args.mode])
    t0 = time.time()
    for mode in modes:
        log(f"=== GAZE {args.dataset_name} {mode} seed{args.seed} | {args.images} img "
            f"| {args.saccades}x{args.ticks_per_sacc} | readout {args.readout} ===")
        tt, oa = run_arm(cfg, make_ds, order, args.probe_every, mode,
                         args.saccades, args.ticks_per_sacc, log, max_step=args.max_step)
        out = os.path.join(args.out, f"{args.dataset_name}_{args.readout}_{mode}_seed{args.seed}.json")
        json.dump({"mode": mode, "seed": args.seed, "traj_t": tt, "traj_oa": oa,
                   "final": oa[-1] if oa else None}, open(out, "w"))
        log(f"[{time.time()-t0:.0f}s] {mode} seed{args.seed} final {oa[-1] if oa else float('nan'):.3f} -> {out}")


if __name__ == "__main__":
    main()
