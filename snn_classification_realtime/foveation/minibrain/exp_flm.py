"""Find-Lock-Memorize on the cluttered canvas: does a teacher-guided gaze learn to
FIND the object (dense distance reward) and lift accuracy from blind-chance (~0.10)
toward the oracle ceiling (~0.18)?

Arms (frozen substrate, decoder trained in all), online acc + mean final gaze-to-object
distance over time:
  random  : blind random-walk search + memorize (no gaze learning)   -> ~chance
  flm     : dense teacher find-reward, fixed search len, then memorize -> should FIND
  flmlock : flm + FE-stability lock (stop searching when readout settles)
"""
from __future__ import annotations

import argparse, os, time, json, glob
import numpy as np

from snn_classification_realtime.foveation.minibrain.core import MiniBrainConfig, MiniBrain
from snn_classification_realtime.foveation.minibrain.cluttered import make_cluttered_cfg
from snn_classification_realtime.foveation.minibrain.active_vision import (
    GazePolicy, present_flm, periph_dim,
)


def run_arm(cfg, make_ds, order, probe_every, mode, args, log):
    ds_cfg = make_ds(); brain = MiniBrain(cfg, ds_cfg); ds = ds_cfg.dataset
    pdim = (periph_dim(brain) if args.where == "periph" else brain.dim) + 2
    policy = GazePolicy(pdim, seed=cfg.seed, max_step=args.max_step)
    learn = (mode not in ("random", "oracle"))
    lock = args.lock_std if mode == "flmlock" else None
    nsacc = 0 if mode == "oracle" else args.saccades
    tt, oa_t, dist_t = [], [], []
    correct, dsum = 0, 0.0
    for i, idx in enumerate(order):
        img, y = ds[idx]; y = int(y)
        oc = ds.object_center(idx)
        pred, r, locked, fd = present_flm(
            brain, img, y, policy, oc, max_saccades=nsacc,
            ticks_per_sacc=args.ticks_per_sacc, memorize_ticks=args.memorize_ticks,
            lock_std=lock, canvas=args.canvas, where_signal=args.where,
            memorize_fovea_only=bool(args.memorize_fovea_only),
            learn_gaze=learn, train_decoder=True)
        correct += int(pred == y); dsum += fd
        if (i + 1) % probe_every == 0:
            oa = correct / probe_every; md = dsum / probe_every
            correct, dsum = 0, 0.0
            tt.append(i + 1); oa_t.append(oa); dist_t.append(md)
            log(f"    [{mode:8s}] {i+1:5d} online {oa:.3f} mean-dist {md:5.1f}px "
                f"|W_g| {np.linalg.norm(policy.W):.2f}")
    return tt, oa_t, dist_t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-name", default="cifar10_grayscale")
    p.add_argument("--images", type=int, default=1600)
    p.add_argument("--saccades", type=int, default=6)
    p.add_argument("--ticks-per-sacc", type=int, default=8)
    p.add_argument("--memorize-ticks", type=int, default=30)
    p.add_argument("--canvas", type=int, default=72)
    p.add_argument("--distractors", type=int, default=4)
    p.add_argument("--fovea", type=int, default=28)
    p.add_argument("--grid", type=int, default=14)
    p.add_argument("--max-step", type=float, default=18.0)
    p.add_argument("--lock-std", type=float, default=0.02)
    p.add_argument("--where", default="periph", choices=["periph", "readout"],
                   help="gaze policy input: retinotopic periphery (where) vs reservoir readout")
    p.add_argument("--readout", default="all", choices=["input", "pool", "all"])
    p.add_argument("--probe-every", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mode", default="flm", choices=["random", "flm", "flmlock", "oracle"])
    p.add_argument("--memorize-fovea-only", type=int, default=1)
    p.add_argument("--aggregate", action="store_true")
    p.add_argument("--out", default="foveation_results/minibrain/flm")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.aggregate:
        return _aggregate(args)

    lf = open(os.path.join(args.out, f"{args.mode}_seed{args.seed}.log"), "a")
    def log(m): print(m, flush=True); lf.write(m + "\n"); lf.flush()
    make_ds = lambda: make_cluttered_cfg(args.dataset_name, args.canvas,
                                         args.distractors, 16, seed=999)
    ds = make_ds().dataset
    order = np.random.RandomState(args.seed).randint(0, len(ds), size=args.images)
    net_dir = os.path.join(args.out, f"net_{args.mode}_seed{args.seed}_{os.getpid()}")
    cfg = MiniBrainConfig(dataset_name=args.dataset_name, retinal_conv=True,
                          tonic_drive=0.15, target_participation=0.05, freeze_reservoir=True,
                          readout=args.readout, fovea=args.fovea, periph=args.canvas,
                          grid=args.grid, dwell=args.ticks_per_sacc, seed=args.seed,
                          output_dir=net_dir)
    t0 = time.time()
    log(f"=== FLM {args.mode} seed{args.seed} | canvas {args.canvas} fovea {args.fovea} "
        f"| {args.saccades}sacc+{args.memorize_ticks}mem | {args.images} img ===")
    tt, oa, dist = run_arm(cfg, make_ds, order, args.probe_every, args.mode, args, log)
    out = os.path.join(args.out, f"{args.mode}_seed{args.seed}.json")
    json.dump({"mode": args.mode, "seed": args.seed, "traj_t": tt, "traj_oa": oa,
               "traj_dist": dist, "final": oa[-1] if oa else None,
               "final_dist": dist[-1] if dist else None}, open(out, "w"))
    log(f"[{time.time()-t0:.0f}s] {args.mode} final acc {oa[-1]:.3f} dist {dist[-1]:.1f}px")


def _aggregate(args):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    runs = [json.load(open(f)) for f in glob.glob(os.path.join(args.out, "*_seed*.json"))]
    if not runs: print("no runs"); return
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    summ = {}
    for m, c in [("oracle", "tab:purple"), ("random", "tab:orange"),
                 ("flm", "tab:green"), ("flmlock", "tab:blue")]:
        rs = [r for r in runs if r["mode"] == m]
        if not rs: continue
        t = np.array(rs[0]["traj_t"])
        A = np.array([r["traj_oa"] for r in rs if len(r["traj_oa"]) == len(t)])
        D = np.array([r["traj_dist"] for r in rs if len(r["traj_dist"]) == len(t)])
        ax[0].plot(t, A.mean(0), "-o", color=c, ms=3, label=f"{m} (n={len(A)})")
        ax[0].fill_between(t, A.mean(0)-A.std(0), A.mean(0)+A.std(0), color=c, alpha=0.18)
        ax[1].plot(t, D.mean(0), "-o", color=c, ms=3, label=m)
        summ[m] = (float(A.mean(0)[-1]), float(A.std(0)[-1]), float(D.mean(0)[-1]))
    ax[0].axhline(0.1, color="k", ls=":", lw=1, label="chance")
    ax[0].axhline(0.183, color="purple", ls="--", lw=1, label="oracle 0.18")
    ax[0].set_title("online decoder acc"); ax[0].set_xlabel("images"); ax[0].legend(fontsize=8)
    ax[1].set_title("mean final gaze->object distance (px)"); ax[1].set_xlabel("images")
    ax[1].legend(fontsize=8)
    fig.tight_layout(); png = os.path.join(args.out, "flm.png"); fig.savefig(png, dpi=120)
    print("FINAL:  " + "  ".join(f"{m} acc {v[0]:.3f}±{v[1]:.3f} dist {v[2]:.1f}px"
                                 for m, v in summ.items()))
    print(f"saved {png}")


if __name__ == "__main__":
    main()
