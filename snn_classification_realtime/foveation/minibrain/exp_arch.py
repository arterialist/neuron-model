"""Unified architecture-sweep driver for the three proposals (2026-07-07).

One (arm, seed) per process -> JSON + static plot (+ optional animation). Aggregate
mode combines arms into comparison figures. See FOVEATION_RESEARCH.md §SESSION
2026-07-07 for the design and the MNIST self-correction that motivates it.

Locked params: dwell=500 (memory forms ~500t), warmup=500 per fresh net, 400 img/arm,
gray AND color. Each fresh net is warmed to its attractor before ANY measurement.

Experiments:
  sep   -- Exp1: frozen-encoder separability. arch{reservoir_random,reservoir_retino,
           conv} x conv_bank{default,rich} x readout_head{linear,mlp} x dataset. Does
           conv/retinotopic topology + a trained MLP readout recover beyond-linear?
  plast -- Exp2: substrate-plasticity trajectory. {frozen, reward_hebb(+/-teacher),
           ext oja/bcm/rmhebb, assoc} x dataset (retinotopic). Does reward-gated
           plasticity ACCUMULATE structure in the right topology? Long run, probed
           every `probe_every`; includes a fresh-net counterfactual.
  gaze  -- Exp3: cluttered find-lock-memorize, policy{linear,mlp,recurrent} + oracle/random.
"""
from __future__ import annotations

import argparse, os, json, time, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from snn_classification_realtime.activity_dataset_builder.vision_datasets import load_dataset_by_name
from snn_classification_realtime.foveation.minibrain.core import MiniBrainConfig, MiniBrain

CIFAR10 = ["plane", "auto", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
_Cs = (0.0005, 0.002, 0.01, 0.05)


# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #
def best_linear(X, y, split=0.7):
    X = np.asarray(X, float); y = np.asarray(y); n = len(y); ntr = int(n * split)
    mu, sd = X[:ntr].mean(0), X[:ntr].std(0) + 1e-6; Xn = (X - mu) / sd
    return max(LogisticRegression(max_iter=1500, C=C).fit(Xn[:ntr], y[:ntr])
               .score(Xn[ntr:], y[ntr:]) for C in _Cs)


def mlp_probe(X, y, hidden, seed, split=0.7):
    from snn_classification_realtime.foveation.minibrain.heads import TorchMLPHead
    X = np.asarray(X, float); y = np.asarray(y); n = len(y); ntr = int(n * split)
    head = TorchMLPHead(X.shape[1], 10, hidden=hidden, seed=seed)
    head.fit(X[:ntr], y[:ntr]); return head.score(X[ntr:], y[ntr:])


def make_cfg(args, out_dir):
    """Map the arm spec to a MiniBrainConfig (defaults preserved except the locked
    dwell + the arm's opted-in flags)."""
    arch = args.arch
    substrate = "conv" if arch == "conv" else "reservoir"
    wiring = "retinotopic" if arch == "reservoir_retino" else "random"
    return MiniBrainConfig(
        dataset_name=args.dataset, dwell=args.dwell, seed=args.seed,
        substrate_type=substrate, wiring=wiring, conv_bank=args.conv_bank,
        readout=args.readout, freeze_reservoir=not args.unfreeze,
        plasticity_mode=args.plasticity_mode, nm_kappa=args.nm_kappa, rh_decay=args.rh_decay,
        eta_post_res=(args.eta_post_res if args.eta_post_res >= 0 else None),
        ext_plasticity=args.ext_plasticity, ext_plast_eta=args.ext_plast_eta,
        assoc_layer=args.assoc, assoc_dim=args.assoc_dim,
        teacher_mode=args.teacher_mode, reward_gain=args.reward_gain, stress_gain=args.stress_gain,
        target_participation=args.participation,
        output_dir=out_dir)


def warmup(brain, ds, order, ticks):
    """Run a fresh net to its attractor (learning OFF -- settle DYNAMICS, not weights)
    before any measurement, streaming warmup images with tonic on."""
    if ticks <= 0:
        return
    per = max(1, brain.cfg.input_period); done = 0; i = 0
    brain.sub.set_learning(False)
    while done < ticks:
        img, _ = ds[int(order[i % len(order)])]
        sig = brain.sub.patch_to_signals(brain._encode(img))
        for t in range(brain.cfg.dwell):
            brain.sub.step((sig if t % per == 0 else []) + brain._tonic); done += 1
            if done >= ticks:
                break
        i += 1
    brain.sub.set_learning(True)


def buffer_reps(brain, ds, idxs, dwell):
    """Frozen-probe: encode idxs, read the raw substrate rep (NOT the assoc layer).
    Uses `dwell` ticks/image for the probe settle -- SHORTER than the run's 500-tick
    memory-formation fixation, since the probe measures single-glimpse separability of
    the CURRENT substrate, not cross-image memory (which lives in the run itself)."""
    brain.sub.set_learning(False)
    m = brain._readout_mask; per = max(1, brain.cfg.input_period)
    X, Y = [], []
    for i in idxs:
        img, y = ds[int(i)]; Y.append(int(y))
        sig = brain.sub.patch_to_signals(brain._encode(img))
        st = [brain.sub.step((sig if t % per == 0 else []) + brain._tonic)
              for t in range(dwell)]
        X.append(np.concatenate([np.mean([s.S[m] for s in st], 0),
                                 np.mean([s.F_avg[m] for s in st], 0),
                                 np.mean([s.O[m] for s in st], 0)]))
    brain.sub.set_learning(True)
    return np.array(X), np.array(Y)


# --------------------------------------------------------------------------- #
#  Exp1: frozen-encoder separability
# --------------------------------------------------------------------------- #
def run_sep(args, log):
    ds = load_dataset_by_name(args.dataset, train=True).dataset
    order = np.random.RandomState(args.seed).randint(0, len(ds), size=args.images + 400)
    cfg = make_cfg(args, os.path.join(args.out, f"net_{args.tag}_{os.getpid()}"))
    t0 = time.time(); brain = MiniBrain(cfg, load_dataset_by_name(args.dataset, train=True))
    warmup(brain, ds, order[:50], args.warmup)
    X, Y = buffer_reps(brain, ds, order[:args.images], args.probe_dwell)
    lin = best_linear(X, Y)
    mlp = mlp_probe(X, Y, args.mlp_hidden, args.seed) if args.readout_head == "mlp" else None
    out = dict(exp="sep", tag=args.tag, arch=args.arch, conv_bank=args.conv_bank,
               readout_head=args.readout_head, dataset=args.dataset, seed=args.seed,
               dim=int(X.shape[1]), n_neurons=int(brain.n_neurons),
               lin=float(lin), mlp=(float(mlp) if mlp is not None else None), complete=True)
    json.dump(out, open(os.path.join(args.out, f"{args.tag}_seed{args.seed}.json"), "w"))
    log(f"[{time.time()-t0:.0f}s] {args.tag} lin {lin:.3f}" +
        (f" mlp {mlp:.3f}" if mlp is not None else ""))
    if args.animate:
        _animate(brain, ds, order[args.images:args.images + args.anim_images], args, log)


# --------------------------------------------------------------------------- #
#  Exp2: substrate-plasticity trajectory
# --------------------------------------------------------------------------- #
def run_plast(args, log):
    ds = load_dataset_by_name(args.dataset, train=True).dataset
    order = np.random.RandomState(args.seed).randint(0, len(ds), size=args.images)
    probe_idx = np.random.RandomState(args.seed + 7).randint(0, len(ds), size=args.probe_n)
    cfg = make_cfg(args, os.path.join(args.out, f"net_{args.tag}_{os.getpid()}"))
    t0 = time.time(); brain = MiniBrain(cfg, load_dataset_by_name(args.dataset, train=True))
    warmup(brain, ds, order[:50], args.warmup)
    learn = args.unfreeze or args.ext_plasticity != "none" or args.assoc  # substrate/assoc plasticity on?
    tt, sep, oa = [], [], []
    fresh_sep = []
    base = best_linear(*buffer_reps(brain, ds, probe_idx, args.probe_dwell))
    tt.append(0); sep.append(base); oa.append(0.0); fresh_sep.append(base)
    log(f"    [{args.tag}] t=0 sep {base:.3f}")
    jpath = os.path.join(args.out, f"{args.tag}_seed{args.seed}.json")
    def snapshot(complete):
        return dict(exp="plast", tag=args.tag, arch=args.arch, dataset=args.dataset, seed=args.seed,
                    plasticity_mode=args.plasticity_mode, ext_plasticity=args.ext_plasticity,
                    assoc=bool(args.assoc), teach=bool(args.teach), unfreeze=bool(args.unfreeze),
                    traj_t=tt, traj_sep=sep, traj_oa=oa, fresh_sep=fresh_sep,
                    final_sep=sep[-1], final_oa=oa[-1] if oa else None, complete=complete)
    correct = 0
    for i, idx in enumerate(order):
        img, y = ds[int(idx)]; y = int(y)
        _, pred = brain.present(img, y, learn=learn, teach=args.teach, train_decoder=True)
        correct += int(pred == y)
        if (i + 1) % args.probe_every == 0:
            s = best_linear(*buffer_reps(brain, ds, probe_idx, args.probe_dwell))
            tt.append(i + 1); sep.append(s); oa.append(correct / args.probe_every); correct = 0
            # fresh-net counterfactual (a brand-new same-cfg net, warmed) is expensive
            # -- a full rebuild -- so only compute it at the FINAL probe: it answers
            # "did adaptation, not warmup, cause any gain vs an unadapted net?"
            if i + 1 >= len(order):
                fb = MiniBrain(cfg, load_dataset_by_name(args.dataset, train=True))
                warmup(fb, ds, order[:50], args.warmup)
                fresh_sep.append(best_linear(*buffer_reps(fb, ds, probe_idx, args.probe_dwell)))
            else:
                fresh_sep.append(np.nan)
            json.dump(snapshot(False), open(jpath, "w"))   # checkpoint: survive a kill
            log(f"    [{args.tag}] {i+1:4d} sep {s:.3f} onlineAcc {oa[-1]:.3f} fresh {fresh_sep[-1]:.3f}")
    out = snapshot(True)
    json.dump(out, open(jpath, "w"))
    log(f"[{time.time()-t0:.0f}s] {args.tag} final sep {sep[-1]:.3f} onlineAcc {oa[-1]:.3f}")
    _plot_traj(out, os.path.join(args.out, f"{args.tag}_seed{args.seed}.png"))
    if args.animate:
        _animate(brain, ds, order[:args.anim_images], args, log)


# --------------------------------------------------------------------------- #
#  Exp3: gaze (cluttered find-lock-memorize)
# --------------------------------------------------------------------------- #
def run_gaze(args, log):
    from snn_classification_realtime.foveation.minibrain.cluttered import make_cluttered_cfg
    from snn_classification_realtime.foveation.minibrain.active_vision import (
        GazePolicy, MLPGazePolicy, RecurrentGazePolicy, present_flm, periph_dim)
    make_ds = lambda: make_cluttered_cfg(args.dataset, args.canvas, args.distractors, 16, seed=999)
    ds = make_ds().dataset
    order = np.random.RandomState(args.seed).randint(0, len(ds), size=args.images)
    cfg = make_cfg(args, os.path.join(args.out, f"net_{args.tag}_{os.getpid()}"))
    cfg.fovea = args.fovea; cfg.periph = args.canvas; cfg.grid = args.grid
    t0 = time.time(); brain = MiniBrain(cfg, make_ds())
    warmup(brain, ds, order[:50], args.warmup)
    pdim = (periph_dim(brain) if args.where == "periph" else brain.dim) + 2
    Pol = {"linear": GazePolicy, "mlp": MLPGazePolicy, "recurrent": RecurrentGazePolicy}[args.policy]
    policy = Pol(pdim, seed=args.seed, max_step=args.max_step)
    mode = args.gaze_mode                        # flm / flmlock / random / oracle
    learn = mode not in ("random", "oracle")
    lock = args.lock_std if mode == "flmlock" else None
    nsacc = 0 if mode == "oracle" else args.saccades
    jpath = os.path.join(args.out, f"{args.tag}_seed{args.seed}.json")
    tt, oa_t, dist_t = [], [], []; correct, dsum = 0, 0.0
    for i, idx in enumerate(order):
        img, y = ds[int(idx)]; y = int(y); oc = ds.object_center(int(idx))
        pred, r, locked, fd = present_flm(
            brain, img, y, policy, oc, max_saccades=nsacc, ticks_per_sacc=args.ticks_per_sacc,
            memorize_ticks=args.memorize_ticks, lock_std=lock, canvas=args.canvas,
            where_signal=args.where, memorize_fovea_only=True, learn_gaze=learn, train_decoder=True)
        correct += int(pred == y); dsum += fd
        if (i + 1) % args.probe_every == 0:
            tt.append(i + 1); oa_t.append(correct / args.probe_every); dist_t.append(dsum / args.probe_every)
            json.dump(dict(exp="gaze", tag=args.tag, policy=args.policy, gaze_mode=mode,
                           dataset=args.dataset, seed=args.seed, traj_t=tt, traj_oa=oa_t,
                           traj_dist=dist_t, final_oa=oa_t[-1], final_dist=dist_t[-1],
                           complete=False), open(jpath, "w"))   # checkpoint
            log(f"    [{args.tag}] {i+1:4d} acc {oa_t[-1]:.3f} dist {dist_t[-1]:.1f}px")
            correct, dsum = 0, 0.0
    out = dict(exp="gaze", tag=args.tag, policy=args.policy, gaze_mode=mode, dataset=args.dataset,
               seed=args.seed, traj_t=tt, traj_oa=oa_t, traj_dist=dist_t,
               final_oa=oa_t[-1] if oa_t else None, final_dist=dist_t[-1] if dist_t else None,
               complete=True)
    json.dump(out, open(jpath, "w"))
    log(f"[{time.time()-t0:.0f}s] {args.tag} final acc {out['final_oa']} dist {out['final_dist']}")


# --------------------------------------------------------------------------- #
#  Animation (short dwell for watchability -- the 500t measure dwell = 12k frames)
# --------------------------------------------------------------------------- #
def _animate(brain, ds, idxs, args, log):
    from snn_classification_realtime.foveation.minibrain.render_run import record_run, render_video
    imgs = [ds[int(i)][0] for i in idxs]; labs = [int(ds[int(i)][1]) for i in idxs]
    rec = record_run(brain, imgs, labs, ticks_per_image=args.anim_ticks, teach=True,
                     learn=False, gaze_mode="klinotaxis", seed=args.seed)
    path = os.path.join(args.out, f"{args.tag}_seed{args.seed}.mp4")
    p = render_video(rec, path, fps=12)
    log(f"    [anim] {p}")


def _plot_traj(out, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    t = out["traj_t"]
    ax[0].plot(t, out["traj_sep"], "-o", ms=3, label="adapted sep")
    ax[0].plot(t, out["fresh_sep"], "--s", ms=3, color="gray", label="fresh-net sep")
    ax[0].axhline(0.10, color="k", ls=":", lw=1, label="chance")
    ax[0].set_title(f"{out['tag']} — frozen-probe separability"); ax[0].set_xlabel("images")
    ax[0].legend(fontsize=8)
    ax[1].plot(t, out["traj_oa"], "-o", ms=3, color="tab:green")
    ax[1].set_title("online decoder acc"); ax[1].set_xlabel("images"); ax[1].axhline(0.10, color="k", ls=":", lw=1)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


# --------------------------------------------------------------------------- #
def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True, choices=["sep", "plast", "gaze"])
    p.add_argument("--tag", default="arm")
    p.add_argument("--dataset", default="cifar10_grayscale")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="foveation_results/minibrain/arch")
    # locked run params
    p.add_argument("--dwell", type=int, default=500)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--images", type=int, default=400)
    p.add_argument("--probe-every", type=int, default=100)
    p.add_argument("--probe-n", type=int, default=300)
    p.add_argument("--probe-dwell", type=int, default=200,
                   help="ticks/image for the separability probe settle (< run dwell)")
    # architecture axes
    p.add_argument("--arch", default="reservoir_random",
                   choices=["reservoir_random", "reservoir_retino", "conv"])
    p.add_argument("--conv-bank", default="default", choices=["default", "rich"])
    p.add_argument("--readout", default="input", choices=["input", "pool", "all"])
    p.add_argument("--readout-head", default="linear", choices=["linear", "mlp"])
    p.add_argument("--mlp-hidden", type=int, default=128)
    # plasticity axes
    p.add_argument("--unfreeze", action="store_true")
    p.add_argument("--plasticity-mode", default="legacy_multiplicative",
                   choices=["legacy_multiplicative", "error_correcting", "reward_hebb"])
    p.add_argument("--nm-kappa", type=float, default=0.0)
    p.add_argument("--rh-decay", type=float, default=0.1)
    p.add_argument("--eta-post-res", type=float, default=-1.0)
    p.add_argument("--ext-plasticity", default="none", choices=["none", "oja", "bcm", "rmhebb"])
    p.add_argument("--ext-plast-eta", type=float, default=0.002)
    p.add_argument("--assoc", action="store_true")
    p.add_argument("--assoc-dim", type=int, default=256)
    p.add_argument("--teach", type=int, default=1)
    p.add_argument("--teacher-mode", default="rpe", choices=["rpe", "graded", "binary"])
    p.add_argument("--reward-gain", type=float, default=3.0)
    p.add_argument("--stress-gain", type=float, default=3.0)
    p.add_argument("--participation", type=float, default=0.04)
    # gaze axes
    p.add_argument("--policy", default="linear", choices=["linear", "mlp", "recurrent"])
    p.add_argument("--gaze-mode", default="flmlock", choices=["flm", "flmlock", "random", "oracle"])
    p.add_argument("--canvas", type=int, default=72)
    p.add_argument("--distractors", type=int, default=4)
    p.add_argument("--fovea", type=int, default=44)
    p.add_argument("--grid", type=int, default=16)
    p.add_argument("--saccades", type=int, default=8)
    p.add_argument("--ticks-per-sacc", type=int, default=8)
    p.add_argument("--memorize-ticks", type=int, default=200)
    p.add_argument("--max-step", type=float, default=18.0)
    p.add_argument("--lock-std", type=float, default=0.02)
    p.add_argument("--where", default="periph", choices=["periph", "readout"])
    # animation
    p.add_argument("--animate", action="store_true")
    p.add_argument("--anim-images", type=int, default=16)
    p.add_argument("--anim-ticks", type=int, default=60)
    p.add_argument("--aggregate", action="store_true")
    return p


def _mean_std(rs, key):
    A = np.array([r[key] for r in rs if r.get(key) is not None], float)
    return (A.mean(0), A.std(0)) if len(A) else (None, None)


def aggregate(out):
    """Build per-experiment comparison figures from all JSONs in `out`."""
    runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(out, "*_seed*.json")))]
    by_exp = {}
    for r in runs:
        by_exp.setdefault(r["exp"], []).append(r)
    summary = {}
    # --- Exp1 sep: grouped bars (lin / mlp) per arm ---
    if "sep" in by_exp:
        rs = by_exp["sep"]
        tags = sorted({r["tag"] for r in rs})
        lin = [np.mean([r["lin"] for r in rs if r["tag"] == t]) for t in tags]
        mlp = [np.nanmean([r["mlp"] if r["mlp"] is not None else np.nan for r in rs if r["tag"] == t]) for t in tags]
        fig, ax = plt.subplots(figsize=(max(7, len(tags) * 0.9), 4.4))
        x = np.arange(len(tags))
        ax.bar(x - 0.2, lin, 0.38, label="linear probe", color="tab:blue")
        ax.bar(x + 0.2, mlp, 0.38, label="MLP readout", color="tab:orange")
        ax.axhline(0.10, color="k", ls=":", lw=1, label="chance")
        ax.set_xticks(x); ax.set_xticklabels(tags, rotation=40, ha="right", fontsize=7)
        ax.set_ylabel("held-out accuracy"); ax.set_title("Exp1 — frozen-encoder separability")
        ax.legend(fontsize=8); fig.tight_layout()
        fig.savefig(os.path.join(out, "AGG_sep.png"), dpi=130); plt.close(fig)
        summary["sep"] = {t: dict(lin=round(l, 3), mlp=(round(m, 3) if not np.isnan(m) else None))
                          for t, l, m in zip(tags, lin, mlp)}
    # --- Exp2 plast: separability + online-acc trajectories ---
    if "plast" in by_exp:
        rs = by_exp["plast"]
        tags = sorted({r["tag"] for r in rs})
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
        for t in tags:
            g = [r for r in rs if r["tag"] == t]
            tt = g[0]["traj_t"]; m, s = _mean_std(g, "traj_sep")
            ax[0].plot(tt, m, "-o", ms=3, label=t)
            mo, _ = _mean_std(g, "traj_oa"); ax[1].plot(tt, mo, "-o", ms=3, label=t)
        ax[0].axhline(0.10, color="k", ls=":", lw=1); ax[0].set_title("Exp2 — frozen-probe separability")
        ax[0].set_xlabel("images"); ax[0].legend(fontsize=7)
        ax[1].axhline(0.10, color="k", ls=":", lw=1); ax[1].set_title("online decoder acc")
        ax[1].set_xlabel("images"); fig.tight_layout()
        fig.savefig(os.path.join(out, "AGG_plast.png"), dpi=130); plt.close(fig)
        summary["plast"] = {t: dict(final_sep=round(float(np.mean([r["final_sep"] for r in rs if r["tag"] == t])), 3),
                                    final_oa=round(float(np.mean([r["final_oa"] for r in rs if r["tag"] == t])), 3))
                            for t in tags}
    # --- Exp3 gaze: acc + distance trajectories ---
    if "gaze" in by_exp:
        rs = by_exp["gaze"]
        tags = sorted({r["tag"] for r in rs})
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
        for t in tags:
            g = [r for r in rs if r["tag"] == t]; tt = g[0]["traj_t"]
            mo, _ = _mean_std(g, "traj_oa"); md, _ = _mean_std(g, "traj_dist")
            ax[0].plot(tt, mo, "-o", ms=3, label=t); ax[1].plot(tt, md, "-o", ms=3, label=t)
        ax[0].axhline(0.10, color="k", ls=":", lw=1); ax[0].axhline(0.18, color="purple", ls="--", lw=1)
        ax[0].set_title("Exp3 — gaze online acc"); ax[0].set_xlabel("images"); ax[0].legend(fontsize=7)
        ax[1].set_title("gaze→object distance (px)"); ax[1].set_xlabel("images")
        fig.tight_layout(); fig.savefig(os.path.join(out, "AGG_gaze.png"), dpi=130); plt.close(fig)
        summary["gaze"] = {t: dict(final_oa=round(float(np.mean([r["final_oa"] for r in rs if r["tag"] == t])), 3),
                                   final_dist=round(float(np.mean([r["final_dist"] for r in rs if r["tag"] == t])), 1))
                           for t in tags}
    json.dump(summary, open(os.path.join(out, "AGG_summary.json"), "w"), indent=2)
    print("=== AGGREGATE SUMMARY ===")
    print(json.dumps(summary, indent=2))
    return summary


def main():
    args = build_argparser().parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.aggregate:
        return aggregate(args.out)
    args.teach = bool(args.teach)
    lf = open(os.path.join(args.out, f"{args.tag}_seed{args.seed}.log"), "a")
    def log(m): print(m, flush=True); lf.write(m + "\n"); lf.flush()
    log(f"=== {args.exp} {args.tag} seed{args.seed} dataset={args.dataset} "
        f"dwell={args.dwell} warmup={args.warmup} images={args.images} ===")
    {"sep": run_sep, "plast": run_plast, "gaze": run_gaze}[args.exp](args, log)


if __name__ == "__main__":
    main()
