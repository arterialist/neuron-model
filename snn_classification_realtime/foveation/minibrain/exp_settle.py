"""Part B (+ Part A) -- settling-ticks sweep, storage-lean, sharded-parallel.

Question 1 (Part A, >=0.50): does the SUBSTRATE PRESERVE the ~0.48 CIFAR signal the
retinal-conv encoder already extracts (measured by exp_scale convfeat ceiling)? The
original pipeline hit ~0.50 with a conv-PAULA substrate, so the `conv` arm is the one
expected to reach the goal; the random/retinotopic reservoirs test how lossy a generic
spiking substrate is.

Question 2 (Part B): as the network is allowed to SETTLE longer on one image
(dwell in {25,50,100,200,400,800}), does class separability grow, and does a DYNAMICS
readout (binned firing-rate profile over the settle window) overtake the MEAN readout?
Per the model's spatiotemporal-representation thesis the dynamics profile should carry
information the mean discards -- this measures whether that shows up for CIFAR labels.

Storage-lean: one pass of `max_dwell` ticks per image; we accumulate (a) running sums of
S/F_avg/O snapshotted at each checkpoint -> per-checkpoint MEAN features, and (b) firing
counts per inter-checkpoint INTERVAL -> the rate PROFILE (dynamics feature). Never store
the (T x N) trajectory. ~8 MB per shard.

Sharded-parallel: one arm at a time uses all cores; each worker warms an identical brain
(same seed) and processes a disjoint shard of samples, writing an .npz. Resumable at both
arm level (skip complete arm JSON) and shard level (skip existing .npz).

Run it yourself (no background cap):
    PYTHONPATH=. OMP_NUM_THREADS=1 .venv/bin/python -m \
        snn_classification_realtime.foveation.minibrain.exp_settle
Smoke-test first (seconds):
    ... exp_settle --smoke
Live progress: one tqdm bar per shard. Per-arm JSON + AGG_settle.png/json at the end.
"""
from __future__ import annotations

import argparse, os, json, time, glob
from multiprocessing import Process, RLock
import numpy as np
from tqdm import tqdm

from snn_classification_realtime.activity_dataset_builder.vision_datasets import load_dataset_by_name
from snn_classification_realtime.foveation.minibrain.core import MiniBrainConfig, MiniBrain
from snn_classification_realtime.foveation.minibrain.heads import TorchMLPHead

CHECKPTS = [25, 50, 100, 200, 400, 800]          # cumulative settle-tick counts
INTERVALS = [(0, 25), (25, 50), (50, 100), (100, 200), (200, 400), (400, 800)]


def make_cfg(arch, dataset, out_dir, seed, dwell):
    substrate = "conv" if arch == "conv" else "reservoir"
    wiring = "retinotopic" if arch == "reservoir_retino" else "random"
    return MiniBrainConfig(dataset_name=dataset, dwell=dwell, seed=seed, substrate_type=substrate,
                           wiring=wiring, conv_bank="rich", readout="all",
                           fovea=32, periph=32, grid=16, output_dir=out_dir)


def warmup(brain, ds, order, ticks):
    per = max(1, brain.cfg.input_period); done = 0; i = 0
    brain.sub.set_learning(False)
    while done < ticks:
        sig = brain.sub.patch_to_signals(brain._encode(ds[int(order[i % 50])][0]))
        for t in range(brain.cfg.dwell):
            brain.sub.step((sig if t % per == 0 else []) + brain._tonic); done += 1
            if done >= ticks:
                break
        i += 1
    brain.sub.set_learning(True)


def shard(arch, dataset, seed, idxs, max_dwell, warm, spath, pos, desc):
    """Process one shard of images; write per-checkpoint mean sums + per-interval firing
    counts to `spath`.npz. Skips if already present (resume)."""
    if os.path.exists(spath):
        return
    ds = load_dataset_by_name(dataset, train=True).dataset
    dste = load_dataset_by_name(dataset, train=False).dataset
    cfg = make_cfg(arch, dataset, spath + "_net", seed, max_dwell)
    brain = MiniBrain(cfg, load_dataset_by_name(dataset, train=True))
    warm_order = np.random.RandomState(seed).randint(0, len(ds), size=50)
    warmup(brain, ds, warm_order, warm)
    brain.sub.set_learning(False)
    per = max(1, cfg.input_period); m = brain._readout_mask; N = int(m.sum())
    C = len(CHECKPTS); nI = len(INTERVALS); n = len(idxs)
    Ssum = np.zeros((n, C, N), np.float32); Fsum = np.zeros((n, C, N), np.float32)
    Osum = np.zeros((n, C, N), np.float32); Oint = np.zeros((n, nI, N), np.float32)
    Y = np.zeros(n, np.int64)
    cp_set = set(CHECKPTS)
    # map each tick -> interval index
    tick_iv = np.zeros(max_dwell, np.int64)
    for iv, (lo, hi) in enumerate(INTERVALS):
        tick_iv[lo:hi] = iv
    bar = tqdm(total=n, position=pos, desc=desc, leave=True, ncols=100)
    for k, i in enumerate(idxs):
        img, y = dste[int(i)]; Y[k] = int(y)
        sig = brain.sub.patch_to_signals(brain._encode(img))
        rs = np.zeros(N); rf = np.zeros(N); ro = np.zeros(N)  # running sums (mean feat)
        ci = 0
        for t in range(max_dwell):
            st = brain.sub.step((sig if t % per == 0 else []) + brain._tonic)
            rs += st.S[m]; rf += st.F_avg[m]; o = (st.O[m] > 0).astype(np.float32)
            ro += o; Oint[k, tick_iv[t]] += o
            if (t + 1) in cp_set:
                Ssum[k, ci] = rs; Fsum[k, ci] = rf; Osum[k, ci] = ro; ci += 1
        bar.update(1)
    bar.close()
    np.savez_compressed(spath, Ssum=Ssum, Fsum=Fsum, Osum=Osum, Oint=Oint, Y=Y)


def worker(lock, *args):
    tqdm.set_lock(lock)
    shard(*args)


def run_arm(arch, dataset, seed, images, max_dwell, warm, out, par):
    tag = f"settle_{arch}_{dataset.replace('cifar10', 'c10')}_s{seed}"
    jpath = os.path.join(out, f"{tag}.json")
    if os.path.exists(jpath):
        try:
            if json.load(open(jpath)).get("complete"):
                print(f"[skip] {tag} complete", flush=True); return
        except Exception:
            pass
    dste = load_dataset_by_name(dataset, train=False).dataset
    idxs = np.random.RandomState(seed + 7).randint(0, len(dste), size=images)
    shards = np.array_split(idxs, par)
    sdir = os.path.join(out, tag + "_shards"); os.makedirs(sdir, exist_ok=True)
    lock = RLock(); procs = []
    for p, sh in enumerate(shards):
        spath = os.path.join(sdir, f"sh{p}.npz")
        pr = Process(target=worker, args=(lock, arch, dataset, seed, sh, max_dwell, warm,
                                          spath, p, f"{tag} sh{p}"))
        pr.start(); procs.append(pr)
    for pr in procs:
        pr.join()
    # combine
    parts = [np.load(os.path.join(sdir, f"sh{p}.npz")) for p in range(par)]
    Ssum = np.concatenate([d["Ssum"] for d in parts]); Fsum = np.concatenate([d["Fsum"] for d in parts])
    Osum = np.concatenate([d["Osum"] for d in parts]); Oint = np.concatenate([d["Oint"] for d in parts])
    Y = np.concatenate([d["Y"] for d in parts])
    n = len(Y); ntr = int(n * 0.7)
    rng = np.random.RandomState(0); perm = rng.permutation(n)
    tr, te = perm[:ntr], perm[ntr:]
    res = {}
    for ci, cp in enumerate(CHECKPTS):
        # MEAN feature: running sums / cp  (S, F, O of readout layer)
        mean_feat = np.concatenate([Ssum[:, ci] / cp, Fsum[:, ci] / cp, Osum[:, ci] / cp], axis=1)
        # DYNAMICS feature: per-interval firing RATE profile for intervals within [0,cp]
        n_iv = ci + 1
        rates = np.stack([Oint[:, iv] / (INTERVALS[iv][1] - INTERVALS[iv][0]) for iv in range(n_iv)], axis=1)
        dyn_feat = rates.reshape(n, -1)
        acc_mean = _mlp(mean_feat, Y, tr, te, seed)
        acc_dyn = _mlp(dyn_feat, Y, tr, te, seed)
        res[str(cp)] = dict(mean=round(acc_mean, 3), dyn=round(acc_dyn, 3),
                            delta=round(acc_dyn - acc_mean, 3))
        print(f"  {tag} cp={cp:4d}: mean={acc_mean:.3f} dyn={acc_dyn:.3f} "
              f"delta={acc_dyn-acc_mean:+.3f}", flush=True)
    json.dump(dict(exp="settle", tag=tag, arch=arch, dataset=dataset, seed=seed,
                   images=images, max_dwell=max_dwell, n_neurons=int(Ssum.shape[2]),
                   n_test=int(len(te)), checkpoints=CHECKPTS, results=res, complete=True),
              open(jpath, "w"))
    # clean shards to save disk
    for f in glob.glob(os.path.join(sdir, "*.npz")):
        os.remove(f)


def _mlp(X, Y, tr, te, seed):
    head = TorchMLPHead(X.shape[1], 10, hidden=256, seed=seed)
    head.fit(X[tr], Y[tr])
    return float(head.score(X[te], Y[te]))


def aggregate(out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    runs = [json.load(open(f)) for f in glob.glob(os.path.join(out, "settle_*.json"))]
    if not runs:
        print("no runs"); return
    fig, ax = plt.subplots(figsize=(8, 5.2))
    colors = {"conv": "#0d818f", "reservoir_random": "#b45309", "reservoir_retino": "#6d28d9"}
    summ = {}
    for d in sorted(runs, key=lambda r: r["arch"]):
        cps = d["checkpoints"]; arch = d["arch"]
        mean = [d["results"][str(c)]["mean"] for c in cps]
        dyn = [d["results"][str(c)]["dyn"] for c in cps]
        c = colors.get(arch, "#374151")
        ax.plot(cps, mean, "-o", color=c, label=f"{arch} mean")
        ax.plot(cps, dyn, "--s", color=c, alpha=0.6, label=f"{arch} dyn")
        summ[arch] = d["results"]
    ax.axhline(0.483, color="k", ls=":", lw=1); ax.text(28, 0.49, "encoder ceiling 0.48", fontsize=8)
    ax.axhline(0.1, color="gray", ls=":", lw=1)
    ax.set_xscale("log"); ax.set_xticks(CHECKPTS); ax.set_xticklabels(CHECKPTS)
    ax.set_xlabel("settle ticks (dwell)"); ax.set_ylabel("held-out CIFAR-10 acc")
    ax.set_title("Settling sweep: substrate preservation + mean vs dynamics")
    ax.legend(fontsize=7, ncol=3); fig.tight_layout()
    fig.savefig(os.path.join(out, "AGG_settle.png"), dpi=130)
    json.dump(summ, open(os.path.join(out, "AGG_settle_summary.json"), "w"), indent=2)
    print("\n=== SETTLE SUMMARY ==="); print(json.dumps(summ, indent=2))
    print(f"saved {out}/AGG_settle.png + AGG_settle_summary.json")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--images", type=int, default=1000)
    p.add_argument("--max-dwell", type=int, default=800)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--par", type=int, default=9, help="shards (cores) per arm")
    p.add_argument("--archs", default="conv,reservoir_random,reservoir_retino")
    p.add_argument("--out", default="foveation_results/minibrain/settle")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if a.smoke:
        a.images, a.max_dwell, a.warmup, a.par = 60, 100, 60, 3
        global CHECKPTS, INTERVALS
        CHECKPTS = [25, 50, 100]; INTERVALS = [(0, 25), (25, 50), (50, 100)]
    os.makedirs(a.out, exist_ok=True)
    archs = a.archs.split(",")
    arms = [(arch, s) for arch in archs for s in range(a.seeds)]
    print(f"SETTLE SWEEP: {len(arms)} arms | images={a.images} max_dwell={a.max_dwell} "
          f"| par={a.par} | checkpoints={CHECKPTS} | out={a.out}", flush=True)
    t0 = time.time()
    for arch, seed in arms:
        print(f"\n--- arm {arch} seed={seed} ---", flush=True)
        run_arm(arch, a.dataset, seed, a.images, a.max_dwell, a.warmup, a.out, a.par)
    print(f"\nall arms done in {time.time()-t0:.0f}s", flush=True)
    aggregate(a.out)


if __name__ == "__main__":
    main()
