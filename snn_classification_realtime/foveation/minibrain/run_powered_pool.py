"""POWERED spatiotemporal confirmation run (user-launched, single command).

Purpose: give the GRU/LSTM enough data to settle the ~1-sigma pool-temporal effect
seen in the small run -- does reading the recurrent POOL's tick-by-tick trajectory
beat reading its mean? Records (O spikes full, plus running means of S and t_ref) for
many images, classifies input/pool/all with mean / temporal-linear / GRU / LSTM.

Run it yourself (no background cap):
    .venv/bin/python -m snn_classification_realtime.foveation.minibrain.run_powered_pool
Smoke-test first (seconds):
    .venv/bin/python -m snn_classification_realtime.foveation.minibrain.run_powered_pool --smoke

Live progress: one tqdm bar per arm (stacked). Results saved to --out; a combined
AGG_powered.png + AGG_powered_summary.json are written at the end. Tell me when it's
done and I read the JSON.
"""
from __future__ import annotations

import argparse, os, json, time
from multiprocessing import Process, RLock
import numpy as np
from tqdm import tqdm

from snn_classification_realtime.activity_dataset_builder.vision_datasets import load_dataset_by_name
from snn_classification_realtime.foveation.minibrain.core import MiniBrainConfig, MiniBrain
from snn_classification_realtime.foveation.minibrain.exp_temporal import (
    best_lin, temporal_features, RNNReadout,
)


def make_cfg(arch, dataset, out_dir, seed, dwell):
    substrate = "conv" if arch == "conv" else "reservoir"
    wiring = "retinotopic" if arch == "reservoir_retino" else "random"
    return MiniBrainConfig(dataset_name=dataset, dwell=dwell, seed=seed, substrate_type=substrate,
                           wiring=wiring, conv_bank="rich", readout="all", output_dir=out_dir)


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


def record(brain, ds, idxs, dwell, pos, desc):
    """Full spike trajectory O (n,T,N) + running means of S and t_ref (n,N) -- memory-
    lean: only O is kept per-tick (the RNN needs it); S/t_ref collapse to their mean
    (the standing state / regime remnant used by the linear features)."""
    brain.sub.set_learning(False)
    per = max(1, brain.cfg.input_period); N = brain.sub.num_neurons; n = len(idxs)
    O = np.zeros((n, dwell, N), np.float32); Sm = np.zeros((n, N), np.float32)
    Tm = np.zeros((n, N), np.float32); Y = np.zeros(n, np.int64)
    bar = tqdm(total=n, position=pos, desc=desc, leave=True, ncols=100)
    for k, i in enumerate(idxs):
        img, y = ds[int(i)]; Y[k] = int(y)
        sig = brain.sub.patch_to_signals(brain._encode(img))
        ss = np.zeros(N); tt = np.zeros(N)
        for t in range(dwell):
            st = brain.sub.step((sig if t % per == 0 else []) + brain._tonic)
            O[k, t] = (st.O > 0); ss += st.S; tt += st.t_ref
        Sm[k] = ss / dwell; Tm[k] = tt / dwell
        bar.update(1)
    bar.close()
    brain.sub.set_learning(True)
    return O, Sm, Tm, Y


def run_arm(arch, dataset, seed, images, dwell, warm, out, pos, gru):
    tag = f"pool_{arch}_{dataset.replace('cifar10','c10')}_s{seed}"
    jpath = os.path.join(out, f"{tag}.json")
    if os.path.exists(jpath):
        try:
            if json.load(open(jpath)).get("complete"):
                return
        except Exception:
            pass
    ds = load_dataset_by_name(dataset, train=True).dataset
    order = np.random.RandomState(seed).randint(0, len(ds), size=images + 60)
    cfg = make_cfg(arch, dataset, os.path.join(out, f"net_{tag}"), seed, dwell)
    brain = MiniBrain(cfg, load_dataset_by_name(dataset, train=True))
    warmup(brain, ds, order[:50], warm)
    O, Sm, Tm, Y = record(brain, ds, order[:images], dwell, pos, tag)
    lay = brain.sub.layer_of_pos
    groups = {"input": lay == min(brain.sub.layer_indices),
              "pool": lay == max(brain.sub.layer_indices),
              "all": np.ones(len(lay), bool)}
    ntr = int(images * 0.7); res = {}
    for g, mask in groups.items():
        Og = O[:, :, mask]; Sg = Sm[:, mask]; Tg = Tm[:, mask]
        mean_feat = np.concatenate([Og.mean(1), Sg, Tg], axis=1)
        # temporal_features expects (n,T,N) for S,TR; feed the mean broadcast over 1 step
        temp_feat = temporal_features(Og, Sg[:, None, :], Tg[:, None, :])
        r = dict(mean_lin=round(best_lin(mean_feat, Y, ntr), 3),
                 temporal_lin=round(best_lin(temp_feat, Y, ntr), 3), n_neurons=int(mask.sum()))
        if gru:
            r["gru"] = round(RNNReadout(int(mask.sum()), rnn="gru", seed=seed).fit(Og, Y), 3)
            r["lstm"] = round(RNNReadout(int(mask.sum()), rnn="lstm", seed=seed).fit(Og, Y), 3)
        res[g] = r
    json.dump(dict(exp="pool_powered", tag=tag, arch=arch, dataset=dataset, seed=seed,
                   images=images, dwell=dwell, groups=res, complete=True),
              open(jpath, "w"))


def worker(lock, *args):
    tqdm.set_lock(lock)
    run_arm(*args)


def aggregate(out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import glob
    from collections import defaultdict
    runs = [json.load(open(f)) for f in glob.glob(os.path.join(out, "pool_*.json"))]
    if not runs:
        print("no runs"); return
    agg = defaultdict(lambda: defaultdict(list))
    for d in runs:
        arch = d["arch"].replace("reservoir_", "")
        for g, r in d["groups"].items():
            agg[(arch, g)]["mean"].append(r["mean_lin"])
            agg[(arch, g)]["temp"].append(max(r["temporal_lin"], r.get("gru", 0), r.get("lstm", 0)))
    archs = sorted({a for a, _ in agg}); layers = ["input", "pool", "all"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    summ = {}
    for ax, lay in zip(axes, layers):
        x = np.arange(len(archs))
        m = [np.mean(agg[(a, lay)]["mean"]) if (a, lay) in agg else 0 for a in archs]
        t = [np.mean(agg[(a, lay)]["temp"]) if (a, lay) in agg else 0 for a in archs]
        ax.bar(x - 0.2, m, 0.38, label="mean", color="#6b7280")
        ax.bar(x + 0.2, t, 0.38, label="best temporal", color="#0d818f")
        ax.axhline(0.1, color="k", ls=":", lw=1); ax.set_xticks(x); ax.set_xticklabels(archs)
        ax.set_title(("pool  <- KEY" if lay == "pool" else lay))
        summ[lay] = {a: dict(mean=round(mm, 3), temporal=round(tt, 3), delta=round(tt - mm, 3))
                     for a, mm, tt in zip(archs, m, t)}
    axes[0].set_ylabel("held-out acc"); axes[0].legend(fontsize=8)
    fig.suptitle("POWERED pool run: mean vs temporal by layer"); fig.tight_layout()
    fig.savefig(os.path.join(out, "AGG_powered.png"), dpi=130)
    json.dump(summ, open(os.path.join(out, "AGG_powered_summary.json"), "w"), indent=2)
    print("\n=== POWERED SUMMARY (mean vs temporal, delta) ===")
    print(json.dumps(summ, indent=2))
    print(f"saved {out}/AGG_powered.png + AGG_powered_summary.json")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", type=int, default=1200)
    p.add_argument("--dwell", type=int, default=200)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--par", type=int, default=6, help="max arms in parallel")
    p.add_argument("--out", default="foveation_results/minibrain/pool_powered")
    p.add_argument("--smoke", action="store_true", help="tiny params to verify it runs")
    a = p.parse_args()
    if a.smoke:
        a.images, a.dwell, a.warmup, a.seeds = 40, 30, 30, 1
    os.makedirs(a.out, exist_ok=True)
    archs = ["conv", "reservoir_random", "reservoir_retino"]
    arms = [(arch, "cifar10_grayscale", s) for arch in archs for s in range(a.seeds)]
    print(f"POWERED POOL RUN: {len(arms)} arms | images={a.images} dwell={a.dwell} "
          f"| par={a.par} | out={a.out}", flush=True)
    t0 = time.time(); lock = RLock(); queue = list(enumerate(arms)); running = []
    while queue or running:
        while queue and len(running) < a.par:
            pos, (arch, ds, seed) = queue.pop(0)
            pr = Process(target=worker, args=(lock, arch, ds, seed, a.images, a.dwell,
                                              a.warmup, a.out, pos % a.par, bool(1)))
            pr.start(); running.append(pr)
        for pr in running:
            if not pr.is_alive():
                pr.join()
        running = [pr for pr in running if pr.is_alive()]
        time.sleep(1)
    print(f"\nall arms done in {time.time()-t0:.0f}s", flush=True)
    aggregate(a.out)


if __name__ == "__main__":
    main()
