"""Does SCALING PAULA help on CIFAR-10 — and does signal fade through depth?

The user's framing: the real classifier nets are only a few hundred neurons. The
question is whether growing the PAULA substrate (more neurons) raises how much
class structure it holds. But the user's warning is the crux: "the signal slowly
fades as it propagates through layers." So we cannot just stack layers and read a
global participation number — a deep layer can be silent while the input layer
spikes happily.

This probe therefore measures, per architecture:
  - PER-LAYER participation (fraction active), exposing the depth-fade directly.
  - Decodability (kNN + regularized linear, PCA) of the settled activity, read
    two ways: ALL neurons, and LAST layer only. If last-layer decodability
    collapses with depth while per-layer participation shows the deep layers
    going dark, that IS the fade.

Two scaling families:
  WIDTH  — one strided conv, more filters (pure high-dim storage, no fade path).
  DEPTH  — stacked convs, run under two threshold policies:
     * 'uniform'  : global input-gain calibrated, r_base left uniform  -> fade.
     * 'homeo'    : per-layer firing-threshold homeostasis (the user's "careful
                    threshold tuning" lever) -> tests whether tuning rescues it.

Honesty notes:
- kNN/linear on a settled signature is a LOWER BOUND; the trained temporal
  decoder can do better. We compare architectures under the SAME readout.
- Every net is compared in-regime (calibrated), so architecture, not drive, is
  the variable.
- No HDF5: signatures held in memory (storage-safe).
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from snn_classification_realtime.activity_dataset_builder.vision_datasets import (
    load_dataset_by_name,
)
from snn_classification_realtime.foveation.fovea import Fovea
from snn_classification_realtime.foveation.perception import (
    PerceptionNetwork,
    build_fovea_network_json,
)

# Architectures kept to a few hundred..few thousand neurons via strided convs.
# On 32x32: k4s2 -> 15x15=225 positions/filter; k3s2 after -> ~7x7=49; etc.
WIDTH = {
    "w2":  [{"type": "conv", "kernel_size": 4, "stride": 2, "filters": 2}],
    "w4":  [{"type": "conv", "kernel_size": 4, "stride": 2, "filters": 4}],
    "w8":  [{"type": "conv", "kernel_size": 4, "stride": 2, "filters": 8}],
    "w16": [{"type": "conv", "kernel_size": 4, "stride": 2, "filters": 16}],
}
DEPTH = {
    "d1": [{"type": "conv", "kernel_size": 4, "stride": 2, "filters": 4}],
    "d2": [{"type": "conv", "kernel_size": 4, "stride": 2, "filters": 4},
           {"type": "conv", "kernel_size": 3, "stride": 2, "filters": 6}],
    "d3": [{"type": "conv", "kernel_size": 4, "stride": 2, "filters": 4},
           {"type": "conv", "kernel_size": 3, "stride": 2, "filters": 6},
           {"type": "conv", "kernel_size": 3, "stride": 1, "filters": 8}],
    "d4": [{"type": "conv", "kernel_size": 4, "stride": 2, "filters": 4},
           {"type": "conv", "kernel_size": 3, "stride": 2, "filters": 6},
           {"type": "conv", "kernel_size": 3, "stride": 1, "filters": 8},
           {"type": "conv", "kernel_size": 3, "stride": 1, "filters": 8}],
}


def run_image(perc, whole, image, ticks, last):
    """Return per-position feature matrix (N,4)=[rate_all,rate_late,S_late,tref_late]
    and per-layer late participation."""
    perc.reset()
    sig = perc.patch_to_signals(whole.crop(image))
    n = perc.num_neurons
    O = np.zeros((ticks, n), np.float32)
    Slate = np.zeros(n, np.float32)
    Tlate = np.zeros(n, np.float32)
    for t in range(ticks):
        st = perc.step(sig)
        O[t] = st.O
        if t >= ticks - last:
            Slate += st.S
            Tlate += st.t_ref
    rate_all = O.mean(0)
    rate_late = O[ticks - last:].mean(0)
    feats = np.stack([rate_all, rate_late, Slate / last, Tlate / last], axis=1)
    return feats, rate_late


def layer_participation(perc, rate_late):
    return {l: float(rate_late[perc.layer_of_pos == l].mean())
            for l in perc.layer_indices}


def measure_participation(perc, whole, ds, idxs, ticks, last):
    accs = {l: [] for l in perc.layer_indices}
    for i in idxs:
        image, _ = ds[i]
        _, rl = run_image(perc, whole, image, ticks, last)
        for l, v in layer_participation(perc, rl).items():
            accs[l].append(v)
    return {l: float(np.mean(v)) for l, v in accs.items()}


def calibrate_input_gain(perc, whole, ds, ds_cfg, idxs, ticks, last, target):
    """Uniform-threshold policy: scan global signal_gain for overall participation."""
    best = (1.0, 1e9, 0.0)
    for g in (0.3, 1.0, 3.0, 10.0, 30.0):
        ds_cfg.signal_gain = g
        part = measure_participation(perc, whole, ds, idxs, ticks, last)
        overall = float(np.mean(list(part.values())))
        if abs(overall - target) < best[1]:
            best = (g, abs(overall - target), overall)
    ds_cfg.signal_gain = best[0]
    return best[0]


def calibrate_layer_homeostasis(perc, whole, ds, ds_cfg, idxs, ticks, last,
                                target, iters=5, k=0.6):
    """Per-layer firing-threshold homeostasis: raise r_base where a layer is too
    active, lower it where a layer is fading. Fixes gain at 1 and lets thresholds
    carry every layer to the target participation."""
    ds_cfg.signal_gain = 1.0
    for _ in range(iters):
        part = measure_participation(perc, whole, ds, idxs, ticks, last)
        for l in perc.layer_indices:
            p = part[l]
            factor = ((p + 1e-3) / (target + 1e-3)) ** k  # >1 if too active
            factor = float(np.clip(factor, 0.4, 2.5))
            perc.scale_layer_threshold(l, factor)
    return measure_participation(perc, whole, ds, idxs, ticks, last)


def collect(perc, whole, ds, idxs, ticks, last):
    feats, y = [], []
    for i in tqdm(idxs, desc="signatures", leave=False):
        image, lbl = ds[i]
        f, _ = run_image(perc, whole, image, ticks, last)
        feats.append(f)
        y.append(int(lbl))
    return np.stack(feats), np.array(y)  # (M, N, 4), (M,)


def probe(Ftr, ytr, Fte, yte, pca_dim):
    Xtr = Ftr.reshape(len(Ftr), -1)
    Xte = Fte.reshape(len(Fte), -1)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    d = min(pca_dim, Xtr.shape[1], Xtr.shape[0] - 1)
    pca = PCA(n_components=d).fit(Xtr)
    Ztr, Zte = pca.transform(Xtr), pca.transform(Xte)
    knn = KNeighborsClassifier(n_neighbors=5, metric="cosine").fit(Ztr, ytr)
    lr = LogisticRegression(max_iter=400, C=0.5).fit(Ztr, ytr)
    return float(knn.score(Zte, yte)), float(lr.score(Zte, yte)), d


def build_perc(name, layers, channels, H, ds_cfg, args):
    net_path = os.path.join(args.output_dir, f"paula_cap_{channels}x{H}_{name}.json")
    os.makedirs(args.output_dir, exist_ok=True)
    build_fovea_network_json(net_path, channels=channels, size=H,
                             layers=layers, seed=args.seed)
    return PerceptionNetwork(net_path, ds_cfg, ablation=args.ablation)


def evaluate(name, layers, policy, channels, H, whole, ds, ds_cfg, args,
             train_idx, test_idx, cal_idx):
    perc = build_perc(name, layers, channels, H, ds_cfg, args)
    if policy == "homeo":
        part = calibrate_layer_homeostasis(perc, whole, ds, ds_cfg, cal_idx,
                                           args.ticks, args.last,
                                           args.target_participation)
        gain = 1.0
    else:
        gain = calibrate_input_gain(perc, whole, ds, ds_cfg, cal_idx,
                                    args.ticks, args.last,
                                    args.target_participation)
        part = measure_participation(perc, whole, ds, cal_idx, args.ticks, args.last)
    part_str = " ".join(f"L{l}:{part[l]:.3f}" for l in perc.layer_indices)
    print(f"[{name}/{policy}] {perc.num_neurons} neurons | gain {gain} | "
          f"participation {part_str}")

    Ftr, ytr = collect(perc, whole, ds, train_idx, args.ticks, args.last)
    Fte, yte = collect(perc, whole, ds, test_idx, args.ticks, args.last)
    knn_all, lin_all, d_all = probe(Ftr, ytr, Fte, yte, args.pca_dim)

    last = max(perc.layer_indices)
    mask = perc.layer_of_pos == last
    knn_last, lin_last, d_last = probe(Ftr[:, mask], ytr, Fte[:, mask], yte, args.pca_dim)
    print(f"[{name}/{policy}] decodability  ALL kNN {knn_all:.3f} lin {lin_all:.3f} | "
          f"LAST(L{last},{int(mask.sum())}n) kNN {knn_last:.3f} lin {lin_last:.3f} "
          f"(chance 0.10)")
    return {
        "num_neurons": perc.num_neurons,
        "policy": policy,
        "gain": gain,
        "participation": part,
        "last_layer": int(last),
        "last_layer_neurons": int(mask.sum()),
        "knn_all": knn_all, "linear_all": lin_all,
        "knn_last": knn_last, "linear_last": lin_last,
        "pca_dim_all": d_all, "pca_dim_last": d_last,
    }


def run(args):
    torch.manual_seed(0)
    np.random.seed(0)
    ds_cfg = load_dataset_by_name(args.dataset_name, train=True)
    ds = ds_cfg.dataset
    img0, _ = ds[0]
    channels, H, W = img0.shape
    whole = Fovea(image_h=H, image_w=W, size=H)  # size==H -> whole image

    by_label = {i: [] for i in range(10)}
    need = args.train_per_class + args.test_per_class
    for i in range(len(ds)):
        _, l = ds[i]; l = int(l)
        if len(by_label[l]) < need:
            by_label[l].append(i)
        if all(len(by_label[c]) >= need for c in range(10)):
            break
    train_idx, test_idx = [], []
    for c in range(10):
        train_idx += by_label[c][:args.train_per_class]
        test_idx += by_label[c][args.train_per_class:need]
    cal_idx = train_idx[::max(1, len(train_idx) // args.cal_images)][:args.cal_images]

    results = {"width": {}, "depth": {}}
    for name in args.width:
        results["width"][name] = evaluate(
            name, WIDTH[name], "homeo", channels, H, whole, ds, ds_cfg, args,
            train_idx, test_idx, cal_idx)
    for name in args.depth:
        for policy in args.depth_policies:
            results["depth"][f"{name}/{policy}"] = evaluate(
                name, DEPTH[name], policy, channels, H, whole, ds, ds_cfg, args,
                train_idx, test_idx, cal_idx)

    tag = f"_{args.tag}" if args.tag else ""
    out = os.path.join(args.output_dir,
                       f"paula_capacity_{args.dataset_name}{tag}_{int(time.time())}.json")
    with open(out, "w") as f:
        json.dump({"config": vars(args), "results": results}, f, indent=2)

    print("\n=== WIDTH scaling (depth 1, homeostatic threshold) ===")
    print(f"{'arch':<6}{'neurons':>9}{'kNN_all':>9}{'lin_all':>9}")
    for name in args.width:
        r = results["width"][name]
        print(f"{name:<6}{r['num_neurons']:>9}{r['knn_all']:>9.3f}{r['linear_all']:>9.3f}")
    print("\n=== DEPTH scaling: does per-layer threshold tuning beat the fade? ===")
    print(f"{'arch/pol':<12}{'neurons':>9}{'lastN':>7}{'kNN_all':>9}{'kNN_last':>10}"
          f"{'lin_last':>9}  per-layer participation")
    for key, r in results["depth"].items():
        pstr = " ".join(f"L{l}:{v:.2f}" for l, v in sorted(r["participation"].items()))
        print(f"{key:<12}{r['num_neurons']:>9}{r['last_layer_neurons']:>7}"
              f"{r['knn_all']:>9.3f}{r['knn_last']:>10.3f}{r['linear_last']:>9.3f}  {pstr}")
    print(f"\nSaved: {out}")
    return results


def main():
    p = argparse.ArgumentParser(description="PAULA CIFAR capacity + depth-fade probe")
    p.add_argument("--dataset-name", default="cifar10_grayscale")
    p.add_argument("--width", nargs="*", default=["w2", "w4", "w8"])
    p.add_argument("--depth", nargs="*", default=["d1", "d2", "d3"])
    p.add_argument("--depth-policies", nargs="+", default=["uniform", "homeo"])
    p.add_argument("--tag", default="")
    p.add_argument("--ticks", type=int, default=60)
    p.add_argument("--last", type=int, default=20)
    p.add_argument("--train-per-class", type=int, default=20)
    p.add_argument("--test-per-class", type=int, default=10)
    p.add_argument("--cal-images", type=int, default=12)
    p.add_argument("--target-participation", type=float, default=0.1)
    p.add_argument("--pca-dim", type=int, default=120)
    p.add_argument("--ablation", default="none")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="foveation_results")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
