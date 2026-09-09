"""R1 readout evaluation -- the REAL goal metric: closed-form (no-gradient) top-1 on a held-out
RGB CIFAR-10 test set, decoded from PAULA substrate activity.

The separability_probe gives a fast proxy on ~400 samples (so wide layers with dim >> n are
under-estimated). This script is the honest goal measurement: record substrate activity for a
TRAIN pool (CIFAR train split) and a disjoint TEST pool (CIFAR test split), fit a closed-form
readout (nearest-centroid / LDA / ridge -- no gradient, so it stays within the "no torch
training" R1 rung) per layer on TRAIN, and score TEST. Reports per-layer, per-readout top-1.

Reuses the parallel per-sample recording (fresh reset per sample). CLI-only.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np

from neuron.nn_core import NNCore
from neuron.network_config import NetworkConfig
from neuron.neuron import Neuron
from snn_classification_realtime.activity_dataset_builder.network_utils import (
    infer_layers_from_metadata, determine_input_mapping,
)
from snn_classification_realtime.activity_dataset_builder.vision_datasets import load_dataset_by_name
# Reuse separability_probe's PROVEN worker functions verbatim (they complete M1/retina gates
# reliably under macOS 'spawn'); _init_worker now takes an is_test flag for the CIFAR test split.
from snn_classification_realtime.paula_vision.separability_probe import (
    _init_worker, _worker_record, lda_acc, ridge_acc,
)


def ncc_fit_eval(Xtr, ytr, Xte, yte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    R = (Xtr - mu) / sd; Q = (Xte - mu) / sd
    classes = np.unique(ytr)
    cents = np.stack([R[ytr == c].mean(0) for c in classes])
    d = np.linalg.norm(Q[:, None, :] - cents[None], axis=2)
    return float((classes[d.argmin(1)] == yte).mean())


def build_pool(ds, classes, per_class, seed):
    rng = np.random.RandomState(seed)
    by_c = defaultdict(list)
    for i in rng.permutation(len(ds)):
        i = int(i); lab = int(ds[i][1])
        if lab in classes and len(by_c[lab]) < per_class:
            by_c[lab].append(i)
        if all(len(by_c[c]) >= per_class for c in classes):
            break
    return [i for c in classes for i in by_c[c]]


def record(idxs, layers, init_args, workers, label):
    import multiprocessing as mp
    rows = {L: {} for L in layers}
    t0 = time.time()
    with mp.Pool(workers, initializer=_init_worker, initargs=init_args) as pool:
        for done, (i, pl, _raw) in enumerate(pool.imap_unordered(_worker_record, idxs, chunksize=1)):
            for L in layers:
                rows[L][i] = pl.get(L, np.zeros(0, np.float32))
            if (done + 1) % 200 == 0:
                print(f"    {label} {done + 1}/{len(idxs)} ({time.time() - t0:.0f}s)", flush=True)
    return {L: np.stack([rows[L][i] for i in idxs]) for L in layers}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True)
    ap.add_argument("--dataset", default="cifar10_color")
    ap.add_argument("--classes", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--train-per-class", type=int, default=300)
    ap.add_argument("--test-per-class", type=int, default=200)
    ap.add_argument("--ticks", type=int, default=100)
    ap.add_argument("--settle-frac", type=float, default=0.5)
    ap.add_argument("--signal-gain", type=float, default=0.336)
    ap.add_argument("--norm", type=float, default=0.6)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="foveation_results/paula_vision/readout")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--freeze", action="store_true",
                    help="freeze plasticity (eta=0) so the net is a deterministic fixed feature "
                         "extractor -- REQUIRED for a fixed-feature readout to generalize")
    ap.add_argument("--feats", default="S", choices=["S", "full"],
                    help="S = mean membrane only; full = canonical [mean_S, mean_t_ref, firing_rate]")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    classes = [int(x) for x in a.classes.split(",")]
    settle = int(a.ticks * a.settle_frac)

    net = NetworkConfig.load_network_config(a.net, neuron_class=Neuron)
    meta = {n["id"]: n["metadata"].get("layer") for n in json.load(open(a.net)).get("neurons", [])}
    layers = sorted({L for L in meta.values() if L is not None})
    N = len(net.network.neurons)
    dc = load_dataset_by_name(a.dataset, train=True, cifar10_color_normalization_factor=a.norm)
    dc_te = load_dataset_by_name(a.dataset, train=False, cifar10_color_normalization_factor=a.norm)
    tr_idx = build_pool(dc.dataset, classes, a.train_per_class, seed=1)
    te_idx = build_pool(dc_te.dataset, classes, a.test_per_class, seed=2)
    ytr = np.array([int(dc.dataset[i][1]) for i in tr_idx])
    yte = np.array([int(dc_te.dataset[i][1]) for i in te_idx])
    print(f"[{a.tag}] net={a.net} N={N} layers={layers} train={len(tr_idx)} test={len(te_idx)} "
          f"ticks={a.ticks} gain={a.signal_gain}", flush=True)

    # _init_worker: (net_path, dataset, signal_gain, ticks, settle, shuffle_time, norm, is_test, freeze, feats)
    tr_act = record(tr_idx, layers, (a.net, a.dataset, a.signal_gain, a.ticks, settle, False, a.norm, False, a.freeze, a.feats),
                    a.workers, "TRAIN")
    te_act = record(te_idx, layers, (a.net, a.dataset, a.signal_gain, a.ticks, settle, False, a.norm, True, a.freeze, a.feats),
                    a.workers, "TEST")

    res = {"tag": a.tag, "net": a.net, "train": len(tr_idx), "test": len(te_idx),
           "classes": classes, "per_layer": {}}
    print(f"\n[{a.tag}] closed-form readout top-1 on held-out TEST (chance {1/len(classes):.3f}):", flush=True)
    best = 0.0; best_where = ""
    for L in layers:
        Xtr, Xte = tr_act[L], te_act[L]
        accs = {"ncc": ncc_fit_eval(Xtr, ytr, Xte, yte),
                "lda": lda_acc(Xtr, ytr, Xte, yte),
                "ridge": ridge_acc(Xtr, ytr, Xte, yte)}
        res["per_layer"][f"L{L}"] = {k: round(v, 4) for k, v in accs.items()}
        for k, v in accs.items():
            if v > best:
                best, best_where = v, f"L{L}/{k}"
        print(f"  L{L} (dim {Xtr.shape[1]}): " + "  ".join(f"{k}={v:.3f}" for k, v in accs.items()), flush=True)
    res["best_top1"] = round(best, 4); res["best_where"] = best_where
    json.dump(res, open(os.path.join(a.out, f"{a.tag}.json"), "w"), indent=2)
    tier = ">=80 PASS" if best >= 0.80 else (">=70 PASS" if best >= 0.70 else "below 70")
    print(f"\n[{a.tag}] BEST closed-form top-1 = {best*100:.2f}% at {best_where}  [{tier}]", flush=True)


if __name__ == "__main__":
    main()
