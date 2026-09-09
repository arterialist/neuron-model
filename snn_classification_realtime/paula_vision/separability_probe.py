"""M0 — the separability gate. The most important instrument in the program.

A PAULA-only >70% result only MEANS something if the substrate itself adds class separability,
rather than being a passthrough whose signal is really carried by the retinal front-end and the
decoder. This module measures, per layer, whether that layer's OUTPUT activity is more linearly
separable than its INPUT, using a CLOSED-FORM (no-gradient) probe so the probe cannot "learn
around" a bad representation.

Representation ladder for a run:
    raw pixels  ->  L0 activity  ->  L1 activity  ->  ...  ->  L_last activity
Each arrow is one PAULA layer. The gate asks, per arrow:
    sep(output) > sep(input)  AND  sep(output) > sep(random_projection(input) -> same dim)

Closed-form separability (NO gradient descent anywhere):
  * ncc  : leave-one-out nearest-class-centroid accuracy (common-mode cancels).
  * lda  : Fisher LDA fit on TRAIN split, scored on TEST split (shrinkage-regularized).
  * ridge: one-vs-rest ridge-regression classifier, closed form (X^T X + lambda I)^-1 X^T Y.
These are the R1 decoder family; using them here means the gate and the eventual readout speak
the same language.

Anti-passthrough controls:
  * random-projection null: a gaussian map from the INPUT rep to the layer's output dim. A real
    layer must beat this, else it adds nothing a random matrix wouldn't.
  * passthrough: sep(output) <= sep(input) is a FAIL for that layer.
  * shuffled-time: recompute the static descriptor after per-neuron temporal shuffling of the
    settled window -> isolates whether the *dynamics* carry the gain vs the static mean code.

No neuron.py edit; drives the net exactly as the existing probes do (fresh reset per sample).
CLI-only. See EXPERIMENT_DESIGN_paula_vision.md (§3).
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
from snn_classification_realtime.core.input_mapping import image_to_signals
from snn_classification_realtime.activity_dataset_builder.vision_datasets import load_dataset_by_name
from snn_classification_realtime.activity_dataset_builder.network_utils import (
    infer_layers_from_metadata, determine_input_mapping,
)


# --------------------------------------------------------------------- closed-form probes

def _standardize(train, test):
    mu, sd = train.mean(0), train.std(0) + 1e-8
    return (train - mu) / sd, (test - mu) / sd


def ncc_loo(X, y):
    """Leave-one-out nearest-class-centroid accuracy. Common-mode (global pull) cancels."""
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-8)
    classes = np.unique(y)
    correct = 0
    for i in range(len(Xs)):
        best, bd = None, np.inf
        xi = Xs[i]
        for c in classes:
            m = (y == c)
            m[i] = False
            if not m.any():
                continue
            d = np.sum((xi - Xs[m].mean(0)) ** 2)
            if d < bd:
                bd, best = d, c
        correct += int(best == y[i])
    return correct / len(Xs)


def lda_acc(Xtr, ytr, Xte, yte, shrink=0.3):
    """Fisher LDA, shrinkage-regularized, closed form. No gradient."""
    Xtr, Xte = _standardize(Xtr, Xte)
    classes = np.unique(ytr)
    d = Xtr.shape[1]
    means = {c: Xtr[ytr == c].mean(0) for c in classes}
    Sw = np.zeros((d, d))
    for c in classes:
        Z = Xtr[ytr == c] - means[c]
        Sw += Z.T @ Z
    Sw /= len(Xtr)
    Sw = (1 - shrink) * Sw + shrink * np.trace(Sw) / d * np.eye(d)
    Swi = np.linalg.pinv(Sw)
    M = np.stack([means[c] for c in classes])            # (C,d)
    W = M @ Swi                                          # (C,d)
    b = -0.5 * np.einsum("cd,cd->c", M @ Swi, M) + np.log(1.0 / len(classes))
    pred = classes[(Xte @ W.T + b).argmax(1)]
    return float((pred == yte).mean())


def ridge_acc(Xtr, ytr, Xte, yte, lam=1.0):
    """One-vs-rest ridge regression classifier, closed form. No gradient."""
    Xtr, Xte = _standardize(Xtr, Xte)
    classes = np.unique(ytr)
    Y = np.stack([(ytr == c).astype(np.float64) for c in classes], 1)  # (n,C)
    A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
    Wc = np.linalg.solve(A, Xtr.T @ Y)                   # (d,C)
    pred = classes[(Xte @ Wc).argmax(1)]
    return float((pred == yte).mean())


def separability(feats, labels, seed=0, test_frac=0.4, probes=("ncc", "lda", "ridge")):
    """All closed-form probes on one representation. Returns dict of accuracies + margin."""
    rng = np.random.RandomState(seed)
    n = len(feats)
    perm = rng.permutation(n)
    nte = int(n * test_frac)
    te, tr = perm[:nte], perm[nte:]
    out = {}
    if "ncc" in probes:
        out["ncc"] = ncc_loo(feats, labels)
    if "lda" in probes:
        out["lda"] = lda_acc(feats[tr], labels[tr], feats[te], labels[te])
    if "ridge" in probes:
        out["ridge"] = ridge_acc(feats[tr], labels[tr], feats[te], labels[te])
    return out


def random_projection_null(prev, out_dim, seed):
    """Gaussian random map of the previous representation to the layer's output dim, then a
    tanh nonlinearity (so the null has a comparable nonlinear expansion, not just a rotation)."""
    rng = np.random.RandomState(seed + 991)
    P = rng.randn(prev.shape[1], out_dim) / np.sqrt(prev.shape[1])
    return np.tanh(prev @ P)


# --------------------------------------------------------------------- run PAULA, record activity

def record_layer_activity(net, core, sig, ticks, settle, layer_of, shuffle_time=False, seed=0,
                          feats="S"):
    """Run one sample from a freshly-reset net; return {layer -> per-neuron feature vector}.
    feats="S": per-neuron settled mean-S (legacy). feats="full": the canonical PAULA feature
    set per neuron -- [mean_S, mean_t_ref, firing_rate] -- concatenated per layer (t_ref /
    adaptive refractory is load-bearing per the paper; avg_S alone under-reads the code)."""
    neurons = list(net.network.neurons.values())
    N = len(neurons)
    W = ticks - settle
    Sbuf = np.empty((W, N), np.float32)
    full = feats == "full"
    if full:
        Tbuf = np.empty((W, N), np.float32)
        Obuf = np.zeros((W, N), np.float32)
    for t in range(ticks):
        core.send_batch_signals(sig)
        core.do_tick()
        if t >= settle:
            r = t - settle
            Sbuf[r] = [nu.S for nu in neurons]
            if full:
                Tbuf[r] = [nu.t_ref for nu in neurons]
                Obuf[r] = [1.0 if nu.O > 0 else 0.0 for nu in neurons]
    if shuffle_time:
        rng = np.random.RandomState(seed)
        for j in range(N):
            Sbuf[:, j] = Sbuf[rng.permutation(Sbuf.shape[0]), j]
    nid_list = list(net.network.neurons.keys())
    mean_s = Sbuf.mean(0)
    if full:
        mean_t = Tbuf.mean(0)
        fire_rate = Obuf.mean(0)
    per_layer = defaultdict(lambda: ([], [], []))
    for j, nid in enumerate(nid_list):
        L = layer_of[nid]
        per_layer[L][0].append(mean_s[j])
        if full:
            per_layer[L][1].append(mean_t[j])
            per_layer[L][2].append(fire_rate[j])
    out = {}
    for L, (s, tr, fr) in per_layer.items():
        if full:
            out[L] = np.concatenate([s, tr, fr]).astype(np.float32)
        else:
            out[L] = np.asarray(s, np.float32)
    return out


_G = {}


def _init_worker(net_path, dataset, signal_gain, ticks, settle, shuffle_time, norm,
                 is_test=False, freeze=False, feats="S"):
    """Each worker loads its own net once (avoids pickling the net + loguru sinks).
    is_test selects the CIFAR test split (used by readout_eval); default False = train split.
    freeze=True zeros eta_post/eta_retro so the net is a DETERMINISTIC fixed feature extractor
    (plasticity left on makes weights drift sample-to-sample -> recording is noise-dominated,
    same image gives ~uncorrelated activity, and no fixed readout can generalize)."""
    net = NetworkConfig.load_network_config(net_path, neuron_class=Neuron)
    core = NNCore(); core.neural_net = net; core.set_log_level("CRITICAL")
    if freeze:
        for nu in net.network.neurons.values():
            nu.params.eta_post = 0.0
            nu.params.eta_retro = 0.0
    meta = {n["id"]: n["metadata"].get("layer") for n in
            json.load(open(net_path)).get("neurons", [])}
    layer_of = {nid: meta.get(nid) for nid in net.network.neurons}
    _lay = infer_layers_from_metadata(net)
    in_ids, insyn = determine_input_mapping(net, _lay)
    dc = load_dataset_by_name(dataset, train=not is_test, cifar10_color_normalization_factor=norm)
    dc.signal_gain = signal_gain
    _G.update(net=net, core=core, layer_of=layer_of, in_ids=in_ids, insyn=insyn,
              dc=dc, ds=dc.dataset, ticks=ticks, settle=settle, shuffle_time=shuffle_time,
              feats=feats)


def _worker_record(i):
    g = _G
    img, _ = g["ds"][i]
    raw = np.asarray(img).ravel().astype(np.float32)
    sig = image_to_signals(img, g["in_ids"], g["insyn"], g["net"], g["dc"])
    g["net"].reset_simulation(); g["core"].state.current_tick = 0; g["net"].current_tick = 0
    pl = record_layer_activity(g["net"], g["core"], sig, g["ticks"], g["settle"],
                               g["layer_of"], g["shuffle_time"], seed=0, feats=g.get("feats", "S"))
    return i, {int(L): v for L, v in pl.items()}, raw


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True)
    ap.add_argument("--dataset", default="cifar10_color")
    ap.add_argument("--classes", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--per-class", type=int, default=40)
    ap.add_argument("--ticks", type=int, default=300)
    ap.add_argument("--settle-frac", type=float, default=0.4)
    ap.add_argument("--signal-gain", type=float, default=0.336)
    ap.add_argument("--norm", type=float, default=0.6, help="cifar10_color normalization factor")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--shuffle-time", action="store_true", help="temporal-shuffle control")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default="foveation_results/paula_vision/sep")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--freeze", action="store_true", help="freeze plasticity (fixed feature extractor)")
    ap.add_argument("--feats", default="S", choices=["S", "full"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    classes = [int(x) for x in a.classes.split(",")]
    seeds = [int(s) for s in a.seeds.split(",")]
    settle = int(a.ticks * a.settle_frac)

    net = NetworkConfig.load_network_config(a.net, neuron_class=Neuron)
    core = NNCore(); core.neural_net = net; core.set_log_level("CRITICAL")
    layer_of = {nid: nu.metadata.get("layer") if hasattr(nu, "metadata") else
                net.network.neurons[nid].metadata.get("layer")
                for nid, nu in net.network.neurons.items()}
    # metadata may live on the config, not the neuron; fall back to config record
    if all(v is None for v in layer_of.values()):
        meta = {n["id"]: n["metadata"].get("layer") for n in
                json.load(open(a.net)).get("neurons", [])}
        layer_of = {nid: meta.get(nid) for nid in net.network.neurons}
    layers_sorted = sorted({L for L in layer_of.values() if L is not None})
    _lay = infer_layers_from_metadata(net)
    in_ids, insyn = determine_input_mapping(net, _lay)
    dc = load_dataset_by_name(a.dataset); dc.signal_gain = a.signal_gain
    ds = dc.dataset

    # sample a fixed balanced pool
    rng = np.random.RandomState(0)
    by_c = defaultdict(list)
    for i in range(len(ds)):
        _, lab = ds[i]
        lab = int(lab)
        if lab in classes and len(by_c[lab]) < a.per_class:
            by_c[lab].append(i)
        if all(len(by_c[c]) >= a.per_class for c in classes):
            break
    idxs = [i for c in classes for i in by_c[c]]
    labels = np.asarray([int(ds[i][1]) for i in idxs])
    print(f"[{a.tag}] net={a.net} N={len(net.network.neurons)} layers={layers_sorted} "
          f"samples={len(idxs)} ({a.per_class}/class) ticks={a.ticks} seeds={seeds}", flush=True)

    # record per-layer activity for every sample (fresh reset per sample), parallel over workers
    t0 = time.time()
    rows = {L: {} for L in layers_sorted}
    raw = {}
    init_args = (a.net, a.dataset, a.signal_gain, a.ticks, settle, a.shuffle_time, a.norm,
                 False, a.freeze, a.feats)
    if a.workers > 1:
        with mp.Pool(a.workers, initializer=_init_worker, initargs=init_args) as pool:
            for done, (i, pl, r) in enumerate(pool.imap_unordered(_worker_record, idxs, chunksize=1)):
                raw[i] = r
                for L in layers_sorted:
                    rows[L][i] = pl.get(L, np.zeros(0, np.float32))
                if (done + 1) % 50 == 0:
                    print(f"  recorded {done + 1}/{len(idxs)} ({time.time() - t0:.0f}s)", flush=True)
    else:
        _init_worker(*init_args)
        for done, i in enumerate(idxs):
            _, pl, r = _worker_record(i)
            raw[i] = r
            for L in layers_sorted:
                rows[L][i] = pl.get(L, np.zeros(0, np.float32))
            if (done + 1) % 50 == 0:
                print(f"  recorded {done + 1}/{len(idxs)} ({time.time() - t0:.0f}s)", flush=True)
    reps = {"pixels": np.stack([raw[i] for i in idxs])}
    for L in layers_sorted:
        reps[f"L{L}"] = np.stack([rows[L][i] for i in idxs])

    # separability ladder + per-layer add + random-projection null
    order = ["pixels"] + [f"L{L}" for L in layers_sorted]
    result = {"tag": a.tag, "net": a.net, "n": len(idxs), "classes": classes,
              "ladder": {}, "adds": {}}
    print(f"\n[{a.tag}] separability ladder (closed-form, chance={1/len(classes):.3f}):", flush=True)
    for name in order:
        accs = defaultdict(list)
        for s in seeds:
            for k, v in separability(reps[name], labels, seed=s).items():
                accs[k].append(v)
        result["ladder"][name] = {k: [round(float(np.mean(v)), 4),
                                      round(float(np.std(v)), 4)] for k, v in accs.items()}
        print(f"  {name:8s} " + "  ".join(f"{k}={np.mean(v):.3f}" for k, v in accs.items()), flush=True)

    print(f"\n[{a.tag}] per-layer ADD (out - in) and vs random-projection null:", flush=True)
    for i in range(1, len(order)):
        prev, cur = reps[order[i - 1]], reps[order[i]]
        din = result["ladder"][order[i - 1]]
        dout = result["ladder"][order[i]]
        # random-projection null of prev -> cur dim
        null_accs = defaultdict(list)
        for s in seeds:
            nullrep = random_projection_null(prev, cur.shape[1], s)
            for k, v in separability(nullrep, labels, seed=s).items():
                null_accs[k].append(v)
        adds = {}
        for k in ("ncc", "lda", "ridge"):
            add = dout[k][0] - din[k][0]
            beat_null = dout[k][0] - float(np.mean(null_accs[k]))
            adds[k] = dict(add=round(add, 4), out=dout[k][0], in_=din[k][0],
                           null=round(float(np.mean(null_accs[k])), 4), beat_null=round(beat_null, 4))
        result["adds"][f"{order[i-1]}->{order[i]}"] = adds
        flag = "PASS" if (adds["ridge"]["add"] > 0 and adds["ridge"]["beat_null"] > 0) else "FAIL"
        print(f"  {order[i-1]:8s} -> {order[i]:8s}  ridge add={adds['ridge']['add']:+.3f} "
              f"(out {adds['ridge']['out']:.3f} vs in {adds['ridge']['in_']:.3f}) "
              f"vs_null={adds['ridge']['beat_null']:+.3f}  [{flag}]", flush=True)

    result["secs"] = round(time.time() - t0)
    json.dump(result, open(os.path.join(a.out, f"{a.tag}.json"), "w"), indent=2)
    print(f"\n[{a.tag}] wrote {os.path.join(a.out, f'{a.tag}.json')} ({result['secs']}s)", flush=True)


if __name__ == "__main__":
    main()
