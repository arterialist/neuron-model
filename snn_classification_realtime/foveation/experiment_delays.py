"""EXP12: do propagation delays (the spatial aspect) meaningfully change PAULA?

Discovery that motivates this: in the current sim, delay lives in TWO places and
neither is a rich spatial embedding:
  - Axonal / network delay is drawn per-signal as randint(MIN,MAX) with
    MIN=MAX=1 -> every inter-neuron signal takes exactly 1 tick. The stored
    axon-terminal distances are ignored for timing.
  - Dendritic delay is real and per-connection: a synapse's signal reaches the
    hillock after `distance_to_hillock` ticks (2..8 as built) AND is attenuated
    by delta_decay**distance. So distance couples DELAY and ATTENUATION.

So "the spatial aspect" is currently ~2-8 ticks of dendritic delay with a
coupled exponential decay, and 1 tick everywhere else. This experiment sweeps
delay and measures whether it changes:
  - classification (decodability, settled readout)  -> accuracy proxy
  - temporal decodability (time-binned readout)      -> does delay create a
    decodable TEMPORAL code (polychronization-style)?
  - representation dimensionality (participation ratio) -> richness/variability
  - regime shift (pre-calibration participation)

Fairness: per-layer threshold homeostasis holds participation at target for
EVERY condition, so we compare delay, not drive. delta_decay=1 variants isolate
pure delay from attenuation.

No HDF5; compact JSON only.
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

import neuron.network as netmod  # to patch axonal delay constants
from snn_classification_realtime.activity_dataset_builder.vision_datasets import (
    load_dataset_by_name,
)
from snn_classification_realtime.foveation.fovea import Fovea
from snn_classification_realtime.foveation.perception import (
    PerceptionNetwork,
    build_fovea_network_json,
)
from snn_classification_realtime.foveation.experiment_paula_capacity import (
    calibrate_layer_homeostasis,
    measure_participation,
)

# Two-layer strided conv, ~871 neurons: multi-hop so delays actually compound.
DELAY_ARCH = [{"type": "conv", "kernel_size": 4, "stride": 2, "filters": 3},
              {"type": "conv", "kernel_size": 3, "stride": 2, "filters": 4}]

# policy -> (dendrite mode, lo, hi, delta_decay override or None, axon (min,max), ticks)
POLICIES = {
    "min1":        ("const", 1, 1, None, (1, 1), 50),   # ~no dendritic delay/decay
    "current":     ("keep", 0, 0, None, (1, 1), 60),    # 2..8 as built (baseline)
    "const8":      ("const", 8, 8, None, (1, 1), 70),   # uniform large delay+decay
    "wide16":      ("uniform", 1, 16, None, (1, 1), 90),
    "wide16_nodecay": ("uniform", 1, 16, 1.0, (1, 1), 90),  # pure delay, no attenuation
    "axon1to10":   ("keep", 0, 0, None, (1, 10), 60),   # network jitter on baseline dendrite
}


def apply_delay_policy(in_path, out_path, mode, lo, hi, delta_decay, seed):
    rng = np.random.default_rng(seed)
    d = json.load(open(in_path))
    for sp in d["synaptic_points"]:
        if sp.get("type") == "postsynaptic":
            if mode == "keep":
                continue
            if mode == "const":
                sp["distance_to_hillock"] = int(lo)
            elif mode == "uniform":
                sp["distance_to_hillock"] = int(rng.integers(lo, hi + 1))
    if delta_decay is not None:
        for nrn in d["neurons"]:
            nrn["params"]["delta_decay"] = float(delta_decay)
    json.dump(d, open(out_path, "w"))
    return out_path


def run_image(perc, whole, image, ticks, last, nbins):
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
    settled = np.concatenate([rate_all, rate_late, Slate / last, Tlate / last])
    # temporal signature: per-neuron mean rate in nbins equal windows
    edges = np.linspace(0, ticks, nbins + 1, dtype=int)
    temporal = np.concatenate([O[edges[b]:edges[b + 1]].mean(0) for b in range(nbins)])
    part = float(rate_late.mean())
    return settled.astype(np.float32), temporal.astype(np.float32), part


def participation_ratio(X):
    """Dimensionality of the representation: (sum eig)^2 / sum(eig^2) of cov."""
    Xc = X - X.mean(0, keepdims=True)
    cov = (Xc.T @ Xc) / max(1, len(Xc) - 1)
    ev = np.linalg.eigvalsh(cov)
    ev = np.clip(ev, 0, None)
    s1 = ev.sum()
    s2 = (ev ** 2).sum()
    return float(s1 * s1 / s2) if s2 > 0 else 0.0


def probe(Xtr, ytr, Xte, yte, pca_dim):
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    d = min(pca_dim, Xtr.shape[1], Xtr.shape[0] - 1)
    pca = PCA(n_components=d).fit(Xtr)
    Ztr, Zte = pca.transform(Xtr), pca.transform(Xte)
    knn = KNeighborsClassifier(n_neighbors=5, metric="cosine").fit(Ztr, ytr)
    lr = LogisticRegression(max_iter=400, C=0.5).fit(Ztr, ytr)
    return float(knn.score(Zte, yte)), float(lr.score(Zte, yte))


def collect(perc, whole, ds, idxs, ticks, last, nbins):
    St, Tp, y = [], [], []
    for i in tqdm(idxs, desc="sig", leave=False):
        image, lbl = ds[i]
        s, t, _ = run_image(perc, whole, image, ticks, last, nbins)
        St.append(s); Tp.append(t); y.append(int(lbl))
    return np.stack(St), np.stack(Tp), np.array(y)


def run(args):
    torch.manual_seed(0); np.random.seed(0)
    mode, lo, hi, dd, (amin, amax), ticks = POLICIES[args.policy]
    if args.ticks:
        ticks = args.ticks
    # Patch axonal (network) delay constants for THIS process.
    netmod.MIN_CONNECTION_SIGNAL_TRAVEL_TICKS = amin
    netmod.MAX_CONNECTION_SIGNAL_TRAVEL_TICKS = amax

    ds_cfg = load_dataset_by_name(args.dataset_name, train=True)
    ds = ds_cfg.dataset
    img0, _ = ds[0]
    channels, H, W = img0.shape
    whole = Fovea(image_h=H, image_w=W, size=H)

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

    base = build_fovea_network_json(
        os.path.join(args.output_dir, f"delay_base_{channels}x{H}.json"),
        channels=channels, size=H, layers=DELAY_ARCH, seed=args.seed)
    net_path = os.path.join(args.output_dir, f"delay_{args.policy}_{channels}x{H}.json")
    apply_delay_policy(base, net_path, mode, lo, hi, dd, args.seed)
    perc = PerceptionNetwork(net_path, ds_cfg, ablation=args.ablation)

    # Regime BEFORE calibration (how much delay shifts drive), then hold fixed.
    ds_cfg.signal_gain = 1.0
    pre = measure_participation(perc, whole, ds, cal_idx, ticks, args.last)
    post = calibrate_layer_homeostasis(perc, whole, ds, ds_cfg, cal_idx, ticks,
                                       args.last, args.target_participation)
    print(f"[{args.policy}] {perc.num_neurons} neurons | ticks {ticks} | axon [{amin},{amax}] "
          f"| dd {dd} | pre-part {[round(pre[l],3) for l in perc.layer_indices]} "
          f"-> post {[round(post[l],3) for l in perc.layer_indices]}")

    Str, Ttr, ytr = collect(perc, whole, ds, train_idx, ticks, args.last, args.nbins)
    Ste, Tte, yte = collect(perc, whole, ds, test_idx, ticks, args.last, args.nbins)
    knn_s, lin_s = probe(Str, ytr, Ste, yte, args.pca_dim)
    knn_t, lin_t = probe(Ttr, ytr, Tte, yte, args.pca_dim)
    pr = participation_ratio(Str)

    res = {
        "policy": args.policy, "config": vars(args),
        "mode": mode, "dend_lo": lo, "dend_hi": hi, "delta_decay": dd,
        "axon": [amin, amax], "ticks": ticks, "num_neurons": perc.num_neurons,
        "pre_participation": pre, "post_participation": post,
        "knn_settled": knn_s, "linear_settled": lin_s,
        "knn_temporal": knn_t, "linear_temporal": lin_t,
        "participation_ratio": pr,
    }
    out = os.path.join(args.output_dir,
                       f"delays_{args.policy}_{int(time.time())}.json")
    json.dump(res, open(out, "w"), indent=2)
    print(f"[{args.policy}] settled kNN {knn_s:.3f} lin {lin_s:.3f} | "
          f"temporal kNN {knn_t:.3f} lin {lin_t:.3f} | dim(PR) {pr:.1f} "
          f"(chance 0.10)\nSaved: {out}")
    return res


def main():
    p = argparse.ArgumentParser(description="EXP12 propagation-delay sweep")
    p.add_argument("--policy", required=True, choices=list(POLICIES))
    p.add_argument("--dataset-name", default="cifar10_grayscale")
    p.add_argument("--ticks", type=int, default=0, help="0 = policy default")
    p.add_argument("--last", type=int, default=20)
    p.add_argument("--nbins", type=int, default=6)
    p.add_argument("--train-per-class", type=int, default=15)
    p.add_argument("--test-per-class", type=int, default=8)
    p.add_argument("--cal-images", type=int, default=6)
    p.add_argument("--target-participation", type=float, default=0.1)
    p.add_argument("--pca-dim", type=int, default=80)
    p.add_argument("--ablation", default="none")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="foveation_results")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
