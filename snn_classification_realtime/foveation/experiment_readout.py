"""#4/#5 — Encoder or internal representation? Readout comparison.

Directive #5: use the trained SNN decoder in addition to kNN/linear on PAULA
features. Directive #4: is PAULA JUST a high-dim dynamical input encoder, or does
it build internal representations?

We hold ONE perception substrate (single full-image conv layer, learning ON,
regime fixed by per-layer threshold homeostasis) and read its response to CIFAR
images four ways, all against the same yardstick (pixel-kNN ~0.21):

  1. pixel-kNN            — raw grayscale pixels (the "no substrate" baseline).
  2. PAULA settled kNN    — kNN on the settled activity signature.
  3. PAULA settled linear — logistic regression on the same.
  4. PAULA temporal SNN   — the stateful 4-layer LIF decoder reading the full
                            [T,N] spike stream (the strong temporal readout).

Plus a control that speaks directly to #4:
  5. SNN on TIME-SHUFFLED streams — each stream's tick order is permuted, which
     destroys temporal dynamics but preserves the per-tick spike statistics.
     SNN(real) >> SNN(shuffled)  => the DYNAMICS carry class info the static
     snapshot misses (temporal computation). SNN(real) ~ SNN(shuffled) => the
     substrate is a static nonlinear encoder; time order adds nothing.

Reading:
  - If nothing beats ~0.21, PAULA is (at most) a re-encoder of the pixels.
  - If temporal SNN >> settled linear AND >> shuffled, PAULA does temporal
    computation the static readout can't see — evidence of dynamical structure.
  - Whether that is "internal representation" in the strong sense (memory of
    inputs no longer present) is the separate memory probe (experiment_memory).
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
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
    PerceptionNetwork, build_fovea_network_json,
)
from snn_classification_realtime.foveation.experiment_paula_capacity import (
    calibrate_layer_homeostasis,
)
from snn_classification_realtime.snn_trainer.model import SNNClassifier


def collect_streams(perc, whole, ds, idxs, ticks, last):
    """Return O streams [M,T,N], settled sigs [M,4N], weight snapshots [M,Wdim],
    labels [M]. The weight snapshot is the plastic (slow) state after exposure —
    the direct test of whether learning writes class structure into the weights."""
    streams, sett, wts, y = [], [], [], []
    n = perc.num_neurons
    for i in tqdm(idxs, desc="paula", leave=False):
        image, lbl = ds[i]
        perc.reset()
        sig = perc.patch_to_signals(whole.crop(image))
        O = np.zeros((ticks, n), np.float32)
        Sl = np.zeros(n, np.float32); Tl = np.zeros(n, np.float32)
        for t in range(ticks):
            st = perc.step(sig)
            O[t] = st.O
            if t >= ticks - last:
                Sl += st.S; Tl += st.t_ref
        streams.append(O)
        sett.append(np.concatenate([O.mean(0), O[ticks - last:].mean(0),
                                    Sl / last, Tl / last]))
        wts.append(perc.efficacy_vector())
        y.append(int(lbl))
    return (np.stack(streams), np.stack(sett).astype(np.float32),
            np.stack(wts).astype(np.float32), np.array(y))


def probe_static(Xtr, ytr, Xte, yte, pca_dim):
    sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    d = min(pca_dim, Xtr.shape[1], Xtr.shape[0] - 1)
    pca = PCA(n_components=d).fit(Xtr); Ztr, Zte = pca.transform(Xtr), pca.transform(Xte)
    knn = KNeighborsClassifier(n_neighbors=5, metric="cosine").fit(Ztr, ytr)
    lr = LogisticRegression(max_iter=400, C=0.5).fit(Ztr, ytr)
    return float(knn.score(Zte, yte)), float(lr.score(Zte, yte))


def train_snn(Str, ytr, Ste, yte, hidden, epochs, device, seed=0):
    """Train the stateful LIF decoder on [M,T,N] spike streams."""
    torch.manual_seed(seed)
    M, T, N = Str.shape
    net = SNNClassifier(N, hidden, int(ytr.max()) + 1, beta=0.9).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    Xtr = torch.tensor(Str, device=device); Ytr = torch.tensor(ytr, device=device)
    Xte = torch.tensor(Ste, device=device)
    bs = 64
    for _ in range(epochs):
        net.train(); perm = torch.randperm(M)
        for b in range(0, M, bs):
            idx = perm[b:b + bs]; xb = Xtr[idx]; yb = Ytr[idx]
            m1 = m2 = m3 = m4 = None; spk_sum = 0.0
            for t in range(T):
                spk4, m1, m2, m3, m4 = net(xb[:, t], m1, m2, m3, m4)
                spk_sum = spk_sum + spk4
            opt.zero_grad(); crit(spk_sum, yb).backward(); opt.step()
    net.eval()
    with torch.no_grad():
        m1 = m2 = m3 = m4 = None; spk_sum = 0.0
        for t in range(T):
            spk4, m1, m2, m3, m4 = net(Xte[:, t], m1, m2, m3, m4)
            spk_sum = spk_sum + spk4
        pred = spk_sum.argmax(1).cpu().numpy()
    return float((pred == yte).mean())


def run(args):
    torch.manual_seed(0); np.random.seed(0)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ds_cfg = load_dataset_by_name(args.dataset_name, train=True)
    ds = ds_cfg.dataset
    img0, _ = ds[0]; channels, H, W = img0.shape
    whole = Fovea(image_h=H, image_w=W, size=H)

    by_label = {i: [] for i in range(10)}; need = args.train_per_class + args.test_per_class
    for i in range(len(ds)):
        _, l = ds[i]; l = int(l)
        if len(by_label[l]) < need: by_label[l].append(i)
        if all(len(by_label[c]) >= need for c in range(10)): break
    tr_idx, te_idx = [], []
    for c in range(10):
        tr_idx += by_label[c][:args.train_per_class]
        te_idx += by_label[c][args.train_per_class:need]
    cal_idx = tr_idx[::max(1, len(tr_idx) // 8)][:8]

    # pixel-kNN baseline (same grayscale pixels PAULA sees)
    def pixels(idxs):
        return np.stack([ds[i][0].numpy().ravel() for i in idxs])
    Xp_tr, Xp_te = pixels(tr_idx), pixels(te_idx)
    ytr = np.array([int(ds[i][1]) for i in tr_idx]); yte = np.array([int(ds[i][1]) for i in te_idx])
    pk = KNeighborsClassifier(n_neighbors=5).fit(Xp_tr, ytr)
    pixel_knn = float(pk.score(Xp_te, yte))

    layers = [{"type": "conv", "kernel_size": 4, "stride": 2, "filters": args.filters}]
    net_path = os.path.join(args.output_dir, f"readout_net_{channels}x{H}.json")
    os.makedirs(args.output_dir, exist_ok=True)
    build_fovea_network_json(net_path, channels=channels, size=H, layers=layers, seed=args.seed)
    perc = PerceptionNetwork(net_path, ds_cfg, ablation=args.ablation)  # learning ON
    perc.set_plasticity(mode=args.plasticity_mode, lr_error=args.lr_error,
                        weight_decay_tau=args.weight_decay_tau,
                        weight_baseline=args.weight_baseline)
    part = calibrate_layer_homeostasis(perc, whole, ds, ds_cfg, cal_idx, args.ticks,
                                       args.last, args.target_participation)
    print(f"{perc.num_neurons} neurons | plasticity {args.plasticity_mode} "
          f"(tau {args.weight_decay_tau}) | gain {ds_cfg.signal_gain} | "
          f"participation {[round(part[l],3) for l in perc.layer_indices]} | device {device}")

    Str, sett_tr, wtr, ytr2 = collect_streams(perc, whole, ds, tr_idx, args.ticks, args.last)
    Ste, sett_te, wte, yte2 = collect_streams(perc, whole, ds, te_idx, args.ticks, args.last)
    assert (ytr2 == ytr).all() and (yte2 == yte).all()

    knn_s, lin_s = probe_static(sett_tr, ytr, sett_te, yte, args.pca_dim)
    knn_w, lin_w = probe_static(wtr, ytr, wte, yte, args.pca_dim)
    snn_real = train_snn(Str, ytr, Ste, yte, args.hidden, args.epochs, device)
    # time-shuffle control (permute tick order per stream, same permutation stats)
    rng = np.random.default_rng(0)
    def shuffle_time(X):
        Y = X.copy()
        for m in range(len(Y)): Y[m] = Y[m][rng.permutation(Y.shape[1])]
        return Y
    snn_shuf = train_snn(shuffle_time(Str), ytr, shuffle_time(Ste), yte,
                         args.hidden, args.epochs, device)

    res = {"config": vars(args), "num_neurons": perc.num_neurons,
           "plasticity_mode": args.plasticity_mode,
           "weight_decay_tau": args.weight_decay_tau,
           "pixel_knn": pixel_knn, "paula_settled_knn": knn_s,
           "paula_settled_linear": lin_s, "paula_temporal_snn": snn_real,
           "paula_temporal_snn_timeshuffled": snn_shuf,
           "paula_weight_knn": knn_w, "paula_weight_linear": lin_w,
           "train_n": len(ytr), "test_n": len(yte)}
    out = os.path.join(args.output_dir,
                       f"readout_{args.plasticity_mode}_{int(time.time())}.json")
    json.dump(res, open(out, "w"), indent=2)
    print(f"\n=== Readout comparison [{args.plasticity_mode}, tau={args.weight_decay_tau}] "
          f"(chance 0.10) ===")
    print(f"  pixel-kNN (baseline)          {pixel_knn:.3f}")
    print(f"  PAULA settled kNN             {knn_s:.3f}")
    print(f"  PAULA settled linear          {lin_s:.3f}")
    print(f"  PAULA temporal SNN            {snn_real:.3f}")
    print(f"  PAULA temporal SNN (shuffled) {snn_shuf:.3f}")
    print(f"  PAULA WEIGHT kNN              {knn_w:.3f}   (slow plastic state)")
    print(f"  PAULA WEIGHT linear          {lin_w:.3f}")
    print(f"Saved: {out}")
    return res


def main():
    p = argparse.ArgumentParser(description="Readout comparison: kNN/linear vs SNN")
    p.add_argument("--dataset-name", default="cifar10_grayscale")
    p.add_argument("--filters", type=int, default=4)
    p.add_argument("--ticks", type=int, default=40)
    p.add_argument("--last", type=int, default=15)
    p.add_argument("--train-per-class", type=int, default=40)
    p.add_argument("--test-per-class", type=int, default=20)
    p.add_argument("--target-participation", type=float, default=0.1)
    p.add_argument("--pca-dim", type=int, default=100)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--plasticity-mode", default="legacy_multiplicative",
                   choices=["legacy_multiplicative", "error_correcting"])
    p.add_argument("--lr-error", type=float, default=0.05)
    p.add_argument("--weight-decay-tau", type=float, default=0.0)
    p.add_argument("--weight-baseline", type=float, default=1.0)
    p.add_argument("--ablation", default="none")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="foveation_results")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
