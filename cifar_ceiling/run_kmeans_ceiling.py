"""EXPERIMENT 1 -- the true fixed-feature ceiling on CIFAR10 (Coates & Ng 2011 pipeline).

WHY THIS EXISTS. Our quick pass (`ceiling_km.py`) got 63.1% with a K=512 k-means dictionary,
against a published 71.0% at K=400 / 79.6% at K=1600. A 10-point deficit at matched K means
the INSTRUMENT is broken, not that the ceiling is low. Sweeping K on a broken pipeline would
produce a confidently wrong ceiling. So this script is staged: validate the instrument against
a published reference point FIRST, find which lever is broken, and only then read the ceiling.

Suspected defect (identified by re-reading ceiling_km.py, to be confirmed by --stage ablate):
pixels were in [0,1] (ToTensor) while Coates' constants assume [0,255]. At [0,1] the patch
variance is ~0.01, so contrast normalization computes (p - mean)/sqrt(0.01 + 10) ~= a constant
divisor of 3.16 -- i.e. contrast normalization was silently OFF and ZCA (eps=0.1) was
over-damped. Secondary suspects: MiniBatchKMeans instead of Lloyd; a fixed readout C=0.01 with
no sweep. The ablation stage measures each rather than assuming.

STAGES
  --stage bench      measure encode throughput, project full-run cost (no extrapolation by hand)
  --stage calibrate  canonical config at K=400 -> must land near the published 71.0%
  --stage ablate     one-factor-at-a-time from canonical at K=400: which lever costs what
  --stage sweep      the ceiling curve, K in {400,800,1600,3200}, validated config only

PRE-REGISTERED READING OF THE RESULT
  H1 (instrument valid): calibrate lands within +/-2 points of 71.0% at K=400.
      If it does NOT, the K-sweep is not run; the ablation identifies the defect first.
  H2 (ceiling): with a valid instrument, test accuracy rises with K and saturates near
      ~79-80% at K=1600 -- i.e. the fixed-feature ceiling is ~80%, not ~65%.
  H2-null: the curve saturates at ~65% even with a validated pipeline -> the published
      numbers do not reproduce here, and our substrate's 62% really is at the ceiling.
  Either outcome is informative and is reported as measured. The published curve is a
  calibration target for the instrument, NOT a result we are trying to reproduce as a finding.

KEY CONTROL (arm `randpatch`): the dictionary is random raw whitened patches -- no learning at
all. Coates reports this nearly matches k-means under triangle encoding. If it replicates, then
dictionary LEARNING is not what buys the points, the encoding+pooling is -- which reframes the
question for PAULA: a random substrate is not obviously the problem.

GUARDS. Dictionary + preprocessing + feature standardization are fit on the TRAIN split only.
Readout C is chosen on a disjoint VAL split. The TEST set is scored exactly once per config,
at the end. Every config writes its own JSON; nothing is aggregated in-script.

CLI-only; run from neuron-model/. See EXPERIMENT_DESIGN_cifar_ceiling.md.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from cifar_ceiling.common import (
    COATES_NG_2011_REFERENCE, Preproc, bootstrap_ci, encode_dataset, eval_test, fit_readout,
    kmeans_lloyd, load_cifar, normalize_dict, pick_device, random_patch_dict, sample_patches,
    train_val_split, experiment_protocol, read_cached_record, write_record,
)

# The canonical Coates & Ng configuration. Ablations are deltas from THIS.
CANONICAL = dict(
    pixel_scale=255.0, patch=6, stride=1, cn_on=True, cn_eps=10.0, zca_on=True, zca_eps=0.1,
    dict_rule="kmeans", kmeans_iters=50, n_patches=400_000, encoder="triangle",
)

# One-factor-at-a-time ablation grid. Each entry is (label, {overrides}). `scale1` is the
# suspected defect in our 63% run; `minibatch_proxy` and `no_c_sweep` are its other deviations.
ABLATIONS = [
    ("canonical",       {}),
    ("scale1",          dict(pixel_scale=1.0)),          # the suspected bug: [0,1] pixels
    ("no_contrast_norm", dict(cn_on=False)),
    ("no_zca",          dict(zca_on=False)),
    ("zca_eps_0.01",    dict(zca_eps=0.01)),
    ("zca_eps_1.0",     dict(zca_eps=1.0)),
    ("kmeans_iters_10", dict(kmeans_iters=10)),          # proxy for under-converged MiniBatchKMeans
    ("randpatch_dict",  dict(dict_rule="randpatch")),    # Coates' no-learning control
    ("encoder_hard",    dict(encoder="hard")),
    ("encoder_thresh",  dict(encoder="thresh")),
    ("stride_2",        dict(stride=2)),
    ("patches_50k",     dict(n_patches=50_000)),
]


def run_config(cfg: dict, K: int, seed: int, a, device, cache: dict) -> dict:
    """One full pipeline: dictionary -> features -> readout -> single test score."""
    t_all = time.time()
    scale = cfg["pixel_scale"]
    if scale not in cache:
        cache[scale] = load_cifar(a.data_root, scale)
    Xtr_all, ytr_all, Xte, yte = cache[scale]

    if a.train_n and a.train_n < len(Xtr_all):
        sub = np.random.RandomState(seed).choice(len(Xtr_all), a.train_n, replace=False)
        Xtr_all, ytr_all = Xtr_all[sub], ytr_all[sub]
    tr_idx, va_idx = train_val_split(len(Xtr_all), a.val_frac, seed)

    # --- preprocessing + dictionary: TRAIN split only ---
    t0 = time.time()
    pw = sample_patches(Xtr_all[tr_idx], cfg["n_patches"], cfg["patch"], seed)
    pre = Preproc(cfg["patch"], cfg["cn_eps"], cfg["cn_on"], cfg["zca_eps"], cfg["zca_on"]).fit(pw)
    pz = pre.apply(pw)
    t_pre = time.time() - t0

    t0 = time.time()
    if cfg["dict_rule"] == "kmeans":
        D = kmeans_lloyd(pz, K, cfg["kmeans_iters"], seed, device, verbose=a.verbose)
    elif cfg["dict_rule"] == "randpatch":
        D = random_patch_dict(pz, K, seed)
    else:
        raise ValueError(cfg["dict_rule"])
    D = normalize_dict(D)
    t_dict = time.time() - t0

    # --- features ---
    t0 = time.time()
    ekw = dict(pre=pre, device=device, P=cfg["patch"], stride=cfg["stride"],
               mode=cfg["encoder"], batch=a.encode_batch, verbose=a.verbose)
    Ftr = encode_dataset(Xtr_all[tr_idx], D, **ekw)
    Fva = encode_dataset(Xtr_all[va_idx], D, **ekw)
    Fte = encode_dataset(Xte, D, **ekw)
    t_enc = time.time() - t0

    # --- readout (C on VAL) then a single TEST read ---
    t0 = time.time()
    ro = fit_readout(Ftr, ytr_all[tr_idx], Fva, ytr_all[va_idx], device,
                     C_grid=a.c_grid, epochs=a.readout_epochs, seed=seed, verbose=a.verbose)
    test_acc = eval_test(ro, Fte, yte, device)
    t_ro = time.time() - t0

    return dict(K=K, seed=seed, cfg=cfg, test_acc=round(test_acc * 100, 3),
                val_acc=round(ro["val_acc"] * 100, 3), C=ro["C"], n_train=int(len(tr_idx)),
                n_val=int(len(va_idx)), feat_dim=int(Ftr.shape[1]),
                secs=dict(pre=round(t_pre), dict=round(t_dict), encode=round(t_enc),
                          readout=round(t_ro), total=round(time.time() - t_all)))


def stage_bench(a, device):
    """Measure, don't extrapolate: time the real encode on a small slice, project from THAT."""
    Xtr, ytr, Xte, yte = load_cifar(a.data_root, CANONICAL["pixel_scale"])
    pw = sample_patches(Xtr, 50_000, CANONICAL["patch"], 0)
    pre = Preproc(CANONICAL["patch"], CANONICAL["cn_eps"], True, CANONICAL["zca_eps"], True).fit(pw)
    pz = pre.apply(pw)
    n_bench = 200
    print(f"bench on {n_bench} images, device={device}\n")
    for K in a.k_sweep:
        D = normalize_dict(random_patch_dict(pz, K, 0))
        t0 = time.time()
        encode_dataset(Xtr[:n_bench], D, pre, device, CANONICAL["patch"], 1, "triangle",
                       batch=a.encode_batch)
        dt = time.time() - t0
        per_img = dt / n_bench
        total = per_img * (len(Xtr) + len(Xte))
        feat_gb = (len(Xtr) + len(Xte)) * 4 * K * 4 / 1e9
        print(f"  K={K:<5d} {per_img * 1000:6.1f} ms/img -> encode all 60k = {total / 60:5.1f} min"
              f"   feature RAM = {feat_gb:.2f} GB", flush=True)
    print("\n(add k-means fit + readout on top; those are measured per-run in the JSON records)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["bench", "calibrate", "ablate", "sweep"], required=True)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", default="cifar_ceiling/results/kmeans")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--k", type=int, default=400, help="K for calibrate/ablate stages")
    ap.add_argument("--k-sweep", default="400,800,1600,3200")
    ap.add_argument("--ablations", default="", help="comma list; default = all")
    ap.add_argument("--train-n", type=int, default=0, help="subsample TRAIN (0 = all 50k)")
    ap.add_argument("--n-patches", type=int, default=0,
                    help="override dictionary-learning patch count (0 = canonical 400k)")
    ap.add_argument("--kmeans-iters", type=int, default=0, help="override k-means iters (0 = canonical)")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--encode-batch", type=int, default=64)
    ap.add_argument("--readout-epochs", type=int, default=60)
    ap.add_argument("--c-grid", default="3e-4,1e-3,3e-3,1e-2,3e-2,1e-1")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    a.k_sweep = [int(x) for x in a.k_sweep.split(",")]
    a.c_grid = [float(x) for x in a.c_grid.split(",")]
    seeds = [int(s) for s in a.seeds.split(",")]
    device = pick_device(a.device)
    os.makedirs(a.out, exist_ok=True)
    print(f"device={device} stage={a.stage} seeds={seeds}\n", flush=True)

    if a.stage == "bench":
        stage_bench(a, device)
        return

    cache: dict = {}
    # CLI budget overrides sit UNDER the ablation overrides, so an ablation always wins over
    # them (e.g. --n-patches must not clobber the `patches_50k` ablation it is being compared to)
    cli_ov = {}
    if a.n_patches:
        cli_ov["n_patches"] = a.n_patches
    if a.kmeans_iters:
        cli_ov["kmeans_iters"] = a.kmeans_iters
    base = {**CANONICAL, **cli_ov}
    if cli_ov:
        print(f"NON-CANONICAL base config (budget overrides): {cli_ov}\n", flush=True)

    if a.stage == "calibrate":
        jobs = [("canonical", dict(base), a.k)]
    elif a.stage == "ablate":
        want = [x for x in a.ablations.split(",") if x] or [n for n, _ in ABLATIONS]
        jobs = [(n, {**base, **ov}, a.k) for n, ov in ABLATIONS if n in want]
    else:
        jobs = [(f"sweep_K{K}", dict(base), K) for K in a.k_sweep]

    if a.stage == "sweep":
        calibration_protocol = experiment_protocol(a, name="canonical", cfg=base, K=a.k, seed=seeds[0])
        calibration_path = os.path.join(a.out, f"canonical_K{a.k}_s{seeds[0]}.json")
        if not os.path.exists(calibration_path):
            ap.error("sweep requires a retained canonical calibration in --out; run --stage calibrate first")
        calibration = read_cached_record(calibration_path, calibration_protocol)
        reference = COATES_NG_2011_REFERENCE.get(a.k)
        if base != CANONICAL or a.train_n or reference is None or abs(calibration["test_acc"] - reference) > 2.0:
            ap.error("sweep requires a full-data canonical calibration within the published tolerance")

    records = []
    for name, cfg, K in jobs:
        accs = []
        for seed in seeds:
            tag = f"{name}_K{K}_s{seed}"
            path = os.path.join(a.out, f"{tag}.json")
            protocol = experiment_protocol(a, name=name, cfg=cfg, K=K, seed=seed)
            if os.path.exists(path):
                r = read_cached_record(path, protocol)
                print(f"[{tag}] cached test={r['test_acc']:.2f}%", flush=True)
            else:
                print(f"[{tag}] running...", flush=True)
                r = run_config(cfg, K, seed, a, device, cache)
                r["name"] = name
                r = write_record(path, r, protocol)
                print(f"[{tag}] test={r['test_acc']:.2f}% val={r['val_acc']:.2f}% "
                      f"C={r['C']:g} ({r['secs']['total']}s)", flush=True)
            accs.append(r["test_acc"])
            records.append(r)
        mean, (lo, hi) = bootstrap_ci(accs)
        ref = COATES_NG_2011_REFERENCE.get(K)
        refs = f"  published@K={K}: {ref:.1f}%  delta {mean - ref:+.1f}" if ref else ""
        ci = f"[{lo:.2f},{hi:.2f}]" if np.isfinite(lo) else "[1 seed: no CI]"
        print(f"  == {name} K={K}: {mean:.2f}% {ci}{refs}\n", flush=True)

    if a.stage == "calibrate":
        m = records[0]["test_acc"]
        ref = COATES_NG_2011_REFERENCE.get(a.k)
        if ref is None:
            print(f"no published reference at K={a.k}; calibrate at K in "
                  f"{sorted(COATES_NG_2011_REFERENCE)}")
        elif abs(m - ref) <= 2.0:
            print(f"CALIBRATION WITHIN TOLERANCE: {m:.2f}% vs published {ref:.1f}% (|delta| <= 2). "
                  f"Proceed to --stage sweep.")
        else:
            print(f"CALIBRATION OUTSIDE TOLERANCE: {m:.2f}% vs published {ref:.1f}% (delta {m - ref:+.1f}). "
                  f"Do NOT read the ceiling from a sweep. Run --stage ablate to find the defect.")


if __name__ == "__main__":
    main()
