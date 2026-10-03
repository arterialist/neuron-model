"""EXPERIMENT 2 -- the learned-feature ladder: at FIXED topology, how much does the rule that
produces the filters actually buy?

THE QUESTION THIS ANSWERS. Our substrate gets ~62% on CIFAR10 and every fixed-feature method we
tried lands at 60-65%. The open question is whether that is because (a) the filters are not
learned, (b) they are not learned *with labels*, or (c) filter learning barely matters and the
bottleneck is elsewhere. Those three have completely different consequences for the model
roadmap, and no amount of arguing settles them -- only a matched comparison does.

THE DESIGN. Every arm is IDENTICAL except for how the dictionary D (K x d) is obtained:
same patch size, same contrast-norm + ZCA preprocessing, same K, same triangle encoder, same
quadrant pooling, same row normalization of D, same linear readout protocol, same splits, same
seeds. So an arm difference is attributable to the LEARNING RULE and nothing else.

  random     gaussian filters                     no learning, no data          (control/floor)
  randpatch  random whitened data patches         no learning, sees data        (Coates' control)
  kmeans     Lloyd k-means                        unsupervised, GLOBAL objective
  krotov     Krotov-Hopfield competing units      unsupervised, LOCAL + kWTA competition  <<<
  supervised D trained by backprop through the    supervised, global, non-local (ceiling)
             exact same encoder, with labels

`krotov` is the arm that matters most for the model roadmap. It is a purely local, unsupervised
Hebbian rule whose only extra ingredient over a random substrate is COMPETITION (winner
potentiated, runner-up depressed) -- which is precisely proposed model-change #1 for PAULA
(lateral inhibition / kWTA). Its measured gain over `random` is a quantified payoff estimate
for building competition into the substrate, obtained BEFORE touching neuron.py.

`supervised` is the honest ceiling for this topology: same encoder, same capacity, labels
allowed. The gap (supervised - best unsupervised) measures what the label is worth here. The
--label-frac sweep asks how much supervision is needed to buy it.

PRE-REGISTERED HYPOTHESES AND THEIR CONSEQUENCES
  H1 (competition is load-bearing): krotov > random, CI on the paired difference excludes 0.
     -> local competitive plasticity builds features from nothing but structure. Quantified
        case for PAULA model-change #1, with an expected payoff = the measured delta.
  H1-null: krotov ~= random. -> adding kWTA to the substrate has low expected payoff. This
        would save building it, and is just as valuable as a positive.
  H2 (learning the dictionary matters at all): kmeans/krotov > randpatch.
     H2-null (Coates' own finding replicates): randpatch ~= kmeans -> dictionary LEARNING is
        not what buys performance; the encoder+pooling is. A random substrate is then not the
        problem, and the roadmap should target input geometry/readout, not filter learning.
  H3 (labels are the missing ingredient): supervised >> all unsupervised arms.
     -> unsupervised feature learning caps out here; PAULA's unsupervised hypothesis is
        fundamentally limited at this task, and the supervised-kappa path (Test C) is the
        priority. If instead supervised ~= kmeans, then labels are NOT the gap and the
        ceiling is the ARCHITECTURE (single layer, 6x6, quadrant pool), not the rule.

  Note H3 and H2-null can both hold. That combination -- filters barely matter, labels barely
  matter, everything lands ~65% -- would mean the single-layer architecture itself is the
  ceiling, and would be the strongest possible evidence that our 62% substrate is not
  underperforming and that DEPTH/structure (the milestone already planned) is the only route.
  `e2e_cnn` exists to bound that: an unmatched multi-layer CNN on the same data.

GUARDS
  * D, preprocessing, and feature standardization are fit on the TRAIN split only; the
    supervised arm's labels come only from TRAIN. Readout C is chosen on a disjoint VAL split.
    TEST is scored exactly once per arm. Asserted in code, not assumed.
  * Matched capacity: identical K across arms (a rule cannot win by using more filters).
  * Matched data budget: no augmentation in any arm, including supervised.
  * Degeneracy guard: dictionary row-norm spread and dead-filter count (filters never winning
    the triangle encoder) are recorded. An arm that "wins" with most filters dead is flagged as
    collapse, not learning.
  * >=3 seeds; paired-by-seed differences with bootstrap CIs. A difference whose CI includes 0
    is reported as no difference, however suggestive the means look.

CLI-only; run from neuron-model/. See EXPERIMENT_DESIGN_cifar_ceiling.md.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from cifar_ceiling.common import (
    Preproc, bootstrap_ci, encode_batch, encode_dataset, eval_test, fit_readout, kmeans_lloyd,
    krotov_hopfield, load_cifar, normalize_dict, pick_device, pool_quadrants,
    random_gaussian_dict, random_patch_dict, sample_patches, train_val_split,
    experiment_protocol, read_cached_record, write_record,
)

ARMS = ["random", "randpatch", "kmeans", "krotov", "supervised", "e2e_cnn"]


# --------------------------------------------------------------------- supervised dictionary

class MatchedEncoderNet(nn.Module):
    """The matched pipeline, made differentiable, so that D can be learned by backprop through
    EXACTLY the encoder the unsupervised arms are evaluated with. This is what makes the
    supervised arm a fair rung on the ladder rather than a different model.

    BatchNorm before the head only stabilizes optimization; it is NOT part of the matched
    evaluation -- the learned D is afterwards re-normalized and pushed through the same frozen
    encode + standard readout as every other arm.
    """

    def __init__(self, D_init: np.ndarray, pre: Preproc, P: int, L: int, ncls: int = 10):
        super().__init__()
        self.D = nn.Parameter(torch.as_tensor(D_init.copy()))
        self.pre, self.P, self.L = pre, P, L
        self.bn = nn.BatchNorm1d(4 * D_init.shape[0], affine=False)
        self.head = nn.Linear(4 * D_init.shape[0], ncls)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        p = Fn.unfold(x, self.P, stride=1).permute(0, 2, 1).reshape(-1, 3 * self.P * self.P)
        pz = self.pre.torch_apply(p)
        act = encode_batch(pz, self.D, "triangle").reshape(B, self.L * self.L, self.D.shape[0])
        return self.head(self.bn(pool_quadrants(act, self.L)))


def learn_supervised_dict(Xtr, ytr, Xva, yva, D_init, pre, P, L, device, a, seed):
    """Train D through the matched encoder with labels. Returns (D, e2e_val_acc)."""
    torch.manual_seed(seed)
    net = MatchedEncoderNet(D_init, pre, P, L).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=a.sup_lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.sup_epochs)
    Xt = torch.as_tensor(Xtr, device=device)
    Yt = torch.as_tensor(ytr, device=device, dtype=torch.long)
    Xv = torch.as_tensor(Xva, device=device)
    Yv = torch.as_tensor(yva, device=device, dtype=torch.long)
    for ep in range(a.sup_epochs):
        net.train()
        perm = torch.randperm(len(Xt), device=device)
        tot = 0.0
        for i in range(0, len(Xt), a.sup_batch):
            b = perm[i:i + a.sup_batch]
            opt.zero_grad()
            loss = Fn.cross_entropy(net(Xt[b]), Yt[b])
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        sched.step()
        net.eval()
        with torch.no_grad():
            acc = float(np.mean([
                (net(Xv[i:i + 256]).argmax(1) == Yv[i:i + 256]).float().mean().item()
                for i in range(0, len(Xv), 256)]))
        print(f"    sup epoch {ep + 1}/{a.sup_epochs} loss={tot / len(Xt):.4f} e2e_val={acc * 100:.2f}%",
              flush=True)
    return net.D.detach().cpu().numpy(), acc


# --------------------------------------------------------------------- unmatched CNN reference

class SmallCNN(nn.Module):
    """UNMATCHED reference arm: a conventional multi-layer CNN. Reported only to bound how much
    of the remaining gap is DEPTH rather than filter learning. It shares nothing with the ladder
    except the data and the no-augmentation budget, and must never be compared arm-to-arm."""

    def __init__(self, ncls: int = 10):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                 nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                 nn.MaxPool2d(2))
        self.f = nn.Sequential(blk(3, 64), blk(64, 128), blk(128, 256), nn.AdaptiveAvgPool2d(1))
        self.head = nn.Linear(256, ncls)

    def forward(self, x):
        return self.head(self.f(x).flatten(1))


def run_e2e_cnn(Xtr, ytr, Xva, yva, Xte, yte, device, a, seed):
    torch.manual_seed(seed)
    net = SmallCNN().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.cnn_epochs)
    Xt = torch.as_tensor(Xtr, device=device); Yt = torch.as_tensor(ytr, device=device, dtype=torch.long)

    def acc_on(X, y):
        net.eval()
        Xd = torch.as_tensor(X, device=device); yd = torch.as_tensor(y, device=device, dtype=torch.long)
        with torch.no_grad():
            correct = sum(int((net(Xd[i:i + 256]).argmax(1) == yd[i:i + 256]).sum())
                          for i in range(0, len(Xd), 256))
        return correct / len(X)

    for ep in range(a.cnn_epochs):
        net.train()
        perm = torch.randperm(len(Xt), device=device)
        for i in range(0, len(Xt), 128):
            b = perm[i:i + 128]
            opt.zero_grad()
            Fn.cross_entropy(net(Xt[b]), Yt[b]).backward()
            opt.step()
        sched.step()
        if (ep + 1) % 5 == 0:
            print(f"    cnn epoch {ep + 1}/{a.cnn_epochs} val={acc_on(Xva, yva) * 100:.2f}%", flush=True)
    return acc_on(Xva, yva), acc_on(Xte, yte)


# --------------------------------------------------------------------- one arm

def dict_health(D: np.ndarray, pz: np.ndarray, device) -> dict:
    """Degeneracy guard: are filters actually used, or has the rule collapsed?"""
    Dt = torch.as_tensor(D, device=device)
    sub = torch.as_tensor(pz[:20000], device=device)
    act = encode_batch(sub, Dt, "triangle")
    used = (act > 0).float().mean(0)
    norms = np.linalg.norm(D, axis=1)
    return dict(dead_filters=int((used == 0).sum()), mean_use=round(float(used.mean()), 4),
                rownorm_cv=round(float(norms.std() / (norms.mean() + 1e-9)), 4))


def run_arm(arm: str, K: int, seed: int, a, device, data) -> dict:
    t_all = time.time()
    Xtr_all, ytr_all, Xte, yte = data
    tr_idx, va_idx = train_val_split(len(Xtr_all), a.val_frac, seed)
    if np.intersect1d(tr_idx, va_idx).size:
        raise RuntimeError("train/validation split overlap")
    Xtr, ytr = Xtr_all[tr_idx], ytr_all[tr_idx]
    Xva, yva = Xtr_all[va_idx], ytr_all[va_idx]
    P, L = a.patch, (32 - a.patch) + 1

    if arm == "e2e_cnn":
        val, test = run_e2e_cnn(Xtr, ytr, Xva, yva, Xte, yte, device, a, seed)
        return dict(arm=arm, K=None, seed=seed, matched=False, test_acc=round(test * 100, 3),
                    val_acc=round(val * 100, 3), secs=round(time.time() - t_all))

    # --- matched preprocessing, fit on TRAIN only ---
    pw = sample_patches(Xtr, a.n_patches, P, seed)
    pre = Preproc(P, a.cn_eps, True, a.zca_eps, True).fit(pw)
    pz = pre.apply(pw)

    # --- the ONLY thing that differs between arms ---
    t0 = time.time()
    e2e_val = None
    if arm == "random":
        D = random_gaussian_dict(pz.shape[1], K, seed)
    elif arm == "randpatch":
        D = random_patch_dict(pz, K, seed)
    elif arm == "kmeans":
        D = kmeans_lloyd(pz, K, a.kmeans_iters, seed, device, verbose=a.verbose)
    elif arm == "krotov":
        D = krotov_hopfield(pz, K, a.krotov_epochs, seed, device, p=a.krotov_p,
                            k_rank=a.krotov_k, delta=a.krotov_delta, verbose=a.verbose)
    elif arm == "supervised":
        n_lab = int(len(Xtr) * a.label_frac)
        lab = np.random.RandomState(seed).choice(len(Xtr), n_lab, replace=False)
        D0 = random_patch_dict(pz, K, seed)          # same init as the randpatch arm
        D, e2e_val = learn_supervised_dict(Xtr[lab], ytr[lab], Xva, yva, D0, pre, P, L,
                                           device, a, seed)
    else:
        raise ValueError(arm)
    t_dict = time.time() - t0
    health = dict_health(D, pz, device)
    D = normalize_dict(D)

    # --- matched evaluation path, identical for every arm ---
    ekw = dict(pre=pre, device=device, P=P, stride=1, mode="triangle",
               batch=a.encode_batch, verbose=a.verbose)
    t0 = time.time()
    Ftr = encode_dataset(Xtr, D, **ekw)
    Fva = encode_dataset(Xva, D, **ekw)
    Fte = encode_dataset(Xte, D, **ekw)
    t_enc = time.time() - t0
    ro = fit_readout(Ftr, ytr, Fva, yva, device, C_grid=a.c_grid, epochs=a.readout_epochs,
                     seed=seed, verbose=a.verbose)
    test_acc = eval_test(ro, Fte, yte, device)      # the single test read

    return dict(arm=arm, K=K, seed=seed, matched=True, test_acc=round(test_acc * 100, 3),
                val_acc=round(ro["val_acc"] * 100, 3), C=ro["C"], e2e_val_acc=e2e_val,
                label_frac=a.label_frac if arm == "supervised" else None, health=health,
                n_train=int(len(tr_idx)), secs=dict(dict=round(t_dict), encode=round(t_enc),
                                                    total=round(time.time() - t_all)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="random,randpatch,kmeans,krotov,supervised")
    ap.add_argument("--k", type=int, default=400, help="matched dictionary size for ALL arms")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", default="cifar_ceiling/results/ladder")
    ap.add_argument("--patch", type=int, default=6)
    ap.add_argument("--n-patches", type=int, default=400_000)
    ap.add_argument("--cn-eps", type=float, default=10.0)
    ap.add_argument("--zca-eps", type=float, default=0.1)
    ap.add_argument("--kmeans-iters", type=int, default=50)
    ap.add_argument("--krotov-epochs", type=int, default=50)
    ap.add_argument("--krotov-p", type=float, default=2.0)
    ap.add_argument("--krotov-k", type=int, default=2, help="rank of the anti-Hebbian unit")
    ap.add_argument("--krotov-delta", type=float, default=0.4)
    ap.add_argument("--label-frac", type=float, default=1.0, help="supervised arm label budget")
    ap.add_argument("--sup-epochs", type=int, default=20)
    ap.add_argument("--sup-batch", type=int, default=64)
    ap.add_argument("--sup-lr", type=float, default=1e-3)
    ap.add_argument("--cnn-epochs", type=int, default=30)
    ap.add_argument("--train-n", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--encode-batch", type=int, default=64)
    ap.add_argument("--readout-epochs", type=int, default=60)
    ap.add_argument("--c-grid", default="3e-4,1e-3,3e-3,1e-2,3e-2,1e-1")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    a.c_grid = [float(x) for x in a.c_grid.split(",")]
    arms = [x for x in a.arms.split(",") if x]
    seeds = [int(s) for s in a.seeds.split(",")]
    if not arms or not all(x in ARMS for x in arms):
        ap.error(f"unknown or empty arm list; choose from {ARMS}")
    if not 0.0 < a.label_frac <= 1.0:
        ap.error("--label-frac must be in (0, 1]")
    device = pick_device(a.device)
    os.makedirs(a.out, exist_ok=True)

    data = load_cifar(a.data_root, 255.0)
    if a.train_n:
        sub = np.random.RandomState(0).choice(len(data[0]), a.train_n, replace=False)
        data = (data[0][sub], data[1][sub], data[2], data[3])
    print(f"device={device} K={a.k} arms={arms} seeds={seeds} train={len(data[0])} "
          f"test={len(data[2])} label_frac={a.label_frac}\n", flush=True)

    res = {}
    for arm in arms:
        accs = []
        for seed in seeds:
            lf = f"_lf{a.label_frac}" if arm == "supervised" and a.label_frac < 1.0 else ""
            tag = f"{arm}{lf}_K{a.k}_s{seed}"
            path = os.path.join(a.out, f"{tag}.json")
            protocol = experiment_protocol(a, arm=arm, K=a.k, seed=seed)
            if os.path.exists(path):
                r = read_cached_record(path, protocol)
                print(f"[{tag}] cached test={r['test_acc']:.2f}%", flush=True)
            else:
                print(f"[{tag}] running...", flush=True)
                r = run_arm(arm, a.k, seed, a, device, data)
                r = write_record(path, r, protocol)
                h = r.get("health") or {}
                print(f"[{tag}] test={r['test_acc']:.2f}% val={r['val_acc']:.2f}% "
                      f"dead={h.get('dead_filters')} use={h.get('mean_use')} "
                      f"({r['secs'] if isinstance(r['secs'], int) else r['secs']['total']}s)",
                      flush=True)
            accs.append(r["test_acc"])
        res[arm] = accs
        m, (lo, hi) = bootstrap_ci(accs)
        ci = f"[{lo:.2f},{hi:.2f}]" if np.isfinite(lo) else "[<2 seeds: no CI]"
        print(f"  == {arm}: {m:.2f}% {ci}\n", flush=True)

    # paired-by-seed contrasts that the hypotheses are actually about
    print("=== pre-registered contrasts (paired by seed; CI excluding 0 = real) ===")
    for lo_arm, hi_arm, label in [("random", "krotov", "H1: competition load-bearing"),
                                  ("randpatch", "kmeans", "H2: dictionary learning matters"),
                                  ("randpatch", "krotov", "H2: local learning matters"),
                                  ("kmeans", "supervised", "H3: labels are the gap"),
                                  ("random", "kmeans", "learning vs no learning")]:
        if lo_arm in res and hi_arm in res:
            n = min(len(res[lo_arm]), len(res[hi_arm]))
            d = np.asarray(res[hi_arm][:n]) - np.asarray(res[lo_arm][:n])
            m, (l, h) = bootstrap_ci(d)
            sig = "*" if np.isfinite(l) and (l > 0 or h < 0) else " "
            ci = f"[{l:+.2f},{h:+.2f}]" if np.isfinite(l) else "[no CI]"
            print(f"  {sig} {hi_arm} - {lo_arm}: {m:+.2f} {ci}   ({label})")
    # This derived summary may be refreshed; per-arm primary records above are never replaced.
    with open(os.path.join(a.out, "ladder_summary.json"), "w", encoding="utf-8") as stream:
        json.dump(res, stream, indent=2)


if __name__ == "__main__":
    main()
