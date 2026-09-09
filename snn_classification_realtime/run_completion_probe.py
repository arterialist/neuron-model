"""Test A (pattern completion): as a continuous TRAIN stream forms whatever structure it
forms, do genuinely-unseen OCCLUDED test samples get pulled to their class attractor (i.e.
complete), and does that completion SHARPEN with training?

Pattern completion is the defining signature of an attractor basin: a degraded input should
settle to the stored class pattern. The earlier negative used whole inputs + a mean
descriptor and could miss this. Here, at each checkpoint we snapshot the live net and probe:
  * REF  (held-out, UN-occluded)  -> build per-class centroids for scoring.
  * QUERY (held-out) under conditions: whole (rho=0), crop-0.5 (bottom half blanked),
    rand-0.5 (half the pixels blanked at random), shuf (all pixels spatially scrambled,
    values intact = spatial-structure control).
REF/QUERY are disjoint from TRAIN and from each other and never drive plasticity. Two probe
modes: frozen (eta=0) and on. No neuron.py edit (freeze = zero the learning rates on the
copy; snapshot via pickle with loggers/history stripped). CLI-only. See
foveation_results/basin_generalization/EXPERIMENT_DESIGN_no_model_change.md (Test A).
"""
import argparse
import json
import os
import pickle
import time

import numpy as np

from neuron.nn_core import NNCore
from neuron.network_config import NetworkConfig
from neuron.neuron import Neuron
from snn_classification_realtime.core.input_mapping import image_to_signals
from snn_classification_realtime.activity_dataset_builder.vision_datasets import load_dataset_by_name
from snn_classification_realtime.activity_dataset_builder.network_utils import (
    infer_layers_from_metadata, determine_input_mapping,
)
from snn_classification_realtime.run_basin_generalization import (
    _spec, total_weight, _make_picklable, _restore, _rehydrate,
)

# probe conditions: (family, rho). "whole" is the un-occluded reference; "shuf" scrambles
# pixel positions (spatial-structure control); crop/rand remove rho fraction of drive.
CONDITIONS = [("whole", 0.0), ("crop", 0.5), ("rand", 0.5), ("shuf", 0.0)]

_W = {}


def _dyn_feats(Sbuf):
    """Per-neuron DYNAMICAL features of the settled window (Test B): AC power, dominant
    frequency, spectral centroid of the DC-removed membrane trace. Length 3N."""
    ac = Sbuf - Sbuf.mean(0)                       # (T,N) DC-removed
    mag = np.abs(np.fft.rfft(ac, axis=0)); mag[0] = 0.0
    freqs = np.fft.rfftfreq(Sbuf.shape[0])
    power = ac.var(0)                              # N: AC power
    tot = mag.sum(0) + 1e-9
    dom = freqs[mag.argmax(0)]                     # N: dominant freq
    centroid = (mag * freqs[:, None]).sum(0) / tot  # N: spectral centroid
    return np.concatenate([power, dom, centroid]).astype(np.float32)


def _descriptor_ext(net, core, sig, ticks, settle):
    """Run one sample from the net's current state; return (full_descriptor, static_dim).
    full = [STATIC: meanS, mean_tref, stdS, spec(popS)] + [DYNAMICAL: _dyn_feats(Sbuf)].
    Test A scores the static block; Test B re-scores the dynamical block. Same run -> B is
    free."""
    N = len(net.network.neurons)
    Sbuf = np.empty((ticks - settle, N), np.float32)
    Tbuf = np.empty((ticks - settle, N), np.float32)
    for t in range(ticks):
        core.send_batch_signals(sig); core.do_tick()
        if t >= settle:
            r = t - settle
            Sbuf[r] = [nu.S for nu in net.network.neurons.values()]
            Tbuf[r] = [nu.t_ref for nu in net.network.neurons.values()]
    static = np.concatenate([Sbuf.mean(0), Tbuf.mean(0), Sbuf.std(0), _spec(Sbuf.mean(1))]).astype(np.float32)
    full = np.concatenate([static, _dyn_feats(Sbuf)]).astype(np.float32)
    return full, len(static)


def occlude(img, family, rho, seed):
    """img: torch.Tensor [1,H,W] in [-1,1]. Return an occluded clone (background=min)."""
    import torch
    x = img.clone()
    bg = float(x.min())
    _, H, W = x.shape
    if family == "whole" or (rho <= 0 and family in ("crop", "rand")):
        return x
    rng = np.random.RandomState(seed)
    flat = x.view(-1)
    n = flat.numel()
    if family == "rand":
        k = int(rho * n)
        idx = rng.choice(n, k, replace=False)
        flat[torch.as_tensor(idx, dtype=torch.long)] = bg
    elif family == "crop":
        rows = int(rho * H)
        x[:, H - rows:, :] = bg
    elif family == "shuf":
        perm = torch.as_tensor(rng.permutation(n), dtype=torch.long)
        x.view(-1)[:] = flat[perm]
    return x


def _init(dataset, signal_gain, in_ids, insyn, ticks, settle):
    dc = load_dataset_by_name(dataset)
    dc.signal_gain = signal_gain
    _W.update(dc=dc, ds=dc.dataset, in_ids=in_ids, insyn=insyn, ticks=ticks, settle=settle)


def _probe(args):
    """Worker: (net_bytes, idx, family, rho, mode, mask_seed) -> descriptor of the
    (possibly occluded) held-out sample run from a fresh copy of the checkpoint net."""
    net_bytes, idx, family, rho, mode, mask_seed = args
    net = pickle.loads(net_bytes)
    _rehydrate(net)
    if mode == "frozen":
        for nu in net.network.neurons.values():
            nu.params.eta_post = 0.0
            nu.params.eta_retro = 0.0
    core = NNCore(); core.neural_net = net; core.set_log_level("CRITICAL")
    img, _ = _W["ds"][int(idx)]
    img = occlude(img, family, rho, mask_seed)
    sig = image_to_signals(img, _W["in_ids"], _W["insyn"], net, _W["dc"])
    full, _sd = _descriptor_ext(net, core, sig, _W["ticks"], _W["settle"])
    return full


def nearest_centroid_acc(ref_D, ref_y, q_D, q_y):
    """Standardize on REF, centroids = REF class means, classify QUERY by nearest centroid.
    Returns (accuracy, own-vs-other margin)."""
    m, sd = ref_D.mean(0), ref_D.std(0) + 1e-9
    R = (ref_D - m) / sd
    Q = (q_D - m) / sd
    classes = sorted(set(ref_y.tolist()))
    cents = np.stack([R[ref_y == c].mean(0) for c in classes])
    yi = {c: i for i, c in enumerate(classes)}
    d = np.linalg.norm(Q[:, None, :] - cents[None], axis=2)  # (nq, ncl)
    pred = np.array(classes)[d.argmin(1)]
    acc = float((pred == q_y).mean())
    own = np.array([d[j, yi[int(q_y[j])]] for j in range(len(q_y))])
    other = np.array([np.delete(d[j], yi[int(q_y[j])]).mean() for j in range(len(q_y))])
    margin = float((other - own).mean() / (other.mean() + 1e-9))
    return acc, margin


def build_pools(ds, classes, train_pc, ref_pc, query_pc, order, seed):
    need = train_pc + ref_pc + query_pc
    by = {c: [] for c in classes}
    for i in np.random.RandomState(seed).permutation(len(ds)):
        _, yv = ds[int(i)]; yv = int(yv)
        if yv in by and len(by[yv]) < need:
            by[yv].append(int(i))
        if all(len(v) >= need for v in by.values()):
            break
    train, ref, query = {}, [], []
    for c in classes:
        train[c] = by[c][:train_pc]
        ref += [(i, c) for i in by[c][train_pc:train_pc + ref_pc]]
        query += [(i, c) for i in by[c][train_pc + ref_pc:need]]
    if order == "blocked":
        seq = [(i, c) for c in classes for i in train[c]]
    else:
        seq = []
        for k in range(train_pc):
            for c in classes:
                if k < len(train[c]):
                    seq.append((train[c][k], c))
    ref_idx = np.array([i for i, _ in ref]); ref_lab = np.array([c for _, c in ref])
    q_idx = np.array([i for i, _ in query]); q_lab = np.array([c for _, c in query])
    # leakage assertion: pools disjoint
    assert len(set(ref_idx) & set(q_idx)) == 0
    assert len(set(ref_idx) & set(i for i, _ in seq)) == 0
    assert len(set(q_idx) & set(i for i, _ in seq)) == 0
    return seq, ref_idx, ref_lab, q_idx, q_lab


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True)
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--classes", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--train-per-class", type=int, default=80)
    ap.add_argument("--ref-per-class", type=int, default=6)
    ap.add_argument("--query-per-class", type=int, default=6)
    ap.add_argument("--ticks", type=int, default=3000)
    ap.add_argument("--settle-frac", type=float, default=0.4)
    ap.add_argument("--checkpoint-every", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=1, help="Test D: re-present the SAME train set E times")
    ap.add_argument("--order", choices=["interleaved", "blocked"], default="interleaved")
    ap.add_argument("--modes", default="frozen,on")
    ap.add_argument("--signal-gain", type=float, default=0.336)
    ap.add_argument("--probe-workers", type=int, default=10)
    ap.add_argument("--out", default="foveation_results/completion_probe")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    classes = [int(x) for x in a.classes.split(",")]
    modes = [m.strip() for m in a.modes.split(",") if m.strip()]
    settle = int(a.ticks * a.settle_frac)

    net = NetworkConfig.load_network_config(a.net, neuron_class=Neuron)
    core = NNCore(); core.neural_net = net; core.set_log_level("CRITICAL")
    layers = infer_layers_from_metadata(net)
    in_ids, insyn = determine_input_mapping(net, layers)
    dc = load_dataset_by_name(a.dataset); dc.signal_gain = a.signal_gain
    ds = dc.dataset
    seq, ref_idx, ref_lab, q_idx, q_lab = build_pools(
        ds, classes, a.train_per_class, a.ref_per_class, a.query_per_class, a.order, a.seed)
    net.reset_simulation(); core.state.current_tick = 0; net.current_tick = 0
    N = len(net.network.neurons)
    static_dim = 3 * N + 24  # [meanS, mean_tref, stdS](3N) + spec(24); dyn block follows
    total_present = len(seq) * a.epochs  # Test D: E re-presentations of the same train set
    print(f"[{a.tag}] net={a.net} N={N} train={len(seq)}x{a.epochs}ep={total_present} "
          f"ref={len(ref_idx)} query={len(q_idx)} order={a.order} conds={CONDITIONS} modes={modes} "
          f"ticks={a.ticks} ckpt_every={a.checkpoint_every} workers={a.probe_workers} seed={a.seed}", flush=True)

    pool = mp.Pool(a.probe_workers, initializer=_init,
                   initargs=(a.dataset, a.signal_gain, in_ids, insyn, a.ticks, settle))
    ckpts = sorted(set([0] + list(range(a.checkpoint_every, total_present + 1, a.checkpoint_every)) + [total_present]))

    rec = dict(n_train=[], weight=[])
    # per (mode) -> list over checkpoints of ref descriptors; per (mode,cond) -> query descs
    ref_store = {m: [] for m in modes}
    q_store = {(m, f, r): [] for m in modes for (f, r) in CONDITIONS}
    t0 = time.time()

    def probe(n_seen):
        saved = _make_picklable(net)
        try:
            net_bytes = pickle.dumps(net, protocol=pickle.HIGHEST_PROTOCOL)
        finally:
            _restore(net, saved)
        rec["n_train"].append(int(n_seen)); rec["weight"].append(total_weight(net))
        for m in modes:
            ref_tasks = [(net_bytes, int(i), "whole", 0.0, m, a.seed) for i in ref_idx]
            ref_D = np.stack(pool.map(_probe, ref_tasks)).astype(np.float32)
            ref_store[m].append(ref_D)
            line = [f"n_train={n_seen:4d} {m:6s}"]
            for (f, r) in CONDITIONS:
                # deterministic per-(query,cond) mask seed
                q_tasks = [(net_bytes, int(qi), f, r, m, a.seed * 100003 + int(qi) * 131 + int(r * 100) + hash(f) % 97)
                           for qi in q_idx]
                q_D = np.stack(pool.map(_probe, q_tasks)).astype(np.float32)
                q_store[(m, f, r)].append(q_D)
                # online metric = Test A (static block only); Test B re-scores the dyn block offline
                acc, mar = nearest_centroid_acc(ref_D[:, :static_dim], ref_lab, q_D[:, :static_dim], q_lab)
                line.append(f"{f}{r:g}:acc={acc:.2f}")
            print(f"[{a.tag}] " + " ".join(line) + f"  W={rec['weight'][-1]:.0f} ({time.time()-t0:.0f}s)", flush=True)

    probe(0)
    ci = 1
    gpos = 0
    for ep in range(a.epochs):
        for (idx, c) in seq:
            gpos += 1
            img, _ = ds[idx]
            sig = image_to_signals(img, in_ids, insyn, net, dc)
            for t in range(a.ticks):
                core.send_batch_signals(sig); core.do_tick()
            if ci < len(ckpts) and gpos == ckpts[ci]:
                probe(gpos); ci += 1
    pool.close(); pool.join()

    save = dict(n_train=np.asarray(rec["n_train"]), weight=np.asarray(rec["weight"], np.float32),
                ref_lab=ref_lab, q_lab=q_lab, classes=np.asarray(classes),
                modes=np.asarray(modes), conditions=np.asarray([f"{f}_{int(r*100)}" for f, r in CONDITIONS]),
                order=a.order, ticks=a.ticks, settle=settle, seed=a.seed,
                static_dim=static_dim, epochs=a.epochs, train_len=len(seq))
    for m in modes:
        save[f"ref_{m}"] = np.asarray(ref_store[m], np.float32)
        for (f, r) in CONDITIONS:
            save[f"q_{m}__{f}_{int(r*100)}"] = np.asarray(q_store[(m, f, r)], np.float32)
    np.savez_compressed(os.path.join(a.out, f"{a.tag}.npz"), **save)
    json.dump(dict(tag=a.tag, net=a.net, classes=classes, order=a.order, seed=a.seed,
                   train=len(seq), epochs=a.epochs, total_present=total_present,
                   ref=len(ref_idx), query=len(q_idx), modes=modes,
                   conditions=[f"{f}_{int(r*100)}" for f, r in CONDITIONS], ticks=a.ticks,
                   static_dim=static_dim, n_checkpoints=len(rec["n_train"]), secs=round(time.time() - t0)),
              open(os.path.join(a.out, f"{a.tag}_meta.json"), "w"), indent=2)
    print(f"[{a.tag}] DONE {len(rec['n_train'])} checkpoints ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
