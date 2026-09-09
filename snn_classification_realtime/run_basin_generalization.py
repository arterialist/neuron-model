"""Does sustained same-class exposure SHARPEN per-class attractor basins so that
genuinely-unseen test samples fall progressively closer to their OWN class centroid?

This is the corrected form of the continuous slow-adaptation hypothesis. Earlier runs
measured presented samples and were confounded by (a) a global settling transient and
(b) class-time confound in blocked order. Here:

  * TRAINING set is streamed through the network continuously (NO reset between samples),
    interleaved (class decorrelated from time) so plasticity forms whatever structure it
    forms. Plasticity mode / rh_decay come from the arm JSON global_params.
  * A fixed, DISJOINT, never-trained TEST set is probed at checkpoints (every K training
    samples). At each checkpoint we snapshot the live network and, for each test sample,
    run a FRESH copy from that snapshot for `ticks` and record a settled-window attractor
    descriptor. Test samples never contaminate each other or the training trajectory.
  * TWO probe conditions per checkpoint:
      - "frozen": eta_post = eta_retro = 0 on the copy  -> weights held; tests whether the
        FORMED structure alone places unseen samples near their class basin.
      - "on"    : plasticity intact on the copy          -> each unseen sample may re-fit
        during its own run; isolates ongoing adaptation from the pre-formed structure.
  * Metric (computed here + re-derivable in analysis): for the held-out test descriptors,
    leave-one-out nearest-OWN-class-centroid accuracy, own-vs-other distance margin, and
    between/within centroid ratio. The global pull is common-mode and cancels in a
    nearest-own-centroid comparison, so a rise over training = class-specific sharpening.

Raw per-checkpoint test descriptors are kept (tiny) for the animation. CLI-only.
No neuron.py edits (freeze = zero the learning rates on the copy).
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

KSPEC = 24

# ---- worker globals (set once per process by _init) --------------------------
_W = {}


def _spec(sig, k=KSPEC):
    mag = np.abs(np.fft.rfft(sig - sig.mean()))
    if len(mag) < 2:
        return np.zeros(k, np.float32)
    return np.interp(np.linspace(0, len(mag) - 1, k), np.arange(len(mag)), mag).astype(np.float32)


def total_weight(net):
    return float(sum(psp.u_i.info for nu in net.network.neurons.values()
                     for psp in nu.postsynaptic_points.values()))


def _descriptor(net, core, sig, ticks, settle):
    """Run one sample on `net`/`core` from its current state for `ticks`; return the
    settled-window descriptor [meanS, mean_tref, stdS, spec(popS)]."""
    N = len(net.network.neurons)
    Sbuf = np.empty((ticks - settle, N), np.float32)
    Tbuf = np.empty((ticks - settle, N), np.float32)
    for t in range(ticks):
        core.send_batch_signals(sig)
        core.do_tick()
        if t >= settle:
            r = t - settle
            Sbuf[r] = [nu.S for nu in net.network.neurons.values()]
            Tbuf[r] = [nu.t_ref for nu in net.network.neurons.values()]
    mS, mT, sdS = Sbuf.mean(0), Tbuf.mean(0), Sbuf.std(0)
    popS = Sbuf.mean(1)
    return np.concatenate([mS, mT, sdS, _spec(popS)]).astype(np.float32)


def _init(dataset, signal_gain, in_ids, insyn, ticks, settle):
    dc = load_dataset_by_name(dataset)
    dc.signal_gain = signal_gain
    _W.update(dc=dc, ds=dc.dataset, in_ids=in_ids, insyn=insyn, ticks=ticks, settle=settle)


def _empty_history():
    from collections import deque
    return {"ticks": deque(), "neuron_states": {}, "network_activity": deque()}


def _make_picklable(net):
    """Detach the unpicklable bits (loguru-bound-with-lambda-sink per-neuron loggers and
    the defaultdict(lambda) history) so the net can be pickled to workers. Returns saved
    state for restoration on the live net afterwards."""
    saved_loggers = [(nu, nu.logger) for nu in net.network.neurons.values()]
    for nu, _ in saved_loggers:
        nu.logger = None
    saved_history = net.history
    net.history = None
    return saved_loggers, saved_history


def _restore(net, saved):
    saved_loggers, saved_history = saved
    for nu, lg in saved_loggers:
        nu.logger = lg
    net.history = saved_history


def _rehydrate(net):
    """Give a freshly-unpickled net working (silent) loggers and a benign history again."""
    from loguru import logger
    for nid, nu in net.network.neurons.items():
        nu.logger = logger.bind(neuron_int=nid, neuron_hex=f"{nid:09x}")
        nu.logger_active = False
    if getattr(net, "history", None) is None:
        net.history = _empty_history()


def _probe_one(args):
    """Worker: (net_bytes, test_idx, mode) -> descriptor for one unseen test sample run
    from a fresh copy of the checkpoint network."""
    net_bytes, test_idx, mode = args
    net = pickle.loads(net_bytes)
    _rehydrate(net)
    if mode == "frozen":
        for nu in net.network.neurons.values():
            nu.params.eta_post = 0.0
            nu.params.eta_retro = 0.0
    core = NNCore(); core.neural_net = net; core.set_log_level("CRITICAL")
    img, _ = _W["ds"][int(test_idx)]
    sig = image_to_signals(img, _W["in_ids"], _W["insyn"], net, _W["dc"])
    return _descriptor(net, core, sig, _W["ticks"], _W["settle"])


# ---- metrics -----------------------------------------------------------------
def loo_nearest_centroid(X, y):
    """Leave-one-out nearest-OWN-class-centroid on standardized descriptors.
    Returns (accuracy, own_vs_other_margin, between/within)."""
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    classes = sorted(set(y.tolist()))
    yi = {c: i for i, c in enumerate(classes)}
    # per-class sum/count for O(1) leave-one-out centroids
    sums = {c: X[y == c].sum(0) for c in classes}
    cnts = {c: int((y == c).sum()) for c in classes}
    cents_full = np.stack([sums[c] / max(cnts[c], 1) for c in classes])
    correct, owns, others = 0, [], []
    for j in range(len(X)):
        cj = int(y[j])
        cents = []
        for c in classes:
            if c == cj and cnts[c] > 1:
                cents.append((sums[c] - X[j]) / (cnts[c] - 1))  # LOO own centroid
            else:
                cents.append(sums[c] / max(cnts[c], 1))
        cents = np.stack(cents)
        d = np.linalg.norm(cents - X[j], axis=1)
        if classes[int(d.argmin())] == cj:
            correct += 1
        owns.append(d[yi[cj]])
        others.append(np.delete(d, yi[cj]).mean())
    owns, others = np.asarray(owns), np.asarray(others)
    margin = float((others - owns).mean() / (others.mean() + 1e-9))
    within = np.mean([np.linalg.norm(X[y == c] - cents_full[i], axis=1).mean()
                      for i, c in enumerate(classes)])
    betw = np.mean([np.linalg.norm(cents_full[i] - cents_full[k])
                    for i in range(len(classes)) for k in range(i + 1, len(classes))])
    return correct / len(X), margin, float(betw / (within + 1e-9))


def build_sets(ds, classes, train_pc, test_pc, seed):
    """Disjoint train/test index pools per class (test never trained)."""
    by = {c: [] for c in classes}
    need = train_pc + test_pc
    for i in np.random.RandomState(seed).permutation(len(ds)):
        _, yv = ds[int(i)]; yv = int(yv)
        if yv in by and len(by[yv]) < need:
            by[yv].append(int(i))
        if all(len(v) >= need for v in by.values()):
            break
    train, test = {}, {}
    for c in classes:
        train[c] = by[c][:train_pc]
        test[c] = by[c][train_pc:train_pc + test_pc]
    # interleaved training order (round-robin), class decorrelated from time
    seq = []
    for k in range(train_pc):
        for c in classes:
            if k < len(train[c]):
                seq.append((train[c][k], c, k))
    test_idx = np.array([i for c in classes for i in test[c]])
    test_lab = np.array([c for c in classes for _ in test[c]])
    return seq, test_idx, test_lab


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True)
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--classes", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--train-per-class", type=int, default=50)
    ap.add_argument("--test-per-class", type=int, default=6)
    ap.add_argument("--ticks", type=int, default=3000)
    ap.add_argument("--settle-frac", type=float, default=0.4)
    ap.add_argument("--checkpoint-every", type=int, default=25, help="probe every K training samples")
    ap.add_argument("--modes", default="frozen,on")
    ap.add_argument("--signal-gain", type=float, default=0.336)
    ap.add_argument("--probe-workers", type=int, default=6)
    ap.add_argument("--out", default="foveation_results/basin_generalization")
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
    in_ids, insyn = determine_input_mapping(net, layers)  # constant topology -> reuse in workers
    dc = load_dataset_by_name(a.dataset); dc.signal_gain = a.signal_gain
    ds = dc.dataset
    seq, test_idx, test_lab = build_sets(ds, classes, a.train_per_class, a.test_per_class, a.seed)
    net.reset_simulation(); core.state.current_tick = 0; net.current_tick = 0
    N = len(net.network.neurons)
    print(f"[{a.tag}] net={a.net} N={N} train={len(seq)} test={len(test_idx)} "
          f"modes={modes} ticks={a.ticks} settle={settle} ckpt_every={a.checkpoint_every} "
          f"workers={a.probe_workers}", flush=True)

    pool = mp.Pool(a.probe_workers, initializer=_init,
                   initargs=(a.dataset, a.signal_gain, in_ids, insyn, a.ticks, settle))

    # checkpoint indices: 0 (untrained), then every K, plus the final one
    ckpts = sorted(set([0] + list(range(a.checkpoint_every, len(seq) + 1, a.checkpoint_every)) + [len(seq)]))

    ck_ntrain, ck_weight, ck_desc, ck_metrics = [], [], {m: [] for m in modes}, {m: [] for m in modes}
    t0 = time.time()

    def probe(n_seen):
        saved = _make_picklable(net)
        try:
            net_bytes = pickle.dumps(net, protocol=pickle.HIGHEST_PROTOCOL)
        finally:
            _restore(net, saved)
        w = total_weight(net)
        ck_ntrain.append(int(n_seen)); ck_weight.append(w)
        for m in modes:
            tasks = [(net_bytes, int(ti), m) for ti in test_idx]
            descs = np.stack(pool.map(_probe_one, tasks))  # (n_test, D)
            ck_desc[m].append(descs.astype(np.float32))
            acc, margin, bw = loo_nearest_centroid(descs, test_lab)
            ck_metrics[m].append((acc, margin, bw))
            print(f"[{a.tag}] ckpt n_train={n_seen:4d} W={w:7.1f} {m:6s} "
                  f"NC_acc={acc:.3f} margin={margin:+.3f} betw/within={bw:.2f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    probe(0)  # baseline: untrained network
    ci = 1
    for gpos, (idx, c, k) in enumerate(seq, start=1):
        img, _ = ds[idx]
        sig = image_to_signals(img, in_ids, insyn, net, dc)
        for t in range(a.ticks):
            core.send_batch_signals(sig); core.do_tick()
        if ci < len(ckpts) and gpos == ckpts[ci]:
            probe(gpos); ci += 1

    pool.close(); pool.join()

    np.savez_compressed(
        os.path.join(a.out, f"{a.tag}.npz"),
        n_train=np.asarray(ck_ntrain), weight=np.asarray(ck_weight, np.float32),
        test_lab=test_lab, test_idx=test_idx, classes=np.asarray(classes),
        modes=np.asarray(modes), ticks=a.ticks, settle=settle,
        **{f"desc_{m}": np.asarray(ck_desc[m], np.float32) for m in modes},
        **{f"metrics_{m}": np.asarray(ck_metrics[m], np.float32) for m in modes},
    )
    meta = dict(tag=a.tag, net=a.net, dataset=a.dataset, classes=classes,
                train_per_class=a.train_per_class, test_per_class=a.test_per_class,
                ticks=a.ticks, settle=settle, checkpoint_every=a.checkpoint_every,
                modes=modes, neurons=N, n_checkpoints=len(ck_ntrain),
                secs=round(time.time() - t0))
    json.dump(meta, open(os.path.join(a.out, f"{a.tag}_meta.json"), "w"), indent=2)
    for m in modes:
        arr = np.asarray(ck_metrics[m])
        print(f"[{a.tag}] {m}: NC_acc {arr[0,0]:.3f}->{arr[-1,0]:.3f} "
              f"margin {arr[0,1]:+.3f}->{arr[-1,1]:+.3f} (chance {1/len(classes):.2f})", flush=True)
    print(f"[{a.tag}] DONE {len(ck_ntrain)} checkpoints ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
