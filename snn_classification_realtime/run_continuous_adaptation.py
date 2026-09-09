"""Continuous slow-adaptation run: does sustained same-class exposure reshape a PAULA
substrate so successive samples fall into progressively more similar attractors?

The network is loaded and reset ONCE, then samples are presented sequentially (3k ticks each)
with NO reset between them -- so synaptic plasticity, t_ref homeostasis, F_avg and the
neuromodulator state all carry across samples. This is the slow-adaptation regime (the same
mechanism that, measured per-sample, was the 'state-bleed' confound; here it IS the object of
study).

Per sample we compute a compact settled-window attractor descriptor ONLINE (no full-trajectory
dumps -> tiny storage) and log total synaptic weight. Presentation order is 'blocked'
(class 0 x K, class 1 x K, ...) or 'interleaved' (round-robin) -- the interleaved run is the
control that separates class-specific generalization from monotonic time-drift.

Plasticity mode / rh_decay come from the network JSON global_params (set by the arm generator);
kappa=0 => unsupervised. No neuron.py edits. CLI-only.
"""
import argparse
import json
import os
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


def total_weight(net):
    return float(sum(psp.u_i.info for nu in net.network.neurons.values()
                     for psp in nu.postsynaptic_points.values()))


def _spec(sig, k=KSPEC):
    mag = np.abs(np.fft.rfft(sig - sig.mean()))
    if len(mag) < 2:
        return np.zeros(k, np.float32)
    return np.interp(np.linspace(0, len(mag) - 1, k), np.arange(len(mag)), mag).astype(np.float32)


def build_order(ds, classes, per_class, order, seed):
    by = {c: [] for c in classes}
    for i in np.random.RandomState(seed).permutation(len(ds)):
        _, y = ds[int(i)]; y = int(y)
        if y in by and len(by[y]) < per_class:
            by[y].append(int(i))
        if all(len(v) >= per_class for v in by.values()):
            break
    if order == "blocked":
        seq = [(i, c, k) for c in classes for k, i in enumerate(by[c])]
    else:  # interleaved round-robin; block-position k = round index
        seq = []
        for k in range(per_class):
            for c in classes:
                if k < len(by[c]):
                    seq.append((by[c][k], c, k))
    return seq, by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True, help="arm network JSON (plasticity_mode set in global_params)")
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--classes", default="0,1,2")
    ap.add_argument("--per-class", type=int, default=150)
    ap.add_argument("--ticks", type=int, default=3000)
    ap.add_argument("--settle-frac", type=float, default=0.4)
    ap.add_argument("--order", choices=["blocked", "interleaved"], default="blocked")
    ap.add_argument("--signal-gain", type=float, default=0.336)
    ap.add_argument("--exemplars-per-class", type=int, default=6, help="full trajectories kept per class (even spread)")
    ap.add_argument("--out", default="foveation_results/continuous_adapt")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    classes = [int(x) for x in a.classes.split(",")]
    settle = int(a.ticks * a.settle_frac)

    net = NetworkConfig.load_network_config(a.net, neuron_class=Neuron)
    core = NNCore(); core.neural_net = net; core.set_log_level("CRITICAL")
    layers = infer_layers_from_metadata(net)
    in_ids, insyn = determine_input_mapping(net, layers)
    dc = load_dataset_by_name(a.dataset); dc.signal_gain = a.signal_gain
    ds = dc.dataset
    seq, by = build_order(ds, classes, a.per_class, a.order, a.seed)
    # exemplar positions per class (evenly spread across the block)
    ex_pos = {c: set(np.linspace(0, len(by[c]) - 1, a.exemplars_per_class).astype(int).tolist()) for c in classes}

    net.reset_simulation(); core.state.current_tick = 0; net.current_tick = 0
    N = len(net.network.neurons)
    w0 = total_weight(net)

    desc, dcls, dpos, dgpos, wtrace = [], [], [], [], []
    ex_traj, ex_meta = [], []
    slc = None
    t0 = time.time()
    for gpos, (idx, c, k) in enumerate(seq):
        img, y = ds[idx]
        sig = image_to_signals(img, in_ids, insyn, net, dc)
        Sbuf = np.empty((a.ticks - settle, N), np.float32)
        Tbuf = np.empty((a.ticks - settle, N), np.float32)
        Fbuf = np.empty((a.ticks - settle, N), np.float32)
        for t in range(a.ticks):
            core.send_batch_signals(sig); core.do_tick()
            if t >= settle:
                r = t - settle
                Sbuf[r] = [nu.S for nu in net.network.neurons.values()]
                Tbuf[r] = [nu.t_ref for nu in net.network.neurons.values()]
                Fbuf[r] = [nu.F_avg for nu in net.network.neurons.values()]
        # compact settled attractor descriptor
        mS, mT, mF = Sbuf.mean(0), Tbuf.mean(0), Fbuf.mean(0)
        sdS = Sbuf.std(0)
        popS = Sbuf.mean(1)
        d = np.concatenate([mS, mT, sdS, _spec(popS)]).astype(np.float32)
        if slc is None:
            off = 0; slc = {}
            for nm, p in [("meanS", mS), ("mean_tref", mT), ("stdS", sdS), ("spec", _spec(popS))]:
                slc[nm] = [off, off + len(p)]; off += len(p)
        desc.append(d); dcls.append(int(c)); dpos.append(int(k)); dgpos.append(int(gpos))
        wtrace.append(total_weight(net))
        if k in ex_pos[c]:
            ex_traj.append(np.stack([popS, Tbuf.mean(1)], 1).astype(np.float32))  # (T,2) pop phase
            ex_meta.append((int(c), int(k), int(gpos)))
        if gpos % 20 == 0:
            print(f"[{a.tag}] {gpos}/{len(seq)} cls={c} k={k} W={wtrace[-1]:.0f} ({time.time()-t0:.0f}s)", flush=True)

    np.savez_compressed(
        os.path.join(a.out, f"{a.tag}.npz"),
        desc=np.asarray(desc, np.float32), cls=np.asarray(dcls), pos=np.asarray(dpos),
        gpos=np.asarray(dgpos), wtrace=np.asarray(wtrace, np.float32), w0=w0,
        ex_traj=np.asarray(ex_traj, np.float32) if ex_traj else np.zeros((0, a.ticks - settle, 2), np.float32),
        ex_meta=np.asarray(ex_meta, np.int64) if ex_meta else np.zeros((0, 3), np.int64),
        slc=json.dumps(slc), classes=np.asarray(classes), settle=settle, ticks=a.ticks, order=a.order,
    )
    meta = dict(tag=a.tag, net=a.net, dataset=a.dataset, classes=classes, per_class=a.per_class,
                ticks=a.ticks, settle=settle, order=a.order, neurons=N, w0=round(w0, 1),
                w_final=round(wtrace[-1], 1), n=len(desc), secs=round(time.time() - t0))
    json.dump(meta, open(os.path.join(a.out, f"{a.tag}_meta.json"), "w"), indent=2)
    print(f"[{a.tag}] DONE n={len(desc)} W {w0:.0f}->{wtrace[-1]:.0f} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
