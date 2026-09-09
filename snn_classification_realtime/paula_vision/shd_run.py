"""Feed SHD spike streams into a frozen recurrent PAULA reservoir and read out the integrated
state with a closed-form classifier. Research process: first verify the reservoir has real
temporal MEMORY + reproducibility (not just last-bin response), then measure accuracy."""
import argparse, json, time
import numpy as np
import h5py
from collections import defaultdict
from neuron.nn_core import NNCore
from neuron.network_config import NetworkConfig
from neuron.neuron import Neuron

_S = {}


def load_shd(path, n_per_class=None, classes=None):
    f = h5py.File(path, "r")
    labels = f["labels"][:]
    times = f["spikes"]["times"]; units = f["spikes"]["units"]
    idx = list(range(len(labels)))
    if classes is not None:
        idx = [i for i in idx if labels[i] in classes]
    if n_per_class is not None:
        by = defaultdict(list)
        for i in idx:
            if len(by[labels[i]]) < n_per_class: by[labels[i]].append(i)
        idx = [i for c in sorted(by) for i in by[c]]
    samples = [(np.asarray(times[i]), np.asarray(units[i])) for i in idx]
    labs = np.array([int(labels[i]) for i in idx])
    return samples, labs


def init(net_path, chan_path, T, dt, settle, freeze=True):
    net = NetworkConfig.load_network_config(net_path, neuron_class=Neuron)
    core = NNCore(); core.neural_net = net; core.set_log_level("CRITICAL")
    if freeze:
        for nu in net.network.neurons.values():
            nu.params.eta_post = 0.0; nu.params.eta_retro = 0.0
    chan = {int(k): v for k, v in json.load(open(chan_path)).items()}
    _S.update(net=net, core=core, chan=chan, T=T, dt=dt, settle=settle,
              neurons=list(net.network.neurons.values()))


def record(sample, mask=None):
    """sample=(times,units). mask: optional (t0,t1) fraction window to KEEP (memory test)."""
    times, units = sample
    net = _S["net"]; core = _S["core"]; chan = _S["chan"]; T = _S["T"]; dt = _S["dt"]
    neurons = _S["neurons"]
    # bin spikes: tick t gets channels active in [t*dt,(t+1)*dt)
    bins = defaultdict(lambda: defaultdict(float))  # tick -> channel -> count
    tmax = T * dt
    for tm, u in zip(times, units):
        if tm >= tmax: continue
        t = int(tm / dt)
        bins[t][int(u)] += 1.0
    if mask is not None:
        lo, hi = int(mask[0] * T), int(mask[1] * T)
        bins = {t: v for t, v in bins.items() if lo <= t < hi}
    net.reset_simulation(); core.state.current_tick = 0; net.current_tick = 0
    Ssum = np.zeros(len(neurons), np.float32); nrec = 0
    total_ticks = T + _S["settle"]
    for t in range(total_ticks):
        if t in bins:
            for c, cnt in bins[t].items():
                for (nid, sid, w) in chan.get(c, []):
                    net.set_external_input(nid, sid, float(cnt))
        core.do_tick()
        if t >= T - 1:  # accumulate over the tail (integrated state)
            Ssum += np.array([nu.S for nu in neurons], np.float32); nrec += 1
    return Ssum / max(nrec, 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True); ap.add_argument("--chan", required=True)
    ap.add_argument("--T", type=int, default=100); ap.add_argument("--dt", type=float, default=0.007)
    ap.add_argument("--settle", type=int, default=10)
    ap.add_argument("--mode", default="diag", choices=["diag"])
    a = ap.parse_args()
    init(a.net, a.chan, a.T, a.dt, a.settle)
    samples, labs = load_shd("data/shd/shd_test.h5", n_per_class=3, classes=list(range(5)))
    print(f"loaded {len(samples)} samples", flush=True)
    # 1. reproducibility (frozen -> should be deterministic)
    r1 = record(samples[0]); r2 = record(samples[0])
    print(f"REPRODUCIBILITY: corr={np.corrcoef(r1,r2)[0,1]:.3f} (frozen->~1.0)", flush=True)
    # 2. memory: full vs first-half-only vs last-half-only
    full = record(samples[0]); early = record(samples[0], mask=(0,0.5)); late = record(samples[0], mask=(0.5,1.0))
    print(f"MEMORY: corr(full,early_only)={np.corrcoef(full,early)[0,1]:.3f} "
          f"corr(full,late_only)={np.corrcoef(full,late)[0,1]:.3f}", flush=True)
    print("  (full!=late_only => reservoir REMEMBERS early input; full~=late_only => no long memory)", flush=True)
    # 3. quick separability across 5 classes
    States=np.stack([record(s) for s in samples]); 
    from snn_classification_realtime.paula_vision.separability_probe import separability
    sep=separability(States, labs, seed=0)
    print(f"QUICK SEP (5-class, chance 0.20, {len(samples)} samples): ncc={sep['ncc']:.3f} lda={sep['lda']:.3f} ridge={sep['ridge']:.3f}", flush=True)
