"""Parallelized SHD reservoir evaluation. Readout = MULTI-WINDOW temporal feature: split the T
input ticks into K windows, take per-neuron mean-S in each -> (K*N) feature capturing the
reservoir's trajectory (not just the decayed tail). Frozen reservoir (diagnostic). Closed-form
separability first (internal split), then train/test readout on a winner."""
import argparse, json, time
import numpy as np, h5py
from collections import defaultdict
import multiprocessing as mp
from neuron.nn_core import NNCore
from neuron.network_config import NetworkConfig
from neuron.neuron import Neuron

_G = {}


def _load(path, n_per_class, classes):
    f = h5py.File(path, "r"); labels = f["labels"][:]
    by = defaultdict(list)
    for i in range(len(labels)):
        c = int(labels[i])
        if (classes is None or c in classes) and len(by[c]) < n_per_class:
            by[c].append(i)
    idx = [i for c in sorted(by) for i in by[c]]
    times = f["spikes"]["times"]; units = f["spikes"]["units"]
    S = [(np.asarray(times[i]), np.asarray(units[i]), int(labels[i])) for i in idx]
    return S


def _init(net_path, chan_path, T, dt, settle, K, split):
    net = NetworkConfig.load_network_config(net_path, neuron_class=Neuron)
    core = NNCore(); core.neural_net = net; core.set_log_level("CRITICAL")
    for nu in net.network.neurons.values():
        nu.params.eta_post = 0.0; nu.params.eta_retro = 0.0
    chan = {int(k): v for k, v in json.load(open(chan_path)).items()}
    _G.update(net=net, core=core, chan=chan, T=T, dt=dt, settle=settle, K=K,
              neurons=list(net.network.neurons.values()))


def _feat(sample):
    times, units, lab = sample
    net = _G["net"]; core = _G["core"]; chan = _G["chan"]; T = _G["T"]; dt = _G["dt"]; K = _G["K"]
    neurons = _G["neurons"]; N = len(neurons)
    bins = defaultdict(lambda: defaultdict(float))
    for tm, u in zip(times, units):
        if tm < T * dt: bins[int(tm / dt)][int(u)] += 1.0
    net.reset_simulation(); core.state.current_tick = 0; net.current_tick = 0
    win = np.zeros((K, N), np.float32); cnt = np.zeros(K)
    for t in range(T):
        if t in bins:
            for c, v in bins[t].items():
                for (nid, sid, w) in chan.get(c, []):
                    net.set_external_input(nid, sid, float(v))
        core.do_tick()
        k = min(K - 1, t * K // T)
        win[k] += np.array([nu.S for nu in neurons], np.float32); cnt[k] += 1
    win /= np.maximum(cnt[:, None], 1)
    return np.concatenate(win), lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True); ap.add_argument("--chan", required=True)
    ap.add_argument("--T", type=int, default=100); ap.add_argument("--dt", type=float, default=0.007)
    ap.add_argument("--settle", type=int, default=5); ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--per-class", type=int, default=30); ap.add_argument("--classes", default="")
    ap.add_argument("--workers", type=int, default=12); ap.add_argument("--tag", default="shd")
    a = ap.parse_args()
    classes = [int(x) for x in a.classes.split(",")] if a.classes else None
    ncl = len(classes) if classes else 20
    samples = _load("data/shd/shd_test.h5", a.per_class, classes)
    print(f"[{a.tag}] loaded {len(samples)} samples, {ncl} classes, K={a.K} N-feat={a.K}*N", flush=True)
    init_args = (a.net, a.chan, a.T, a.dt, a.settle, a.K, "test")
    t0 = time.time()
    with mp.Pool(a.workers, initializer=_init, initargs=init_args) as pool:
        out = pool.map(_feat, samples)
    X = np.stack([o[0] for o in out]); y = np.array([o[1] for o in out])
    print(f"[{a.tag}] recorded {len(X)} feat-dim={X.shape[1]} ({time.time()-t0:.0f}s)", flush=True)
    from snn_classification_realtime.paula_vision.separability_probe import separability
    # also test per-window (which timepoint carries signal) + full
    N = X.shape[1] // a.K
    for k in range(a.K):
        s = separability(X[:, k*N:(k+1)*N], y, seed=0)
        print(f"  window {k}: ncc={s['ncc']:.3f} ridge={s['ridge']:.3f}", flush=True)
    s = separability(X, y, seed=0)
    print(f"[{a.tag}] FULL multi-window (chance {1/ncl:.3f}): ncc={s['ncc']:.3f} lda={s['lda']:.3f} ridge={s['ridge']:.3f}", flush=True)


if __name__ == "__main__":
    main()
