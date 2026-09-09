"""PAULA on the REAL SHD neuromorphic dataset (Spiking Heidelberg Digits, 20 classes, 700-channel
spike-native). Frozen recurrent PAULA reservoir + ONLINE reward-Hebb decoder. Parallel (12-worker)
feature extraction. Tests 'PAULA works on a real neuromorphic dataset': does the reservoir preserve
class info (ADD/keep separability, not destroy it like static CIFAR) AND can the online decoder learn
it? Baseline = raw spike-count features -> shows whether substrate helps/matches/hurts."""
import argparse, json
import numpy as np, multiprocessing as mp
import h5py
from collections import defaultdict
from neuron.nn_core import NNCore
from neuron.network_config import NetworkConfig
from neuron.neuron import Neuron

_S = {}

def load_shd(path, n_per_class, classes):
    f = h5py.File(path, "r")
    labels = f["labels"][:]
    times = f["spikes"]["times"]; units = f["spikes"]["units"]
    by = defaultdict(list)
    for i in range(len(labels)):
        if labels[i] in classes and len(by[labels[i]]) < n_per_class:
            by[labels[i]].append(i)
    idx = [i for c in sorted(by) for i in by[c]]
    samples = [(np.asarray(times[i]), np.asarray(units[i])) for i in idx]
    labs = np.array([int(labels[i]) for i in idx])
    return samples, labs

def _init(net_path, chan_path, T, dt, settle, K):
    net = NetworkConfig.load_network_config(net_path, neuron_class=Neuron)
    core = NNCore(); core.neural_net = net; core.set_log_level("CRITICAL")
    for nu in net.network.neurons.values(): nu.params.eta_post = 0.0; nu.params.eta_retro = 0.0
    chan = {int(k): v for k, v in json.load(open(chan_path)).items()}
    _S.update(net=net, core=core, chan=chan, T=T, dt=dt, settle=settle, K=K,
              neurons=list(net.network.neurons.values()))

def _feat(sample):
    times, units = sample
    net = _S["net"]; core = _S["core"]; chan = _S["chan"]; T = _S["T"]; dt = _S["dt"]; K = _S["K"]
    neurons = _S["neurons"]; N = len(neurons)
    bins = defaultdict(lambda: defaultdict(float)); tmax = T * dt
    for tm, u in zip(times, units):
        if tm < tmax: bins[int(tm / dt)][int(u)] += 1.0
    net.reset_simulation(); core.state.current_tick = 0; net.current_tick = 0
    win = np.zeros((K, N), np.float32); cnt = np.zeros(K)
    total = T + _S["settle"]
    for t in range(total):
        if t in bins:
            for c, ct in bins[t].items():
                for (nid, sid, w) in chan.get(c, []): net.set_external_input(nid, sid, float(ct))
        core.do_tick()
        k = min(K - 1, t * K // total); win[k] += np.array([nu.S for nu in neurons], np.float32); cnt[k] += 1
    win /= np.maximum(cnt[:, None], 1); return np.concatenate(win)

def raw_feat(sample, T, dt, K, nch=700):
    times, units = sample; tmax = T * dt
    win = np.zeros((K, nch)); cnt = np.zeros(K)
    for tm, u in zip(times, units):
        if tm < tmax:
            k = min(K - 1, int(tm / dt) * K // T); win[k, int(u)] += 1.0; cnt[k] += 1
    return win.ravel()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True); ap.add_argument("--chan", required=True)
    ap.add_argument("--T", type=int, default=100); ap.add_argument("--dt", type=float, default=0.007)
    ap.add_argument("--settle", type=int, default=8); ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--nc", type=int, default=20); ap.add_argument("--ntr", type=int, default=60)
    ap.add_argument("--nte", type=int, default=25); ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    classes = list(range(a.nc))
    tr_s, tr_y = load_shd("data/shd/shd_train.h5", a.ntr, classes)
    te_s, te_y = load_shd("data/shd/shd_test.h5", a.nte, classes)
    print(f"[shd] loaded train={len(tr_s)} test={len(te_s)} classes={a.nc} chance={1/a.nc:.3f}", flush=True)
    from snn_classification_realtime.paula_vision.separability_probe import ridge_acc, ncc_loo
    from snn_classification_realtime.paula_vision.online_decoder import online_reward_competitive
    # PAULA reservoir features (parallel)
    with mp.Pool(a.workers, initializer=_init, initargs=(a.net, a.chan, a.T, a.dt, a.settle, a.K)) as pool:
        Xtr = np.stack(pool.map(_feat, tr_s)); Xte = np.stack(pool.map(_feat, te_s))
    cf = ridge_acc(Xtr, tr_y, Xte, te_y)
    on_acc, curve = online_reward_competitive(Xtr, tr_y, Xte, te_y, a.nc, eta=0.02, rh_decay=0.1, epochs=20)
    print(f"[shd] PAULA reservoir: closed-form-ridge={cf:.3f}  online-decoder={curve[-1]:.3f} (first-pass {on_acc:.3f})", flush=True)
    # raw baseline (same readouts)
    Rtr = np.stack([raw_feat(s, a.T, a.dt, a.K) for s in tr_s]); Rte = np.stack([raw_feat(s, a.T, a.dt, a.K) for s in te_s])
    cfr = ridge_acc(Rtr, tr_y, Rte, te_y)
    onr, curver = online_reward_competitive(Rtr, tr_y, Rte, te_y, a.nc, eta=0.02, rh_decay=0.1, epochs=20)
    print(f"[shd] RAW baseline:      closed-form-ridge={cfr:.3f}  online-decoder={curver[-1]:.3f}", flush=True)
    print(f"[shd] VERDICT: substrate ADDS={cf-cfr:+.3f} (ridge) online={curve[-1]-curver[-1]:+.3f}", flush=True)
