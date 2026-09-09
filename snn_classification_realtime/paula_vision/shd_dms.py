"""LEAKAGE-FREE real-data nonlinear-temporal test: delayed-match-to-sample (DMS) over REAL SHD
spikes. Each trial presents digit A (window 1) then digit B (window 2, time-shifted). Label =
(classA == classB) match/non-match, BALANCED so every digit appears equally in match & non-match
=> identity CANNOT predict the label (no identity leakage, unlike the fL-XOR-fH construction). A
linear readout of the superimposed A+B features cannot compute 'same' without COMPARING => ~0.5.
A reservoir that holds A in memory and nonlinearly compares to B can. Real spikes, comparison label."""
import argparse, json
import numpy as np, multiprocessing as mp
from collections import defaultdict
from neuron.nn_core import NNCore
from neuron.network_config import NetworkConfig
from neuron.neuron import Neuron
from snn_classification_realtime.paula_vision.shd_online import load_shd

_S = {}

def make_pairs(samples, labs, n_pairs, seed):
    rng = np.random.RandomState(seed)
    by = defaultdict(list)
    for i, l in enumerate(labs): by[int(l)].append(i)
    classes = sorted(by); pairs = []
    for _ in range(n_pairs // 2):
        # MATCH: same class
        c = rng.choice(classes); a, b = rng.choice(by[c], 2, replace=len(by[c]) < 2)
        pairs.append((a, b, 1))
        # NON-MATCH: different classes
        c1, c2 = rng.choice(classes, 2, replace=False)
        pairs.append((rng.choice(by[c1]), rng.choice(by[c2]), 0))
    rng.shuffle(pairs); return pairs

def _init(net_path, chan_path, W, gap, dt, settle, K, samples):
    net = NetworkConfig.load_network_config(net_path, neuron_class=Neuron)
    core = NNCore(); core.neural_net = net; core.set_log_level("CRITICAL")
    for nu in net.network.neurons.values(): nu.params.eta_post = 0.0; nu.params.eta_retro = 0.0
    chan = {int(k): v for k, v in json.load(open(chan_path)).items()}
    _S.update(net=net, core=core, chan=chan, W=W, gap=gap, dt=dt, settle=settle, K=K,
              samples=samples, neurons=list(net.network.neurons.values()))

def _binstream(sample, dt, W, offset):
    times, units = sample; bins = defaultdict(lambda: defaultdict(float)); tmax = W * dt
    for tm, u in zip(times, units):
        if tm < tmax: bins[offset + int(tm / dt)][int(u)] += 1.0
    return bins

def _feat(pair):
    a, b, lab = pair; net = _S["net"]; core = _S["core"]; chan = _S["chan"]
    W = _S["W"]; gap = _S["gap"]; dt = _S["dt"]; K = _S["K"]; neurons = _S["neurons"]; N = len(neurons)
    sa = _S["samples"][a]; sb = _S["samples"][b]
    ba = _binstream(sa, dt, W, 0); bb = _binstream(sb, dt, W, W + gap)
    allb = defaultdict(lambda: defaultdict(float))
    for d in (ba, bb):
        for t, chs in d.items():
            for c, v in chs.items(): allb[t][c] += v
    net.reset_simulation(); core.state.current_tick = 0; net.current_tick = 0
    total = 2 * W + gap + _S["settle"]
    # read ONLY the late window (after B) => must remember A to compare
    rw0 = W + gap
    win = np.zeros((K, N), np.float32); cnt = np.zeros(K)
    for t in range(total):
        if t in allb:
            for c, ct in allb[t].items():
                for (nid, sid, w) in chan.get(c, []): net.set_external_input(nid, sid, float(ct))
        core.do_tick()
        if t >= rw0:
            k = min(K - 1, (t - rw0) * K // (total - rw0)); win[k] += np.array([nu.S for nu in neurons], np.float32); cnt[k] += 1
    win /= np.maximum(cnt[:, None], 1); return np.concatenate(win), lab

def raw_pair_feat(pair, samples, dt, W, gap, K, nch=700):
    a, b, lab = pair
    ba = _binstream(samples[a], dt, W, 0); bb = _binstream(samples[b], dt, W, W + gap)
    total = 2 * W + gap; win = np.zeros((K, nch)); cnt = np.zeros(K)
    for d in (ba, bb):
        for t, chs in d.items():
            k = min(K - 1, t * K // total)
            for c, v in chs.items(): win[k, c] += v; cnt[k] += 1
    return win.ravel(), lab

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True); ap.add_argument("--chan", required=True)
    ap.add_argument("--W", type=int, default=60); ap.add_argument("--gap", type=int, default=10)
    ap.add_argument("--dt", type=float, default=0.007); ap.add_argument("--settle", type=int, default=8)
    ap.add_argument("--K", type=int, default=5); ap.add_argument("--nc", type=int, default=10)
    ap.add_argument("--npair", type=int, default=1600); ap.add_argument("--nte_pair", type=int, default=600)
    ap.add_argument("--per", type=int, default=100); ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--tag", default="dms")
    a = ap.parse_args()
    classes = list(range(a.nc))
    tr_s, tr_y = load_shd("data/shd/shd_train.h5", a.per, classes)
    te_s, te_y = load_shd("data/shd/shd_test.h5", a.per // 2, classes)
    tr_pairs = make_pairs(tr_s, tr_y, a.npair, seed=1)
    te_pairs = make_pairs(te_s, te_y, a.nte_pair, seed=2)
    print(f"[{a.tag}] DMS real SHD: {a.nc}-class, train_pairs={len(tr_pairs)} test_pairs={len(te_pairs)} "
          f"match-rate tr={np.mean([p[2] for p in tr_pairs]):.2f} (chance 0.5)", flush=True)
    from snn_classification_realtime.paula_vision.separability_probe import ridge_acc
    from snn_classification_realtime.paula_vision.online_decoder import online_reward_competitive
    ytr = np.array([p[2] for p in tr_pairs]); yte = np.array([p[2] for p in te_pairs])
    # PAULA (parallel); samples passed to workers via initializer
    with mp.Pool(a.workers, initializer=_init, initargs=(a.net, a.chan, a.W, a.gap, a.dt, a.settle, a.K, tr_s)) as pool:
        Xtr = np.stack([o[0] for o in pool.map(_feat, tr_pairs)])
    with mp.Pool(a.workers, initializer=_init, initargs=(a.net, a.chan, a.W, a.gap, a.dt, a.settle, a.K, te_s)) as pool:
        Xte = np.stack([o[0] for o in pool.map(_feat, te_pairs)])
    cf = ridge_acc(Xtr, ytr, Xte, yte); on, cv = online_reward_competitive(Xtr, ytr, Xte, yte, 2, epochs=15)
    Rtr = np.stack([raw_pair_feat(p, tr_s, a.dt, a.W, a.gap, a.K)[0] for p in tr_pairs])
    Rte = np.stack([raw_pair_feat(p, te_s, a.dt, a.W, a.gap, a.K)[0] for p in te_pairs])
    cfr = ridge_acc(Rtr, ytr, Rte, yte); onr, cvr = online_reward_competitive(Rtr, ytr, Rte, yte, 2, epochs=15)
    print(f"[{a.tag}] RAW baseline (superimposed pair, can't compare -> ~0.5): ridge={cfr:.3f} online={cvr[-1]:.3f}", flush=True)
    print(f"[{a.tag}] PAULA reservoir (holds A, compares B): ridge={cf:.3f} online={cv[-1]:.3f} (chance 0.5)", flush=True)
    print(f"[{a.tag}] SUBSTRATE ADDS on REAL DMS: ridge={cf-cfr:+.3f} online={cv[-1]-cvr[-1]:+.3f}", flush=True)
