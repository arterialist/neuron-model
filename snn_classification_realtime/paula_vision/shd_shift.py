"""Test-time input-rate SHIFT robustness (grounded in adaptive-threshold SNN literature: adaptive
LIF tolerates unseen shifts in mean input strength; frozen nonlinearities cannot). Protocol: fit a
decoder ONCE on CLEAN SHD (gain=1), FREEZE it, then evaluate on rate-SHIFTED test sets (input drive
x G, a distribution shift never seen at train). Same frozen decoder scores every representation, so
the test isolates the SUBSTRATE. Compare PAULA rec0 (intrinsic adaptive threshold) vs a random-ReLU
expander (no gain control) vs raw. Hypothesis: PAULA degrades LESS under shift because its adaptive
threshold renormalizes firing; random-ReLU collapses (features scale ~G, out of the decoder's
learned range). If PAULA robustness > random-ReLU, that is a substrate advantage no random projection
can replicate. 12-worker parallel feature extraction."""
import argparse, json
import numpy as np, multiprocessing as mp, h5py
from collections import defaultdict
from neuron.nn_core import NNCore
from neuron.network_config import NetworkConfig
from neuron.neuron import Neuron

_S = {}

def load_shd(path, n_per_class, classes):
    f = h5py.File(path, "r"); labels = f["labels"][:]
    times = f["spikes"]["times"]; units = f["spikes"]["units"]
    by = defaultdict(list)
    for i in range(len(labels)):
        if labels[i] in classes and len(by[labels[i]]) < n_per_class: by[labels[i]].append(i)
    idx = [i for c in sorted(by) for i in by[c]]
    return [(np.asarray(times[i]), np.asarray(units[i])) for i in idx], \
           np.array([int(labels[i]) for i in idx])

def _init(net_path, chan_path, T, dt, settle, K):
    net = NetworkConfig.load_network_config(net_path, neuron_class=Neuron)
    core = NNCore(); core.neural_net = net; core.set_log_level("CRITICAL")
    for nu in net.network.neurons.values(): nu.params.eta_post = 0.0; nu.params.eta_retro = 0.0
    chan = {int(k): v for k, v in json.load(open(chan_path)).items()}
    _S.update(net=net, core=core, chan=chan, T=T, dt=dt, settle=settle, K=K,
              neurons=list(net.network.neurons.values()))

def _feat(arg):
    (times, units), gain = arg
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
                for (nid, sid, w) in chan.get(c, []): net.set_external_input(nid, sid, float(ct) * gain)
        core.do_tick()
        k = min(K - 1, t * K // total); win[k] += np.array([nu.S for nu in neurons], np.float32); cnt[k] += 1
    win /= np.maximum(cnt[:, None], 1); return np.concatenate(win)

def raw_feat(sample, T, dt, K, gain, nch=700):
    times, units = sample; win = np.zeros((K, nch)); tmax = T * dt
    for tm, u in zip(times, units):
        if tm < tmax:
            k = min(K - 1, int(tm / dt) * K // T); win[k, int(u)] += 1.0
    return win.ravel() * gain

def ridge_fit(Xtr, ytr, lam=1.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Z = (Xtr - mu) / sd
    classes = np.unique(ytr)
    Y = np.stack([(ytr == c).astype(np.float64) for c in classes], 1)
    A = Z.T @ Z + lam * np.eye(Z.shape[1])
    W = np.linalg.solve(A, Z.T @ Y)
    return (mu, sd, W, classes)

def ridge_apply(model, X, y):
    mu, sd, W, classes = model
    pred = classes[(((X - mu) / sd) @ W).argmax(1)]
    return float((pred == y).mean())

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True); ap.add_argument("--chan", required=True)
    ap.add_argument("--T", type=int, default=100); ap.add_argument("--dt", type=float, default=0.007)
    ap.add_argument("--settle", type=int, default=8); ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--nc", type=int, default=20); ap.add_argument("--ntr", type=int, default=60)
    ap.add_argument("--nte", type=int, default=25); ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--gains", default="1,3,8")
    a = ap.parse_args()
    gains = [float(g) for g in a.gains.split(",")]
    classes = list(range(a.nc))
    tr_s, tr_y = load_shd("data/shd/shd_train.h5", a.ntr, classes)
    te_s, te_y = load_shd("data/shd/shd_test.h5", a.nte, classes)
    print(f"[shift] train={len(tr_s)} test={len(te_s)} nc={a.nc} chance={1/a.nc:.3f} gains={gains}", flush=True)

    init = (a.net, a.chan, a.T, a.dt, a.settle, a.K)
    with mp.Pool(a.workers, initializer=_init, initargs=init) as pool:
        # PAULA: train clean (gain 1), test at each gain
        PXtr = np.stack(pool.map(_feat, [(s, 1.0) for s in tr_s]))
        pmodel = ridge_fit(PXtr, tr_y)
        paula = {}
        for g in gains:
            PXte = np.stack(pool.map(_feat, [(s, g) for s in te_s]))
            paula[g] = ridge_apply(pmodel, PXte, te_y)
    # RAW + random-ReLU: train clean, test at each gain
    RXtr = np.stack([raw_feat(s, a.T, a.dt, a.K, 1.0) for s in tr_s]).astype(np.float64)
    rmodel = ridge_fit(RXtr, tr_y)
    # random-ReLU expander fixed at train time
    rng = np.random.RandomState(0); D_out = a.K * 800
    mu0, sd0 = RXtr.mean(0), RXtr.std(0) + 1e-8
    W_rp = rng.randn(RXtr.shape[1], D_out) / np.sqrt(RXtr.shape[1])
    def relu_feat(R): return np.maximum(((R - mu0) / sd0) @ W_rp, 0.0)
    rlmodel = ridge_fit(relu_feat(RXtr), tr_y)
    raw_r, relu_r = {}, {}
    for g in gains:
        RXte = np.stack([raw_feat(s, a.T, a.dt, a.K, g) for s in te_s]).astype(np.float64)
        raw_r[g] = ridge_apply(rmodel, RXte, te_y)
        relu_r[g] = ridge_apply(rlmodel, relu_feat(RXte), te_y)

    print("[shift] gain :  " + "  ".join(f"G={g:<4g}" for g in gains), flush=True)
    print("[shift] PAULA:  " + "  ".join(f"{paula[g]:.3f} " for g in gains), flush=True)
    print("[shift] rReLU:  " + "  ".join(f"{relu_r[g]:.3f} " for g in gains), flush=True)
    print("[shift] raw  :  " + "  ".join(f"{raw_r[g]:.3f} " for g in gains), flush=True)
    g0 = gains[0]
    print("[shift] RETENTION (acc_G / acc_clean):", flush=True)
    print("[shift]   PAULA: " + "  ".join(f"{paula[g]/paula[g0]:.2f}" for g in gains), flush=True)
    print("[shift]   rReLU: " + "  ".join(f"{relu_r[g]/relu_r[g0]:.2f}" for g in gains), flush=True)
    print("@@@SHIFT DONE@@@", flush=True)
