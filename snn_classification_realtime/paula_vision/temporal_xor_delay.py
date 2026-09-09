"""DECISIVE test of whether PAULA's RECURRENT DYNAMICS provide MEMORY beyond a static kernel.
Delayed XOR: bit1@t1, bit2@t2 (both EARLY, in [5,W]); then a DELAY of D empty ticks; then read the
reservoir membrane ONLY in the LATE window [W+D, T]. Label = bit1 XOR bit2.
- A static nonlinear kernel on the LATE-window INPUT sees EMPTY input (spikes were early) -> chance.
- A NO-RECURRENCE reservoir has only membrane leak (short memory) -> fails as D grows.
- A RECURRENT reservoir can hold the computed XOR across the delay -> succeeds -> proves the
  recurrent dynamics carry information a static late readout cannot access. Sweep D."""
import argparse, json
import numpy as np, multiprocessing as mp
from neuron.nn_core import NNCore
from neuron.network_config import NetworkConfig
from neuron.neuron import Neuron
from snn_classification_realtime.paula_vision.separability_probe import ridge_acc
from snn_classification_realtime.paula_vision.online_decoder import online_reward_competitive

_G = {}
A0, A1, B0, B1 = 100, 200, 300, 400

def make(n, W, D, rw, seed):
    rng = np.random.RandomState(seed); tr = []
    T = W + D + rw
    for _ in range(n):
        b1 = rng.randint(2); b2 = rng.randint(2)
        t1 = int(rng.randint(5, W // 2)); t2 = int(rng.randint(W // 2, W - 2))
        ev = [(t1, A1 if b1 else A0), (t2, B1 if b2 else B0)]
        tr.append((ev, b1 ^ b2))
    rng.shuffle(tr); return tr, T

def _init(net_path, chan_path, T, W, D, rw, K):
    net = NetworkConfig.load_network_config(net_path, neuron_class=Neuron)
    core = NNCore(); core.neural_net = net; core.set_log_level("CRITICAL")
    for nu in net.network.neurons.values(): nu.params.eta_post = 0.0; nu.params.eta_retro = 0.0
    chan = {int(k): v for k, v in json.load(open(chan_path)).items()}
    _G.update(net=net, core=core, chan=chan, T=T, W=W, D=D, rw=rw, K=K,
              neurons=list(net.network.neurons.values()))

def _feat(trial):
    ev, lab = trial; net = _G["net"]; core = _G["core"]; chan = _G["chan"]
    T = _G["T"]; W = _G["W"]; D = _G["D"]; K = _G["K"]; neurons = _G["neurons"]; N = len(neurons)
    net.reset_simulation(); core.state.current_tick = 0; net.current_tick = 0
    bt = {}
    for (t, c) in ev: bt.setdefault(t, []).append(c)
    late0 = W + D  # read ONLY [late0, T)
    win = np.zeros((K, N), np.float32); cnt = np.zeros(K)
    for t in range(T):
        if t in bt:
            for c in bt[t]:
                for (nid, sid, w) in chan.get(c, []): net.set_external_input(nid, sid, 8.0)
        core.do_tick()
        if t >= late0:
            k = min(K - 1, (t - late0) * K // max(T - late0, 1)); win[k] += np.array([nu.S for nu in neurons], np.float32); cnt[k] += 1
    win /= np.maximum(cnt[:, None], 1); return np.concatenate(win), lab

def run_net(net, chan, T, W, D, rw, K, trials, workers, noise=0.0):
    with mp.Pool(workers, initializer=_init, initargs=(net, chan, T, W, D, rw, K)) as pool:
        out = pool.map(_feat, trials)
    X = np.stack([o[0] for o in out]); y = np.array([o[1] for o in out]); n2 = len(X) // 2
    if noise > 0:
        rng = np.random.RandomState(0); X = X + rng.randn(*X.shape) * (noise * X.std(0, keepdims=True) + 1e-9)
    cf = ridge_acc(X[:n2], y[:n2], X[n2:], y[n2:])
    on, cv = online_reward_competitive(X[:n2], y[:n2], X[n2:], y[n2:], 2, epochs=15)
    return cf, cv[-1]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True); ap.add_argument("--chan", required=True)
    ap.add_argument("--netnorec", required=True); ap.add_argument("--channorec", required=True)
    ap.add_argument("--W", type=int, default=40); ap.add_argument("--rw", type=int, default=15)
    ap.add_argument("--K", type=int, default=4); ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--delays", default="0,20,60,120"); ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--noise", type=float, default=0.0)
    a = ap.parse_args()
    print(f"[xord] delayed XOR, LATE-ONLY readout. bits early in [5,{a.W}], read [W+D, T). chance 0.5", flush=True)
    for D in [int(x) for x in a.delays.split(",")]:
        trials, T = make(a.n, a.W, D, a.rw, seed=777 + D)
        cf_r, on_r = run_net(a.net, a.chan, T, a.W, D, a.rw, a.K, trials, a.workers, a.noise)
        cf_n, on_n = run_net(a.netnorec, a.channorec, T, a.W, D, a.rw, a.K, trials, a.workers, a.noise)
        print(f"[xord] D={D:3d} (mem-span): RECURRENT ridge={cf_r:.3f} online={on_r:.3f} | "
              f"NO-REC ridge={cf_n:.3f} online={on_n:.3f} | recurrent-adds={cf_r-cf_n:+.3f}", flush=True)
    print("@@@XORD DONE@@@", flush=True)
