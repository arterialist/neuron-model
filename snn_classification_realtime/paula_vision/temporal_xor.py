"""DECISIVE test of NONLINEAR temporal computation. Two bits at separated times t1<t2:
bit1 -> channel A0/A1 at t1; bit2 -> channel B0/B1 at t2. Label = bit1 XOR bit2 (NOT linearly
separable). Readout in the late window (pure memory, leak=0). Baselines (raw spike counts,
leaky-trace) are LINEAR in the inputs -> cannot compute XOR -> ~chance. A nonlinear reservoir
that MIXES the temporally-separated bits can. Also compares recurrent vs no-recurrence (feedforward
can't mix bits landing on different neurons)."""
import argparse, json, time
import numpy as np, multiprocessing as mp
from neuron.nn_core import NNCore
from neuron.network_config import NetworkConfig
from neuron.neuron import Neuron
from snn_classification_realtime.paula_vision.separability_probe import ridge_acc

_G = {}
_KGLOBAL = [5]
_FEATMODE = ['S']
A0, A1, B0, B1 = 100, 200, 300, 400  # bit1 channels, bit2 channels

def make(n, t1hi, dlo, dhi, seed):
    rng = np.random.RandomState(seed); tr = []
    for i in range(n):
        b1 = rng.randint(2); b2 = rng.randint(2)
        t1 = int(rng.randint(5, t1hi)); t2 = t1 + int(rng.randint(dlo, dhi))
        ev = [(t1, A1 if b1 else A0), (t2, B1 if b2 else B0)]
        tr.append((ev, b1 ^ b2))
    rng.shuffle(tr); return tr

def _init(net_path, chan_path, T, rw, K=5, fmode='S'):
    net = NetworkConfig.load_network_config(net_path, neuron_class=Neuron)
    core = NNCore(); core.neural_net = net; core.set_log_level("CRITICAL")
    for nu in net.network.neurons.values(): nu.params.eta_post = 0.0; nu.params.eta_retro = 0.0
    chan = {int(k): v for k, v in json.load(open(chan_path)).items()}
    _G.update(net=net, core=core, chan=chan, T=T, rw=rw, K=K, fmode=fmode, neurons=list(net.network.neurons.values()))

def _feat(trial):
    ev, lab = trial; net = _G["net"]; core = _G["core"]; chan = _G["chan"]; T = _G["T"]; rw = _G["rw"]
    neurons = _G["neurons"]; N = len(neurons)
    net.reset_simulation(); core.state.current_tick = 0; net.current_tick = 0
    bt = {}
    for (t, c) in ev: bt.setdefault(t, []).append(c)
    K = _G["K"]; win = np.zeros((K, N), np.float32); cnt = np.zeros(K); leak = 0
    for t in range(T):
        if t in bt:
            for c in bt[t]:
                for (nid, sid, w) in chan.get(c, []): net.set_external_input(nid, sid, 8.0)
        core.do_tick()
        k = min(K - 1, t * K // T)
        win[k] += np.array([(nu.O if _G['fmode']=='rate' else nu.S) for nu in neurons], np.float32); cnt[k] += 1
    win /= np.maximum(cnt[:, None], 1)
    return np.concatenate(win), lab, leak

def paula(net, chan, T, rw, trials, workers, dump=""):
    with mp.Pool(workers, initializer=_init, initargs=(net, chan, T, rw, _KGLOBAL[0], _FEATMODE[0])) as pool:
        out = pool.map(_feat, trials)
    X = np.stack([o[0] for o in out]); y = np.array([o[1] for o in out]); leak = sum(o[2] for o in out)
    if dump:
        np.savez(dump, X=X, y=y); print(f"[xor] dumped feats {X.shape} -> {dump}", flush=True)
    n2 = len(X) // 2
    return ridge_acc(X[:n2], y[:n2], X[n2:], y[n2:]), leak

def baselines(trials, T, rw):
    # raw counts (4 chan) and leaky trace, both LINEAR
    def feat(decay):
        X = []; y = []
        for ev, lab in trials:
            bt = {}
            for (t, c) in ev: bt.setdefault(t, []).append(c)
            trace = np.zeros(700); acc = np.zeros(700); nr = 0
            for t in range(T):
                trace *= decay
                if t in bt:
                    for c in bt[t]: trace[c] += 1.0
                if t >= T - rw: acc += trace; nr += 1
            X.append(acc / nr); y.append(lab)
        X = np.stack(X); y = np.array(y); n2 = len(X) // 2
        return ridge_acc(X[:n2], y[:n2], X[n2:], y[n2:])
    return {"raw(decay0)": feat(0.0), "leaky.95": feat(0.95), "leaky.99": feat(0.99)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True); ap.add_argument("--chan", required=True)
    ap.add_argument("--netnorec"); ap.add_argument("--channorec")
    ap.add_argument("--T", type=int, default=120); ap.add_argument("--rw", type=int, default=15)
    ap.add_argument("--n", type=int, default=600); ap.add_argument("--t1hi", type=int, default=60)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--dump", default="")
    ap.add_argument("--feat", default="S", choices=["S","rate"])
    a = ap.parse_args()
    _KGLOBAL[0] = a.K; _FEATMODE[0] = a.feat
    trials = make(a.n, a.t1hi, 3, 12, seed=777)
    ndist = len(np.unique([f"{ev}" for ev, _ in trials]))
    print(f"[xor] temporal XOR (2-class, XOR NOT linearly separable). n={a.n} distinct-configs={ndist}", flush=True)
    b = baselines(trials, a.T, a.rw)
    print(f"[xor] LINEAR BASELINES (must ~=0.5 if truly XOR): " + " ".join(f"{k}={v:.3f}" for k, v in b.items()), flush=True)
    acc, leak = paula(a.net, a.chan, a.T, a.rw, trials, a.workers, dump=a.dump)
    print(f"[xor] PAULA RECURRENT reservoir: acc={acc:.3f} leak={leak} (chance 0.5)", flush=True)
    if a.netnorec:
        acc2, _ = paula(a.netnorec, a.channorec, a.T, a.rw, trials, a.workers)
        print(f"[xor] PAULA NO-RECURRENCE reservoir: acc={acc2:.3f} (feedforward can't mix bits => should fail)", flush=True)

if __name__ == "__main__":
    main()
