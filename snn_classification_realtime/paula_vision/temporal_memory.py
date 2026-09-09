"""LEAK-FREE test of the substrate's temporal MEMORY. Two pulses (chanA,chanB) at t1<t2 early;
class = ORDER. Readout = reservoir mean-S ONLY in the LATE window [T-rw, T], AFTER both pulses
end -> zero direct input there, so any order info is PURE MEMORY (verified: input-in-window=0).
Vary t1 -> gap = (T-rw)-(t1+delay) = how long order must be remembered. Frozen reservoir."""
import argparse, json, time
import numpy as np, multiprocessing as mp
from neuron.nn_core import NNCore
from neuron.network_config import NetworkConfig
from neuron.neuron import Neuron

_G = {}


def _init(net_path, chan_path, T, rw):
    net = NetworkConfig.load_network_config(net_path, neuron_class=Neuron)
    core = NNCore(); core.neural_net = net; core.set_log_level("CRITICAL")
    for nu in net.network.neurons.values():
        nu.params.eta_post = 0.0; nu.params.eta_retro = 0.0
    chan = {int(k): v for k, v in json.load(open(chan_path)).items()}
    _G.update(net=net, core=core, chan=chan, T=T, rw=rw, neurons=list(net.network.neurons.values()))


def _feat(trial):
    ev, lab = trial
    net = _G["net"]; core = _G["core"]; chan = _G["chan"]; T = _G["T"]; rw = _G["rw"]
    neurons = _G["neurons"]; N = len(neurons)
    net.reset_simulation(); core.state.current_tick = 0; net.current_tick = 0
    bt = {}
    for (t, c) in ev: bt.setdefault(t, []).append(c)
    acc = np.zeros(N, np.float32); nrec = 0; leak = 0
    for t in range(T):
        if t in bt:
            for c in bt[t]:
                for (nid, sid, w) in chan.get(c, []): net.set_external_input(nid, sid, 8.0)
                if t >= T - rw: leak += 1
        core.do_tick()
        if t >= T - rw:
            acc += np.array([nu.S for nu in neurons], np.float32); nrec += 1
    return acc / max(nrec, 1), lab, leak


def make(n, t1lo, t1hi, dlo, dhi, chanA=100, chanB=400, seed=0):
    """RANDOM t1 and delay PER TRIAL -> within-class variability -> non-trivial: the reservoir
    must extract ORDER invariant to absolute timing/position and generalize to unseen positions."""
    rng = np.random.RandomState(seed); tr = []
    for i in range(n):
        lab = i % 2
        t1 = int(rng.randint(t1lo, t1hi)); t2 = t1 + int(rng.randint(dlo, dhi))
        tr.append(([(t1, chanA), (t2, chanB)] if lab == 0 else [(t1, chanB), (t2, chanA)], lab))
    rng.shuffle(tr); return tr


def main():
    from snn_classification_realtime.paula_vision.separability_probe import ridge_acc
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True); ap.add_argument("--chan", required=True)
    ap.add_argument("--T", type=int, default=120); ap.add_argument("--rw", type=int, default=15)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--chanA", type=int, default=100)
    ap.add_argument("--chanB", type=int, default=400)
    ap.add_argument("--dlo", type=int, default=3)
    ap.add_argument("--dhi", type=int, default=12)
    ap.add_argument("--t1only", type=int, default=0)
    ap.add_argument("--only-t1hi", type=int, default=0); ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    print(f"[tmem] LEAK-FREE: readout=last {a.rw} ticks (pure memory, input-in-window must=0). "
          f"gap=(T-rw)-(t1+3)=memory duration:", flush=True)
    # pulses placed randomly in [5, t1hi]; readout is the fixed late window (pure memory).
    _sweep = [a.only_t1hi] if a.only_t1hi>0 else [30, 60, 90]
    for t1hi in _sweep:
        if t1hi + a.dhi >= a.T - a.rw - 2: continue
        trials = make(a.n, 5, t1hi, a.dlo, a.dhi, chanA=a.chanA, chanB=a.chanB, seed=2000 + t1hi)
        t0 = time.time()
        with mp.Pool(a.workers, initializer=_init, initargs=(a.net, a.chan, a.T, a.rw)) as pool:
            out = pool.map(_feat, trials)
        X = np.stack([o[0] for o in out]); y = np.array([o[1] for o in out]); leak = sum(o[2] for o in out)
        # distinct feature vectors? (trivial if only 2) -- report to catch the no-variability trap
        ndistinct = len(np.unique(np.round(X,3), axis=0))
        n2 = len(X) // 2
        acc = ridge_acc(X[:n2], y[:n2], X[n2:], y[n2:])
        print(f"  t1<={t1hi:3d} (rand pos+delay): acc={acc:.3f} (chance 0.5) leak={leak} "
              f"distinct-feats={ndistinct}/{len(X)} [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
