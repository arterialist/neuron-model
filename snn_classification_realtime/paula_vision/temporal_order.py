"""Decisive test of the substrate's TEMPORAL computation, with a raw baseline that is chance BY
CONSTRUCTION. Two pulses on two channels at times t1<t2. Class 0: chanA@t1, chanB@t2. Class 1:
chanA@t2, chanB@t1. Raw per-channel spike COUNTS are identical across classes (1 pulse each) ->
raw readout = 0.5 chance. Only a substrate with memory of ORDER can classify. Sweep the delay
(t2-t1) to measure how far back memory reaches. Frozen reservoir (diagnostic of substrate power)."""
import argparse, time
import numpy as np
import multiprocessing as mp
from snn_classification_realtime.paula_vision import shd_eval as SE
from snn_classification_realtime.paula_vision.separability_probe import ridge_acc


def make_trials(n, delay, T, chanA=100, chanB=400, seed=0, pulse=8):
    """Each trial: (events list [(tick, channel)], label). Two classes = order of A,B."""
    rng = np.random.RandomState(seed)
    trials = []
    for i in range(n):
        t1 = rng.randint(2, T - delay - 2); t2 = t1 + delay
        lab = i % 2
        if lab == 0: ev = [(t1, chanA), (t2, chanB)]
        else:        ev = [(t1, chanB), (t2, chanA)]
        trials.append((ev, lab))
    rng.shuffle(trials)
    return trials


def _feat_trial(trial):
    ev, lab = trial
    g = SE._G; net = g["net"]; core = g["core"]; chan = g["chan"]; T = g["T"]; K = g["K"]
    neurons = g["neurons"]; N = len(neurons)
    net.reset_simulation(); core.state.current_tick = 0; net.current_tick = 0
    bt = {}
    for (t, c) in ev: bt.setdefault(t, []).append(c)
    win = np.zeros((K, N), np.float32); cnt = np.zeros(K)
    for t in range(T):
        if t in bt:
            for c in bt[t]:
                for (nid, sid, w) in chan.get(c, []):
                    net.set_external_input(nid, sid, 8.0)   # strong pulse
        core.do_tick()
        k = min(K - 1, t * K // T); win[k] += np.array([nu.S for nu in neurons], np.float32); cnt[k] += 1
    win /= np.maximum(cnt[:, None], 1)
    return np.concatenate(win), lab


def run_delay(net, chan, T, dt, K, delay, n, workers):
    trials = make_trials(n, delay, T, seed=delay)
    init = (net, chan, T, dt, 5, K, "x")
    with mp.Pool(workers, initializer=SE._init, initargs=init) as pool:
        out = pool.map(_feat_trial, trials)
    X = np.stack([o[0] for o in out]); y = np.array([o[1] for o in out])
    n2 = len(X) // 2
    acc = ridge_acc(X[:n2], y[:n2], X[n2:], y[n2:])  # train/test split
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True); ap.add_argument("--chan", required=True)
    ap.add_argument("--T", type=int, default=100); ap.add_argument("--dt", type=float, default=0.007)
    ap.add_argument("--K", type=int, default=6); ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--workers", type=int, default=12); ap.add_argument("--tag", default="torder")
    a = ap.parse_args()
    print(f"[{a.tag}] temporal-order (2-class, raw=0.5 by construction). delay = inter-pulse ticks:", flush=True)
    for delay in [2, 5, 10, 20, 40, 60]:
        t0 = time.time()
        acc = run_delay(a.net, a.chan, a.T, a.dt, a.K, delay, a.n, a.workers)
        print(f"  delay={delay:3d} ticks: reservoir ridge acc={acc:.3f} (chance 0.5) [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
