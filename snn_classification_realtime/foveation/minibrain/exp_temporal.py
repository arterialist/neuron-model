"""SPATIOTEMPORAL readout -- the load-bearing test (user, 2026-07-07/08).

In the PAULA model the representation is the DYNAMICS over time (the dynamical
regime / oscillatory pattern), NOT a mean over ticks. The arch sweep read the MEAN
of [S,F_avg,O] -> a static-lens LOWER BOUND that collapses exactly the temporal
structure that IS the representation. This experiment records the tick-by-tick
trajectory O(t)/S(t) of the substrate for each glimpse and classifies THAT, so we can
compare -- on the SAME frozen substrate -- reading the representation the wrong way
(mean) vs the right way (the trajectory).

Readouts compared (per layer group: input / pool / all):
  mean_lin   : logistic on the mean rate over the window          (the sweep's lens)
  temporal_lin : logistic on temporal features (time-binned rates + population
                 power-spectrum + activity envelope) -- preserves WHEN, still linear
  temporal_gru : a small GRU trained on the raw (T,N) spike trajectory -- a genuine
                 dynamical classifier (the analogue of the pipeline's SNN readout)

If the representation lives in the dynamics, temporal_* >> mean_lin, and the gap should
be LARGEST in the recurrent POOL (whose mean is near-chance but whose oscillatory
dynamics are rich). That is the prediction this experiment tests.
"""
from __future__ import annotations

import argparse, os, json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from snn_classification_realtime.activity_dataset_builder.vision_datasets import load_dataset_by_name
from snn_classification_realtime.foveation.minibrain.core import MiniBrainConfig, MiniBrain

_Cs = (0.0005, 0.002, 0.01, 0.05)


def best_lin(X, y, ntr):
    mu, sd = X[:ntr].mean(0), X[:ntr].std(0) + 1e-6; Xn = (X - mu) / sd
    return max(LogisticRegression(max_iter=1500, C=C).fit(Xn[:ntr], y[:ntr])
               .score(Xn[ntr:], y[ntr:]) for C in _Cs)


def record_trajectories(brain, ds, idxs, dwell, tag="", every=25):
    """Return O (spikes), S (membrane), TR (t_ref) trajectories, each (n, T, N), over
    `dwell` ticks per glimpse, for ALL neurons -- the SAME (ticks, neurons) continuous
    shape (firings / avg_S / avg_t_ref) the real activity pipeline records. t_ref is
    the state variable that engraves the regime remnant (user), so it is captured too.
    The substrate is NOT reset between images (continuous time); learning off in probe.
    Prints live progress every `every` images (tail -f the .out to watch)."""
    brain.sub.set_learning(False)
    per = max(1, brain.cfg.input_period); N = brain.sub.num_neurons
    Ot, St, Tt, Y = [], [], [], []
    n = len(idxs); t0 = time.time()
    for k, i in enumerate(idxs):
        img, y = ds[int(i)]; Y.append(int(y))
        sig = brain.sub.patch_to_signals(brain._encode(img))
        O = np.zeros((dwell, N), np.float32); S = np.zeros((dwell, N), np.float32)
        TR = np.zeros((dwell, N), np.float32)
        for t in range(dwell):
            st = brain.sub.step((sig if t % per == 0 else []) + brain._tonic)
            O[t] = (st.O > 0).astype(np.float32); S[t] = st.S; TR[t] = st.t_ref
        Ot.append(O); St.append(S); Tt.append(TR)
        if (k + 1) % every == 0:
            el = time.time() - t0; rate = (k + 1) * dwell / el
            eta = (n - k - 1) * dwell / max(rate, 1e-6)
            print(f"  [{tag}] recording {k+1}/{n} imgs | {rate:.0f} ticks/s | "
                  f"rec-ETA {eta:.0f}s", flush=True)
    brain.sub.set_learning(True)
    return np.array(Ot), np.array(St), np.array(Tt), np.array(Y)


def temporal_features(O, S, TR, n_bins=8, n_freq=16):
    """(n,T,N) -> temporal feature matrix preserving WHEN activity happens:
       - time-binned spike rates (n_bins x N): the coarse temporal profile
       - per-neuron mean membrane + mean t_ref (2N): standing state / regime remnant
       - population power spectrum (n_freq): oscillatory content of pop activity
       - population activity envelope over bins (n_bins): rise/settle shape."""
    n, T, N = O.shape
    b = np.array_split(np.arange(T), n_bins)
    binned = np.concatenate([O[:, idx, :].mean(1) for idx in b], axis=1)   # (n, n_bins*N)
    memb = S.mean(1); tref = TR.mean(1)                                     # (n, N) each
    pop = O.mean(2)                                                         # (n, T) population rate
    spec = np.abs(np.fft.rfft(pop - pop.mean(1, keepdims=True), axis=1))[:, 1:n_freq + 1]
    env = np.stack([pop[:, idx].mean(1) for idx in b], axis=1)              # (n, n_bins)
    return np.concatenate([binned, memb, tref, spec, env], axis=1)


class RNNReadout:
    """Small GRU or LSTM over the raw (T,N) spike trajectory -- a genuine dynamical
    classifier (the GRU/LSTM the user was advised to use instead of an SNN, on the
    continuous (ticks,neurons) shape, so the mean does not destroy the signal)."""
    def __init__(self, N, n_classes=10, hidden=64, rnn="gru", seed=0):
        import torch
        torch.manual_seed(seed); torch.set_num_threads(1)
        self.torch = torch; self.rnn_type = rnn
        self.dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        Cell = torch.nn.LSTM if rnn == "lstm" else torch.nn.GRU
        self.rnn = Cell(N, hidden, batch_first=True).to(self.dev)
        self.head = torch.nn.Linear(hidden, n_classes).to(self.dev)

    def _last(self, X):
        out = self.rnn(X)[0]           # (B,T,H); works for both GRU and LSTM
        return out[:, -1]

    def fit(self, O, y, epochs=60, lr=3e-3, val=0.2):
        t = self.torch
        X = t.tensor(O, dtype=t.float32).to(self.dev); Y = t.tensor(y, dtype=t.long).to(self.dev)
        n = len(y); ntr = int(n * (1 - val)); idx = t.randperm(n)
        tr, te = idx[:ntr], idx[ntr:]
        opt = t.optim.Adam(list(self.rnn.parameters()) + list(self.head.parameters()), lr=lr)
        lossf = t.nn.CrossEntropyLoss(); best = 0.0
        for ep in range(epochs):
            self.rnn.train(); opt.zero_grad()
            loss = lossf(self.head(self._last(X[tr])), Y[tr]); loss.backward(); opt.step()
            if ep % 3 == 0:
                self.rnn.eval()
                with t.no_grad():
                    acc = (self.head(self._last(X[te])).argmax(1) == Y[te]).float().mean().item()
                best = max(best, acc)
        return best


def make_cfg(arch, dataset, out_dir, seed):
    substrate = "conv" if arch == "conv" else "reservoir"
    wiring = "retinotopic" if arch == "reservoir_retino" else "random"
    return MiniBrainConfig(dataset_name=dataset, dwell=500, seed=seed, substrate_type=substrate,
                           wiring=wiring, conv_bank="rich", readout="all", output_dir=out_dir)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="reservoir_retino", choices=["reservoir_random", "reservoir_retino", "conv"])
    p.add_argument("--dataset", default="cifar10_grayscale")
    p.add_argument("--images", type=int, default=600)
    p.add_argument("--traj-dwell", type=int, default=300)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--gru", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="temporal")
    p.add_argument("--out", default="foveation_results/minibrain/temporal")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    ds = load_dataset_by_name(args.dataset, train=True).dataset
    order = np.random.RandomState(args.seed).randint(0, len(ds), size=args.images + 60)
    cfg = make_cfg(args.arch, args.dataset, os.path.join(args.out, f"net_{args.tag}_{os.getpid()}"), args.seed)
    brain = MiniBrain(cfg, load_dataset_by_name(args.dataset, train=True))
    # warmup
    per = max(1, cfg.input_period); done = 0; k = 0
    brain.sub.set_learning(False)
    while done < args.warmup:
        sig = brain.sub.patch_to_signals(brain._encode(ds[int(order[k % 50])][0]))
        for tk in range(cfg.dwell):
            brain.sub.step((sig if tk % per == 0 else []) + brain._tonic); done += 1
            if done >= args.warmup: break
        k += 1
    brain.sub.set_learning(True)
    O, S, TR, Y = record_trajectories(brain, ds, order[:args.images], args.traj_dwell, tag=args.tag)
    print(f"[{time.time()-t0:.0f}s] recorded {O.shape} trajectories (O,S,t_ref)", flush=True)
    lay = brain.sub.layer_of_pos
    groups = {"input": lay == min(brain.sub.layer_indices),
              "pool": lay == max(brain.sub.layer_indices),
              "all": np.ones(len(lay), bool)}
    ntr = int(args.images * 0.7)
    res = {}
    for g, mask in groups.items():
        Og, Sg, Tg = O[:, :, mask], S[:, :, mask], TR[:, :, mask]
        mean_feat = np.concatenate([Og.mean(1), Sg.mean(1), Tg.mean(1)], axis=1)
        temp_feat = temporal_features(Og, Sg, Tg)
        r = dict(mean_lin=round(best_lin(mean_feat, Y, ntr), 3),
                 temporal_lin=round(best_lin(temp_feat, Y, ntr), 3),
                 n_neurons=int(mask.sum()))
        if args.gru:
            r["gru"] = round(RNNReadout(int(mask.sum()), rnn="gru", seed=args.seed).fit(Og, Y), 3)
            r["lstm"] = round(RNNReadout(int(mask.sum()), rnn="lstm", seed=args.seed).fit(Og, Y), 3)
        res[g] = r
        print(f"  [{g:5s}] N={r['n_neurons']:3d}  mean_lin {r['mean_lin']:.3f}  "
              f"temporal_lin {r['temporal_lin']:.3f}"
              + (f"  gru {r['gru']:.3f}  lstm {r['lstm']:.3f}" if args.gru else ""), flush=True)
    out = dict(exp="temporal", tag=args.tag, arch=args.arch, dataset=args.dataset,
               seed=args.seed, traj_dwell=args.traj_dwell, images=args.images, groups=res, complete=True)
    json.dump(out, open(os.path.join(args.out, f"{args.tag}_seed{args.seed}.json"), "w"))
    print(f"[{time.time()-t0:.0f}s] {args.tag} done", flush=True)


if __name__ == "__main__":
    main()
