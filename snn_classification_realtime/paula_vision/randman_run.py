"""Feed T-Randman (pure spike-timing) into a frozen PAULA reservoir + online decoder. Baselines:
RATE (spike counts, ignores timing -> must be chance) and RAW-TEMPORAL (windowed spatiotemporal
spike pattern, linear). Decisive: does the PAULA reservoir ADD over raw-temporal, esp. at hard
(high dim_manifold) settings where the timing->class map is more nonlinear."""
import argparse, json
import numpy as np, multiprocessing as mp
from neuron.nn_core import NNCore
from neuron.network_config import NetworkConfig
from neuron.neuron import Neuron
from snn_classification_realtime.paula_vision.randman import generate
from snn_classification_realtime.paula_vision.separability_probe import ridge_acc, lda_acc
from snn_classification_realtime.paula_vision.online_decoder import online_reward_competitive

_G = {}

def _init(net_path, chan_path, T, settle, K, gain):
    net = NetworkConfig.load_network_config(net_path, neuron_class=Neuron)
    core = NNCore(); core.neural_net = net; core.set_log_level("CRITICAL")
    for nu in net.network.neurons.values(): nu.params.eta_post = 0.0; nu.params.eta_retro = 0.0
    chan = {int(k): v for k, v in json.load(open(chan_path)).items()}
    _G.update(net=net, core=core, chan=chan, T=T, settle=settle, K=K, gain=gain,
              neurons=list(net.network.neurons.values()))

def _feat(times):
    net = _G["net"]; core = _G["core"]; chan = _G["chan"]; T = _G["T"]; K = _G["K"]; gain = _G["gain"]
    neurons = _G["neurons"]; N = len(neurons)
    ev = {}  # tick -> [channels]
    for i, tf in enumerate(times):
        tk = int(round(tf * (T - 1))); ev.setdefault(tk, []).append(i)
    net.reset_simulation(); core.state.current_tick = 0; net.current_tick = 0
    total = T + _G["settle"]; win = np.zeros((K, N), np.float32); cnt = np.zeros(K)
    for t in range(total):
        if t in ev:
            for c in ev[t]:
                for (nid, sid, w) in chan.get(c, []): net.set_external_input(nid, sid, gain)
        core.do_tick()
        k = min(K - 1, t * K // total); win[k] += np.array([nu.S for nu in neurons], np.float32); cnt[k] += 1
    win /= np.maximum(cnt[:, None], 1); return np.concatenate(win)

def raw_temporal(Xt, T, K):
    n, nu = Xt.shape; feats = np.zeros((n, K, nu))
    for i in range(n):
        for u in range(nu):
            k = min(K - 1, int(round(Xt[i, u] * (T - 1))) * K // T); feats[i, k, u] += 1.0
    return feats.reshape(n, -1)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True); ap.add_argument("--chan", required=True)
    ap.add_argument("--T", type=int, default=80); ap.add_argument("--settle", type=int, default=8)
    ap.add_argument("--K", type=int, default=6); ap.add_argument("--gain", type=float, default=8.0)
    ap.add_argument("--nc", type=int, default=10); ap.add_argument("--nu", type=int, default=20)
    ap.add_argument("--dim", type=int, default=1); ap.add_argument("--per", type=int, default=200)
    ap.add_argument("--workers", type=int, default=12); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="rm")
    a = ap.parse_args()
    Xt, y = generate(n_classes=a.nc, n_units=a.nu, dim_manifold=a.dim, n_per_class=a.per, seed=a.seed)
    n2 = len(Xt) // 2
    ytr, yte = y[:n2], y[n2:]
    print(f"[{a.tag}] T-Randman nc={a.nc} nu={a.nu} dim={a.dim} n={len(Xt)} chance={1/a.nc:.3f}", flush=True)
    # RATE baseline (constant -> chance, sanity)
    rate = np.ones((len(Xt), a.nu))
    # RAW-TEMPORAL baseline
    RT = raw_temporal(Xt, a.T, a.K)
    cf_rt = ridge_acc(RT[:n2], ytr, RT[n2:], yte); ld_rt = lda_acc(RT[:n2], ytr, RT[n2:], yte)
    on_rt, cv_rt = online_reward_competitive(RT[:n2], ytr, RT[n2:], yte, a.nc, epochs=20)
    print(f"[{a.tag}] RAW-TEMPORAL: ridge={cf_rt:.3f} lda={ld_rt:.3f} online={cv_rt[-1]:.3f}", flush=True)
    # CONTROLS (rule out that PAULA's gain is just dimensionality/overfitting or a generic kernel):
    rp_rng = np.random.RandomState(123); Ddim = 600 * a.K
    P = rp_rng.randn(RT.shape[1], Ddim) / np.sqrt(RT.shape[1])
    RTs = (RT - RT.mean(0)) / (RT.std(0) + 1e-9)
    RPlin = RTs @ P                      # LINEAR random proj: pure dimensionality (must stay ~raw)
    RPnl = np.tanh(RTs @ P)              # static NONLINEAR random-feature kernel (no temporal dynamics)
    cf_lin = ridge_acc(RPlin[:n2], ytr, RPlin[n2:], yte)
    cf_nl = ridge_acc(RPnl[:n2], ytr, RPnl[n2:], yte)
    print(f"[{a.tag}] CONTROLS (raw-temporal->{Ddim}d): lin-randproj ridge={cf_lin:.3f} (dimensionality) "
          f"| nonlin-randfeat ridge={cf_nl:.3f} (static kernel)", flush=True)
    # PAULA reservoir
    with mp.Pool(a.workers, initializer=_init, initargs=(a.net, a.chan, a.T, a.settle, a.K, a.gain)) as pool:
        X = np.stack(pool.map(_feat, list(Xt)))
    cf = ridge_acc(X[:n2], ytr, X[n2:], yte); ld = lda_acc(X[:n2], ytr, X[n2:], yte)
    on, cv = online_reward_competitive(X[:n2], ytr, X[n2:], yte, a.nc, epochs=20)
    print(f"[{a.tag}] PAULA reservoir: ridge={cf:.3f} lda={ld:.3f} online={cv[-1]:.3f}", flush=True)
    print(f"[{a.tag}] SUBSTRATE ADDS over raw-temporal: ridge={cf-cf_rt:+.3f} lda={ld-ld_rt:+.3f} online={cv[-1]-cv_rt[-1]:+.3f}", flush=True)
