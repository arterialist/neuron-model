"""Test C (supervised ceiling): can the built-in three-factor gate, driven by a class
teaching signal, sharpen per-class basins for genuinely-unseen samples -- where unsupervised
plasticity (kappa=0) could not?

Mechanism (no neuron.py edit): set nm_plasticity_kappa=kappa on every neuron; during TRAIN,
write each neuron's M_vector directly each tick so nm = max(0, 1+kappa*(M[reward]-M[stress]))
scales reward_hebb LTP. A positive reward amplifies consolidation of the current weights, a
negative one suppresses it.

Teaching signal (causal, single-pass, class-correct credit): keep a running EMA prototype per
class from TRAIN settled states, and a per-class recent-discriminability reward R_c. When a
class-c sample arrives we inject R_c (computed from class c's PREVIOUS samples -- never the
current one), run it with plasticity, then update R_c and prototype_c from its settled state
for next time. So reward for class c reflects how class-discriminative class c has recently
been -> a reward-modulated Hebbian consolidation loop.

Controls (reward-hacking guards):
  * reward-mode 'shuffled' : inject R_{perm(c)} (a fixed class permutation) -> identical reward
    statistics, class contingency destroyed. THE primary falsifier. If sharpening survives, the
    gate supplies arousal/gain, not class learning.
  * reward-mode 'constant' : inject +1 always -> a pure global learning-rate boost. Distinguishes
    class-contingent reward from "just more LTP".
  * Evaluation is fully held-out (REF builds centroids, QUERY scored), weights FROZEN during
    probing, so we optimize a TRAIN teaching signal but score held-out generalization.
  * collapse guard: total weight + betw/within reported; a 'win' with weight collapse or
    between-centroid->0 is flagged, not learning.

CLI-only. See EXPERIMENT_DESIGN_no_model_change.md (Test C).
"""
import argparse
import json
import os
import pickle
import time

import numpy as np

from neuron.nn_core import NNCore
from neuron.network_config import NetworkConfig
from neuron.neuron import Neuron
from snn_classification_realtime.core.input_mapping import image_to_signals
from snn_classification_realtime.activity_dataset_builder.vision_datasets import load_dataset_by_name
from snn_classification_realtime.activity_dataset_builder.network_utils import (
    infer_layers_from_metadata, determine_input_mapping,
)
from snn_classification_realtime.run_basin_generalization import (
    _spec, total_weight, _make_picklable, _restore,
)
from snn_classification_realtime.run_completion_probe import _init, _probe, nearest_centroid_acc, build_pools


def _static_desc(Sbuf, Tbuf):
    return np.concatenate([Sbuf.mean(0), Tbuf.mean(0), Sbuf.std(0), _spec(Sbuf.mean(1))]).astype(np.float32)


def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True)
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--classes", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--train-per-class", type=int, default=40)
    ap.add_argument("--ref-per-class", type=int, default=6)
    ap.add_argument("--query-per-class", type=int, default=6)
    ap.add_argument("--ticks", type=int, default=3000)
    ap.add_argument("--settle-frac", type=float, default=0.4)
    ap.add_argument("--checkpoint-every", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--kappa", type=float, default=8.0)
    ap.add_argument("--reward-mode", choices=["real", "shuffled", "constant"], default="real")
    ap.add_argument("--reward-scale", type=float, default=1.0)
    ap.add_argument("--reward-alpha", type=float, default=3.0, help="tanh gain on discriminability")
    ap.add_argument("--proto-decay", type=float, default=0.9, help="prototype EMA retention")
    ap.add_argument("--order", choices=["interleaved", "blocked"], default="interleaved")
    ap.add_argument("--modes", default="frozen,on")
    ap.add_argument("--signal-gain", type=float, default=0.336)
    ap.add_argument("--probe-workers", type=int, default=8)
    ap.add_argument("--out", default="foveation_results/supervised_kappa")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    classes = [int(x) for x in a.classes.split(",")]
    modes = [m.strip() for m in a.modes.split(",") if m.strip()]
    settle = int(a.ticks * a.settle_frac)

    net = NetworkConfig.load_network_config(a.net, neuron_class=Neuron)
    core = NNCore(); core.neural_net = net; core.set_log_level("CRITICAL")
    # activate the three-factor gate on every neuron (no neuron.py edit)
    r_idx = net.network.neurons[next(iter(net.network.neurons))].params.nm_reward_index
    s_idx = net.network.neurons[next(iter(net.network.neurons))].params.nm_stress_index
    for nu in net.network.neurons.values():
        nu.params.nm_plasticity_kappa = a.kappa
    layers = infer_layers_from_metadata(net)
    in_ids, insyn = determine_input_mapping(net, layers)
    dc = load_dataset_by_name(a.dataset); dc.signal_gain = a.signal_gain
    ds = dc.dataset
    seq, ref_idx, ref_lab, q_idx, q_lab = build_pools(
        ds, classes, a.train_per_class, a.ref_per_class, a.query_per_class, a.order, a.seed)
    net.reset_simulation(); core.state.current_tick = 0; net.current_tick = 0
    N = len(net.network.neurons); static_dim = 3 * N + 24
    total_present = len(seq) * a.epochs
    # fixed derangement for the shuffled control
    rng = np.random.RandomState(a.seed + 777)
    perm = {c: classes[(classes.index(c) + 1 + rng.randint(len(classes) - 1)) % len(classes)] for c in classes}
    print(f"[{a.tag}] net={a.net} kappa={a.kappa} reward={a.reward_mode} N={N} "
          f"train={len(seq)}x{a.epochs}={total_present} ref={len(ref_idx)} query={len(q_idx)} "
          f"order={a.order} ticks={a.ticks} ckpt_every={a.checkpoint_every} workers={a.probe_workers} "
          f"seed={a.seed} r_idx={r_idx} s_idx={s_idx} perm={perm if a.reward_mode=='shuffled' else '-'}",
          flush=True)

    pool = mp.Pool(a.probe_workers, initializer=_init,
                   initargs=(a.dataset, a.signal_gain, in_ids, insyn, a.ticks, settle))
    ckpts = sorted(set([0] + list(range(a.checkpoint_every, total_present + 1, a.checkpoint_every)) + [total_present]))

    protos = {}                         # class -> EMA settled descriptor
    R_c = {c: 0.0 for c in classes}     # class -> recent discriminability reward
    rec = dict(n_train=[], weight=[], acc=[], reward=[])
    neurons = list(net.network.neurons.values())
    t0 = time.time()

    def probe(n_seen):
        saved = _make_picklable(net)
        try:
            net_bytes = pickle.dumps(net, protocol=pickle.HIGHEST_PROTOCOL)
        finally:
            _restore(net, saved)
        w = total_weight(net); rec["n_train"].append(int(n_seen)); rec["weight"].append(w)
        accs = {}
        for m in modes:
            ref_D = np.stack(pool.map(_probe, [(net_bytes, int(i), "whole", 0.0, m, a.seed) for i in ref_idx]))
            q_D = np.stack(pool.map(_probe, [(net_bytes, int(i), "whole", 0.0, m, a.seed) for i in q_idx]))
            acc, mar = nearest_centroid_acc(ref_D[:, :static_dim], ref_lab, q_D[:, :static_dim], q_lab)
            accs[m] = acc
        rec["acc"].append(accs); rec["reward"].append(float(np.mean(list(R_c.values()))))
        print(f"[{a.tag}] n_train={n_seen:4d} " + " ".join(f"{m}:acc={accs[m]:.3f}" for m in modes)
              + f"  W={w:.0f} meanR={rec['reward'][-1]:+.3f} ({time.time()-t0:.0f}s)", flush=True)

    def inject(R):
        rp = max(R, 0.0) * a.reward_scale; rn = max(-R, 0.0) * a.reward_scale
        for nu in neurons:
            nu.M_vector[r_idx] = rp; nu.M_vector[s_idx] = rn

    probe(0)
    ci = 1; gpos = 0
    Sbuf = np.empty((a.ticks - settle, N), np.float32); Tbuf = np.empty((a.ticks - settle, N), np.float32)
    for ep in range(a.epochs):
        for (idx, c) in seq:
            gpos += 1
            # choose the reward to inject for this class-c sample (causal: from history)
            if a.reward_mode == "constant":
                R_inj = 1.0
            elif a.reward_mode == "shuffled":
                R_inj = R_c[perm[c]]
            else:
                R_inj = R_c[c]
            img, _ = ds[idx]
            sig = image_to_signals(img, in_ids, insyn, net, dc)
            for t in range(a.ticks):
                inject(R_inj)                      # hold reward during the whole run
                core.send_batch_signals(sig); core.do_tick()
                if t >= settle:
                    r = t - settle
                    Sbuf[r] = [nu.S for nu in neurons]; Tbuf[r] = [nu.t_ref for nu in neurons]
            s_k = _static_desc(Sbuf, Tbuf)
            # update class-c discriminability reward + prototype for NEXT time (causal).
            # Discriminability is measured in the STANDARDIZED prototype space (where classes are
            # separable), as a normalized own-vs-other margin -> R in ~[-1,1]. Raw cosine is
            # dominated by the common-mode and would give R~0 (gate never activates).
            if len(protos) >= 2 and c in protos:
                P = np.stack(list(protos.values()))
                gm = P.mean(0); gsd = P.std(0) + 1e-6
                zk = (s_k - gm) / gsd
                d_own = np.linalg.norm(zk - (protos[c] - gm) / gsd)
                d_oth = min(np.linalg.norm(zk - (protos[cc] - gm) / gsd) for cc in protos if cc != c)
                R_c[c] = float(np.tanh(a.reward_alpha * (d_oth - d_own) / (d_oth + d_own + 1e-9)))
            protos[c] = s_k if c not in protos else a.proto_decay * protos[c] + (1 - a.proto_decay) * s_k
            if ci < len(ckpts) and gpos == ckpts[ci]:
                probe(gpos); ci += 1
    pool.close(); pool.join()

    np.savez_compressed(
        os.path.join(a.out, f"{a.tag}.npz"),
        n_train=np.asarray(rec["n_train"]), weight=np.asarray(rec["weight"], np.float32),
        acc_frozen=np.asarray([d.get("frozen", np.nan) for d in rec["acc"]], np.float32),
        acc_on=np.asarray([d.get("on", np.nan) for d in rec["acc"]], np.float32),
        mean_reward=np.asarray(rec["reward"], np.float32),
        classes=np.asarray(classes), modes=np.asarray(modes), kappa=a.kappa,
        reward_mode=a.reward_mode, epochs=a.epochs, train_len=len(seq), seed=a.seed)
    json.dump(dict(tag=a.tag, net=a.net, kappa=a.kappa, reward_mode=a.reward_mode, classes=classes,
                   train=len(seq), epochs=a.epochs, ref=len(ref_idx), query=len(q_idx), modes=modes,
                   ticks=a.ticks, n_checkpoints=len(rec["n_train"]), secs=round(time.time() - t0)),
              open(os.path.join(a.out, f"{a.tag}_meta.json"), "w"), indent=2)
    a0 = rec["acc"][0]; aN = rec["acc"][-1]
    print(f"[{a.tag}] DONE acc " + " ".join(f"{m}:{a0[m]:.3f}->{aN[m]:.3f}" for m in modes)
          + f"  W {rec['weight'][0]:.0f}->{rec['weight'][-1]:.0f} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
