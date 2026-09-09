"""PAULA-ONLY CLASSIFIER (rule-4 'perfect' rung): a PLASTIC PAULA READOUT on a FROZEN reservoir,
learning online via reward_hebb + stress-LTD (the discriminative use of LTD the user flagged).
K class-neurons, each densely wired from every reservoir neuron's presynaptic terminal. Training:
present a sample -> reservoir spikes drive the readout -> REWARD the correct class-neuron (M_reward
-> nm>1 -> LTP on the reservoir units that fired for this class) and STRESS the others (M_stress ->
nm->0 -> decay-LTD, suppressing wrong-class units). Perceptron-like, discriminative, all in PAULA
neurons. Eval: readout neuron with the highest integrated activity = prediction. This is the honest
'learns online' deliverable done with PAULA neurons (not numpy)."""
import argparse, json, copy
import numpy as np
from neuron.nn_core import NNCore
from neuron.network_config import NetworkConfig
from neuron.neuron import Neuron
from snn_classification_realtime.paula_vision.temporal_multiclass import make

def build_combined(res_path, out_path, nclass=4, kappa=1.0, eta=0.02, rh_decay=0.5,
                   r_base=0.5, w0=0.05, gamma=0.9, seed=0):
    rng = np.random.RandomState(seed)
    cfg = json.load(open(res_path))
    # freeze reservoir neurons
    for n in cfg["neurons"]:
        n["params"]["eta_post"] = 0.0; n["params"]["eta_retro"] = 0.0
    res_ids = [n["id"] for n in cfg["neurons"]]
    N = len(res_ids)
    used = set(res_ids); ro_ids = []
    while len(ro_ids) < nclass:
        x = int(rng.randint(0, 2**35))
        if x not in used: used.add(x); ro_ids.append(x)
    for k, rid in enumerate(ro_ids):
        cfg["neurons"].append({"id": rid, "params": {
            "num_inputs": N, "num_neuromodulators": 2, "eta_post": eta, "eta_retro": 0.0,
            "r_base": r_base, "b_base": 1.2, "c": 5, "lambda_param": 20, "p": 1.0,
            "delta_decay": 0.9, "beta_avg": 0.99, "gamma": [gamma, gamma], "w_r": [0, 0],
            "w_b": [0, 0], "w_tref": [0, 0], "plasticity_mode": "reward_hebb", "rh_decay": rh_decay,
            "nm_plasticity_kappa": kappa, "nm_reward_index": 1, "nm_stress_index": 0},
            "metadata": {"layer": 1, "readout_class": k}})
        for i, src in enumerate(res_ids):
            cfg["synaptic_points"].append({"neuron_id": rid, "synapse_id": i, "type": "postsynaptic",
                "distance_to_hillock": 2, "potential": 0.0,
                "u_i": {"info": float(w0 * rng.randn()), "plast": 0.0, "adapt": [0.5, 0.5]}})
            cfg["connections"].append({"source_neuron": src, "source_terminal": 0,
                "target_neuron": rid, "target_synapse": i, "properties": {}})
        cfg["synaptic_points"].append({"neuron_id": rid, "terminal_id": 0, "type": "presynaptic",
            "distance_from_hillock": 3, "u_o": {"info": 1.0, "mod": [0.2, 0.2]}, "u_i_retro": 1.0})
    json.dump(cfg, open(out_path, "w"))
    json.dump({"readout_ids": ro_ids}, open(out_path + ".ro", "w"))
    print(f"built combined net: {N} reservoir + {nclass} plastic readout neurons", flush=True)
    return ro_ids

def run_sample(net, core, chan, res_neurons, ro, ev, T, gain, reward=None, Mset=0.0):
    """Run one sample. If reward=(correct_class,byclass) given, inject reward/stress to readout."""
    net.reset_simulation(); core.state.current_tick = 0; net.current_tick = 0
    bt = {}
    for (t, c) in ev: bt.setdefault(t, []).append(c)
    acc = np.zeros(len(ro))
    for t in range(T):
        if reward is not None:
            cc = reward
            for k, nu in enumerate(ro):
                if k == cc: nu.M_vector[1] = Mset      # reward correct -> LTP
                else: nu.M_vector[0] = Mset            # stress others -> decay-LTD
        if t in bt:
            for c in bt[t]:
                for (nid, sid, w) in chan.get(c, []): net.set_external_input(nid, sid, gain)
        core.do_tick()
        acc += np.array([nu.S for nu in ro])
    return acc

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True); ap.add_argument("--chan", required=True)
    ap.add_argument("--T", type=int, default=120); ap.add_argument("--gain", type=float, default=8.0)
    ap.add_argument("--epochs", type=int, default=6); ap.add_argument("--ntrain", type=int, default=500)
    ap.add_argument("--Mset", type=float, default=1.1); ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    outp = a.res.replace(".json", "_combined.json")
    ro_ids = build_combined(a.res, outp, seed=0)
    net = NetworkConfig.load_network_config(outp, neuron_class=Neuron)
    core = NNCore(); core.neural_net = net; core.set_log_level("CRITICAL")
    chan = {int(k): v for k, v in json.load(open(a.chan)).items()}
    alln = net.network.neurons
    ro = [alln[i] for i in ro_ids]
    res_neurons = [nu for nid, nu in alln.items() if nid not in set(ro_ids)]
    def acc_on(nsamp, seed):
        trials = make(nsamp, 90, seed); cor = 0
        for ev, lab in trials:
            a_ = run_sample(net, core, chan, res_neurons, ro, ev, a.T, a.gain)
            if int(np.argmax(a_)) == lab: cor += 1
        return cor / nsamp
    print(f"[readout] BEFORE (random readout): acc={acc_on(400,333):.3f} (chance 0.25)", flush=True)
    trials = make(a.ntrain, 90, a.seed)
    for ep in range(a.epochs):
        for ev, lab in trials:
            run_sample(net, core, chan, res_neurons, ro, ev, a.T, a.gain, reward=lab, Mset=a.Mset)
        acc = acc_on(300, 333)
        wm = np.mean([abs(nu.postsynaptic_points[s].u_i.info) for nu in ro for s in range(0, len(res_neurons), 50)])
        print(f"  epoch {ep}: readout acc={acc:.3f}  mean|w_ro|~{wm:.3f}", flush=True)
    print(f"[readout] AFTER (learned PAULA readout): acc={acc_on(400,333):.3f} (chance 0.25)", flush=True)
    print("@@@READOUT DONE@@@", flush=True)
