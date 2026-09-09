"""Build a recurrent spiking-reservoir PAULA net for SHD (700-channel temporal spike input).
Each reservoir neuron receives: n_in external synapses (each wired to a random input channel with
a random weight = sparse W_in projection) + recurrent synapses from random other reservoir neurons
(random signed = W). Frozen (eta=0) for a clean fixed-reservoir diagnostic; rec_scale tunes the
echo-state memory (too low -> no memory, too high -> chaos/non-reproducible). Returns the net JSON
plus a channel->[(neuron_id, synapse_id, weight)] map the recorder uses to drive input spikes."""
import json, numpy as np


def build(out, chan_map_out, N=800, n_channels=700, n_in=20, rec_p=0.08, rec_scale=1.0,
          in_scale=1.0, seed=0):
    rng = np.random.RandomState(seed)
    # unique random ids WITHOUT permuting the whole 2**35 range (that OOMs); randint + dedupe
    ids = set()
    while len(ids) < N:
        ids.update(int(x) for x in rng.randint(0, 2**35, N - len(ids) + 8))
    ids = list(ids)[:N]
    neurons, sp, conns, ext = [], [], [], []
    chan_map = {c: [] for c in range(n_channels)}   # channel -> [(neuron_id, synapse_id, weight)]
    # recurrent targets
    rec_src = [[s for s in range(N) if s != t and rng.random() < rec_p] for t in range(N)]
    for idx, nid in enumerate(ids):
        in_ch = rng.choice(n_channels, n_in, replace=False)
        in_w = in_scale * rng.randn(n_in)
        n_rec = len(rec_src[idx])
        nsyn = n_in + n_rec
        params = {"num_inputs": nsyn, "num_neuromodulators": 2, "eta_post": 0.0, "eta_retro": 0.0,
                  "r_base": float(rng.uniform(0.9, 1.1)), "b_base": float(rng.uniform(1.0, 1.3)),
                  "c": 5, "lambda_param": 20, "p": 1.0, "delta_decay": 0.9, "beta_avg": 0.99,
                  "gamma": [0.99, 0.995], "w_r": [-0.2, 0.05], "w_b": [-0.2, 0.05],
                  "w_tref": [-20.0, 10.0], "plasticity_mode": "legacy_multiplicative"}
        neurons.append({"id": nid, "params": params,
                        "metadata": {"layer": 0, "layer_name": "reservoir"}})
        for s in range(n_in):
            sp.append({"neuron_id": nid, "synapse_id": s, "type": "postsynaptic",
                       "distance_to_hillock": 2, "potential": 0.0,
                       "u_i": {"info": float(in_w[s]), "plast": 0.0, "adapt": [0.2, 0.2]}})
            ext.append({"target_neuron": nid, "target_synapse": s, "info": 0.0, "mod": [0.0, 0.0]})
            chan_map[int(in_ch[s])].append((nid, s, float(in_w[s])))
        for j, src in enumerate(rec_src[idx]):
            sid = n_in + j
            sp.append({"neuron_id": nid, "synapse_id": sid, "type": "postsynaptic",
                       "distance_to_hillock": 3, "potential": 0.0,
                       "u_i": {"info": float(rec_scale * rng.randn()), "plast": 0.0, "adapt": [0.2, 0.2]}})
            conns.append({"source_neuron": ids[src], "source_terminal": 0,
                          "target_neuron": nid, "target_synapse": sid, "properties": {}})
        sp.append({"neuron_id": nid, "terminal_id": 0, "type": "presynaptic",
                   "distance_from_hillock": 3, "u_o": {"info": 1.0, "mod": [0.2, 0.2]}, "u_i_retro": 1.0})
    cfg = {"metadata": {"reservoir": True, "N": N, "n_channels": n_channels, "n_in": n_in,
                        "rec_p": rec_p, "rec_scale": rec_scale, "in_scale": in_scale},
           "global_params": {"eta_post": 0.0, "eta_retro": 0.0, "num_neuromodulators": 2, "num_inputs": n_in},
           "simulation_params": {"max_history": 1000}, "neurons": neurons, "synaptic_points": sp,
           "connections": conns, "external_inputs": ext}
    json.dump(cfg, open(out, "w"))
    # chan_map keys as str for json
    json.dump({str(k): v for k, v in chan_map.items()}, open(chan_map_out, "w"))
    print(f"built reservoir N={N} n_in={n_in} rec_p={rec_p} rec_scale={rec_scale}: "
          f"{len(neurons)}n {len(conns)}rec {len(ext)}ext", flush=True)


if __name__ == "__main__":
    import sys
    kw = dict(a.split("=") for a in sys.argv[1:] if "=" in a)
    build(out=kw.get("out", "/Users/arterialist/.claude/jobs/4630c9cc/tmp/shd_res.json"),
          chan_map_out=kw.get("chan", "/Users/arterialist/.claude/jobs/4630c9cc/tmp/shd_chanmap.json"),
          N=int(kw.get("N", 800)), n_in=int(kw.get("n_in", 20)),
          rec_p=float(kw.get("rec_p", 0.08)), rec_scale=float(kw.get("rec_scale", 1.0)),
          in_scale=float(kw.get("in_scale", 1.0)), seed=int(kw.get("seed", 0)))
