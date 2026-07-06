"""Sparse recurrent PAULA reservoir (Liquid State Machine substrate).

Not an MLP: a recurrent pool of PAULA units whose high-dimensional spiking
dynamics a trained readout decodes. Structure follows the user's steer --
sparsity + hyperconnector hubs, a buffer layer that absorbs the input stream,
low fan-in so weakly/intermittently driven units undergo LTD (calm -> decay)
while consistently driven units saturate and become the class's drivers.

Topology:
  input  (layer 0): receives the retina signal on external synapses; projects on.
  buffer (layer 1): 1:1-ish smoothing of the input stream into the reservoir.
  pool   (layer 2): sparse RECURRENT reservoir; a few HUB neurons carry high
                    in/out degree (rich-club). Readout reads this layer.

Builds the same config dict shape as build_network_config_direct, so
PerceptionNetwork / NetworkConfig can load it unchanged.
"""

from __future__ import annotations

import json
import os
import random

import numpy as np

from snn_classification_realtime.network_builder_direct import (
    _make_synapse_point, _make_terminal_point, _make_neuron_params,
)


def build_reservoir_json(
    path, *, channels, grid, n_in=48, n_buffer=48, n_pool=160,
    pool_fan_in=4, buffer_fan_in=2, n_hub=8, hub_fan_in=24, hub_fan_out_frac=0.25,
    recurrent_frac=0.5, num_terminals=12, seed=0,
):
    """channels = image channels (retina emits 2*channels); grid = retinal grid.
    pool_fan_in kept LOW so the t_ref ceiling (c*fan_in) is low -> LTD reachable
    when a unit calms. Hubs are the exception (high fan-in/out)."""
    random.seed(seed); np.random.seed(seed)
    C2 = 2 * channels
    vec = C2 * grid * grid
    s_in = max(1, int(np.ceil(vec / n_in)))     # external synapses per input neuron

    neurons, syn_pts = [], []
    terms_of, syns_of = {}, {}

    def add_neuron(layer, name, n_syn, n_term):
        nid = random.randint(0, 2**36 - 1)
        neurons.append({"id": nid, "params": _make_neuron_params(n_syn),
                        "metadata": {"layer": layer, "layer_name": name}})
        nm = 2
        for sid in range(n_syn):
            syn_pts.append(_make_synapse_point(nid, sid, random.randint(2, 8), nm))
        for tid in range(n_term):
            syn_pts.append(_make_terminal_point(nid, tid, random.randint(2, 8), 1.0, nm))
        terms_of[nid] = list(range(n_term)); syns_of[nid] = list(range(n_syn))
        return nid

    inp = [add_neuron(0, "input", s_in, num_terminals) for _ in range(n_in)]
    buf = [add_neuron(1, "buffer", buffer_fan_in, num_terminals) for _ in range(n_buffer)]
    # pool: designate hubs (high fan-in), rest low fan-in
    hub_ids = set(range(n_pool - n_hub, n_pool))
    pool = []
    for j in range(n_pool):
        fi = hub_fan_in if j in hub_ids else pool_fan_in
        nt = int(num_terminals * 3) if j in hub_ids else num_terminals
        pool.append(add_neuron(2, "hub" if j in hub_ids else "pool", fi, nt))
    hubs = [pool[j] for j in hub_ids]

    conns = set()
    def wire(src, dst, dst_syn):
        if src == dst:
            return False
        c = (src, random.choice(terms_of[src]), dst, dst_syn)
        if c in conns:
            return False
        conns.add(c); return True

    # input -> buffer (each buffer unit samples a few input units: smoothing)
    for b in buf:
        srcs = random.sample(inp, min(buffer_fan_in, len(inp)))
        for k, s in enumerate(srcs):
            wire(s, b, k)

    # pool wiring: each pool synapse fed by either the buffer (drive) or the pool
    # (recurrence); hubs preferentially sample and are sampled.
    feeders = buf + pool                      # candidate sources for the reservoir
    for pj in pool:
        K = len(syns_of[pj])
        for sid in range(K):
            if random.random() < recurrent_frac and len(pool) > 1:
                # recurrent: bias toward hubs as sources
                src = random.choice(hubs) if (hubs and random.random() < 0.4) else random.choice(pool)
            else:
                src = random.choice(buf)      # feedforward drive from buffer
            tries = 0
            while not wire(src, pj, sid) and tries < 6:
                src = random.choice(feeders); tries += 1

    conn_list = list(conns)
    ext = [{"target_neuron": nid, "target_synapse": sid, "info": 0.0, "mod": [0.0, 0.0]}
           for nid in inp for sid in syns_of[nid]]

    cfg = {
        "metadata": {"name": "PAULA reservoir", "description": "LSM substrate",
                     "version": "1.0", "created_by": "minibrain"},
        "global_params": _make_neuron_params(10),
        "simulation_params": {"max_history": 1000, "default_signal_strength": 1.5,
                              "default_travel_time_range": [5, 25]},
        "neurons": neurons, "synaptic_points": syn_pts,
        "connections": [{"source_neuron": c[0], "source_terminal": c[1],
                         "target_neuron": c[2], "target_synapse": c[3],
                         "properties": {}} for c in conn_list],
        "external_inputs": ext,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg, f)
    info = {"n_in": n_in, "n_buffer": n_buffer, "n_pool": n_pool, "n_hub": n_hub,
            "n_neurons": len(neurons), "n_connections": len(conn_list),
            "s_in": s_in, "pool_fan_in": pool_fan_in, "hub_fan_in": hub_fan_in}
    return path, info
