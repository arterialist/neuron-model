"""Non-destructive config post-processor: turn a plain multi-filter PAULA conv net into a
"designed visual cortex" by (1) overwriting conv-layer weights with an ANALYTIC kernel bank
repeated identically at every spatial position (translation invariance), and (2) optionally
adding lateral inhibition between filters at the same location (competition as topology).

This edits only the saved config JSON (weights in synaptic_points[].u_i.info, plus added
inhibitory connections). It touches NO code path in the builder and NO line of neuron.py — the
same category of config-level transform already used to set plasticity via global_params. That
keeps the shared model holy-grail intact and makes the whole front-end a data-free, inspectable
artifact.

Synapse-id decoding (verified against network_builder_direct.py): for a conv neuron with kernel
size k over in_ch channels, synapse_id = (c·k + ky)·k + kx  (non-separate RGB). So bank[f,c,ky,kx]
maps directly onto the neuron's synapse weights, and EVERY neuron sharing filter index f gets the
SAME kernel -> a real repeated convolutional motif.

Requirements enforced (asserted, not assumed):
  * the target conv layer must have been built with connectivity=1.0 (every k·k·in_ch synapse
    present) so the full kernel is realizable; a missing synapse is a hard error.
  * bank F must equal the layer's distinct filter count.

CLI: build a base net with build_network.py (filters=F, connectivity=1.0), then run this to
stamp the kernels and (optionally) inhibition. See EXPERIMENT_DESIGN_paula_vision.md (§2, M1).
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict

import numpy as np

from snn_classification_realtime.paula_vision import kernels as K


def _decode_synapse(sid: int, k: int, in_ch: int):
    """(c, ky, kx) for synapse id sid under the (c·k+ky)·k+kx layout."""
    per_ch = k * k
    c = sid // per_ch
    rem = sid % per_ch
    return c, rem // k, rem % k


def layer_neurons(cfg: dict, layer_idx: int):
    return [n for n in cfg["neurons"] if n["metadata"].get("layer") == layer_idx]


def stamp_kernels(cfg: dict, layer_idx: int, bank: np.ndarray, scale: float = 1.0,
                  strict: bool = True) -> dict:
    """Overwrite the target conv layer's synapse weights with `bank` (F,in_ch,k,k), repeated at
    every position. Returns a small report. bank rows index filter f; neuron metadata['filter']
    selects the row."""
    neurons = layer_neurons(cfg, layer_idx)
    if not neurons:
        raise ValueError(f"no neurons at layer {layer_idx}")
    F, in_ch, k, _ = bank.shape
    filt_ids = sorted({n["metadata"].get("filter", 0) for n in neurons})
    if strict and len(filt_ids) != F:
        raise ValueError(f"layer {layer_idx} has {len(filt_ids)} filters but bank has {F}")
    # index synaptic points by (neuron_id, synapse_id)
    sp_index = {}
    for sp in cfg["synaptic_points"]:
        if sp.get("type") == "postsynaptic":
            sp_index[(sp["neuron_id"], sp["synapse_id"])] = sp
    nid_by = {n["id"]: n for n in neurons}
    stamped, missing = 0, 0
    per_neuron_syn = defaultdict(int)
    for sp in cfg["synaptic_points"]:
        if sp.get("type") == "postsynaptic" and sp["neuron_id"] in nid_by:
            per_neuron_syn[sp["neuron_id"]] += 1
    for n in neurons:
        f = n["metadata"].get("filter", 0)
        kk = int(n["metadata"].get("kernel_size", k))
        ich = int(n["metadata"].get("in_channels", in_ch))
        if kk != k or ich != in_ch:
            raise ValueError(f"neuron kernel_size/in_ch ({kk},{ich}) != bank ({k},{in_ch})")
        expected = k * k * in_ch
        if strict and per_neuron_syn[n["id"]] != expected:
            raise ValueError(f"neuron {n['id']} has {per_neuron_syn[n['id']]} synapses, "
                             f"expected {expected} (build with connectivity=1.0)")
        for sid in range(expected):
            sp = sp_index.get((n["id"], sid))
            if sp is None:
                missing += 1
                continue
            c, ky, kx = _decode_synapse(sid, k, in_ch)
            sp["u_i"]["info"] = float(scale * bank[f, c, ky, kx])
            stamped += 1
    if strict and missing:
        raise ValueError(f"{missing} kernel synapses missing (build with connectivity=1.0)")
    return dict(layer=layer_idx, filters=len(filt_ids), neurons=len(neurons),
                stamped=stamped, missing=missing, k=k, in_ch=in_ch, scale=scale)


def add_lateral_inhibition(cfg: dict, layer_idx: int, weight: float = -0.5,
                           seed: int = 0) -> dict:
    """Add inhibitory synapses between neurons sharing a spatial (y,x) position but different
    filters, within one layer -> cross-filter competition at each location (kWTA as topology).

    Each target neuron gains one extra synapse per competing filter; sources are the competitors'
    terminals. Weight is negative (uses the model's MIN_SYNAPTIC_WEIGHT=-100 support). This grows
    the config (new synaptic_points + connections + bumps num_synapses/num_inputs); it does not
    touch neuron.py."""
    rng = random.Random(seed)
    neurons = layer_neurons(cfg, layer_idx)
    by_pos = defaultdict(list)
    for n in neurons:
        m = n["metadata"]
        by_pos[(m.get("y"), m.get("x"))].append(n)
    # current synapse count per neuron (to append new ids after the kernel synapses)
    syn_count = defaultdict(int)
    term_by_neuron = defaultdict(list)
    num_nm = None
    for sp in cfg["synaptic_points"]:
        if sp.get("type") == "postsynaptic":
            syn_count[sp["neuron_id"]] += 1
            if num_nm is None and "adapt" in sp.get("u_i", {}):
                num_nm = len(sp["u_i"]["adapt"])
        else:
            term_by_neuron[sp["neuron_id"]].append(sp["terminal_id"])
    num_nm = num_nm or 2
    nmap = {n["id"]: n for n in neurons}
    added_syn, added_conn = 0, 0
    for pos, group in by_pos.items():
        if len(group) < 2:
            continue
        for tgt in group:
            for src in group:
                if src["id"] == tgt["id"]:
                    continue
                new_sid = syn_count[tgt["id"]]
                syn_count[tgt["id"]] = new_sid + 1
                cfg["synaptic_points"].append({
                    "neuron_id": tgt["id"], "synapse_id": new_sid, "type": "postsynaptic",
                    "distance_to_hillock": 3, "potential": 0.0,
                    "u_i": {"info": float(weight), "plast": 1.0,
                            "adapt": [0.2] * num_nm},
                    "metadata": {"lateral_inhibition": True},
                })
                terms = term_by_neuron.get(src["id"]) or [0]
                cfg["connections"].append({
                    "source_neuron": src["id"], "source_terminal": rng.choice(terms),
                    "target_neuron": tgt["id"], "target_synapse": new_sid,
                    "properties": {"lateral_inhibition": True},
                })
                added_syn += 1; added_conn += 1
    # bump each target neuron's declared synapse/input counts so the model allocates them
    for n in neurons:
        p = n["params"]
        if "num_inputs" in p:
            p["num_inputs"] = syn_count[n["id"]]
        if "num_synapses" in p:
            p["num_synapses"] = syn_count[n["id"]]
    return dict(layer=layer_idx, weight=weight, added_synapses=added_syn,
                added_connections=added_conn, positions=len([g for g in by_pos.values() if len(g) > 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-net", required=True, help="base multi-filter conv net JSON")
    ap.add_argument("--out-net", required=True)
    ap.add_argument("--layer", type=int, default=0, help="conv layer to stamp")
    ap.add_argument("--bank", choices=["gabor", "dog"], default="gabor")
    ap.add_argument("--kernel-size", type=int, default=0, help="0 = infer from layer metadata")
    ap.add_argument("--orientations", type=int, default=4)
    ap.add_argument("--scales", default="0.5,0.8")
    ap.add_argument("--phases", default="0,1.5708")
    ap.add_argument("--color", choices=["lum", "opponent"], default="lum")
    ap.add_argument("--scale", type=float, default=1.0, help="weight scaling of stamped kernels")
    ap.add_argument("--lateral-inhibition", type=float, default=0.0,
                    help="magnitude>0 adds cross-filter inhibition of that (negative) weight")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    cfg = json.load(open(a.in_net))
    neurons = layer_neurons(cfg, a.layer)
    if not neurons:
        raise SystemExit(f"no neurons at layer {a.layer}")
    k = a.kernel_size or int(neurons[0]["metadata"].get("kernel_size", 3))
    in_ch = int(neurons[0]["metadata"].get("in_channels", 3))
    if a.bank == "gabor":
        bank, meta = K.gabor_bank(k, in_ch, a.orientations,
                                  tuple(float(x) for x in a.scales.split(",")),
                                  tuple(float(x) for x in a.phases.split(",")), a.color)
    else:
        bank, meta = K.dog_bank(k, in_ch, tuple(float(x) for x in a.scales.split(",")), a.color)
    print(f"bank: {bank.shape} — {K.describe(meta)}")
    rep = stamp_kernels(cfg, a.layer, bank, scale=a.scale)
    print(f"stamped: {rep}")
    if a.lateral_inhibition > 0:
        irep = add_lateral_inhibition(cfg, a.layer, weight=-abs(a.lateral_inhibition), seed=a.seed)
        print(f"inhibition: {irep}")
    cfg.setdefault("metadata", {})["paula_vision"] = dict(
        bank=a.bank, kernel_size=k, orientations=a.orientations, color=a.color,
        n_filters=int(bank.shape[0]), lateral_inhibition=a.lateral_inhibition)
    json.dump(cfg, open(a.out_net, "w"))
    print(f"wrote {a.out_net}")


if __name__ == "__main__":
    main()
