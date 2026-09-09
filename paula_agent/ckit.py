"""PAULA circuit-construction kit — treat PAULA as a platform. Helpers to declare neurons, dendritic
synapses (with DELAY via distance), terminals (non-colliding IDs), connections, external drives, and
to simulate + probe. All circuits in the agent library build on this."""
import json, numpy as np, tempfile, os
from neuron.nn_core import NNCore
from neuron.network_config import NetworkConfig
from neuron.neuron import Neuron
from neuron.extensions.conjunctive import ConjunctiveGradedNeuron
from neuron.extensions.graded import GradedNeuron
# Diagnostic-only phenomenological subclass.  It is intentionally separate
# from the ordinary PAULA circuit path; see its module docstring.
from neuron.extensions.experimental.phase_locked import PhaseLockedGradedNeuron
TERM=900  # base terminal id (kept >> any synapse id to avoid distances-dict collision)

# !! THESE DEFAULTS OVERRIDE neuron.py's NeuronParameters. Reading neuron.py and quoting ITS defaults
# as "what the agent does" produced six false findings in one session. What ckit changes:
#   delta_decay 0.95 -> 0.999   (dendritic distance is a DELAY here, ~no attenuation: 0.999^90 = 0.91)
#   gamma [0.99,0.995] -> [0.9] (M_vector tau ~10 ticks, NOT a slow integrator)
#   w_r/w_b/w_tref -> ZEROS     (no threshold or learning-window neuromodulation unless set)
#   eta_post -> 0.0             (PLASTICITY OFF; the mode string is a label on a dead path)
#   eta_retro -> 0.0            (retrograde events fire but the update is x0)
# To know what a network ACTUALLY runs, construct it and query u.params -- never cite a default.
def neuron(nid, r=0.6, b=None, c=2, lam=6, nm=2, w_r=None, w_b=None, w_tref=None,
           plasticity="legacy_multiplicative", eta_post=0.0, eta_retro=0.0, rh_decay=0.1,
           kappa=0.0, delta_decay=0.999, w_min=None, w_max=None, nm_internal=None, meta=None):
    b = r+0.6 if b is None else b
    z=[0.0]*nm
    return {"id":nid,"params":{"num_inputs":1,"num_neuromodulators":nm,"eta_post":eta_post,
        "eta_retro":eta_retro,"r_base":r,"b_base":b,"c":c,"lambda_param":lam,"p":1.0,
        "delta_decay":delta_decay,"beta_avg":0.9,"gamma":[0.9]*nm,"w_r":w_r or z[:],"w_b":w_b or z[:],
        "w_tref":w_tref or z[:],"plasticity_mode":plasticity,"rh_decay":rh_decay,
        "nm_plasticity_kappa":kappa,"nm_reward_index":1,"nm_stress_index":0,
        **({"w_min":float(w_min)} if w_min is not None else {}),
        **({"w_max":float(w_max)} if w_max is not None else {}),
        **({"nm_internal":float(nm_internal)} if nm_internal is not None else {})},"metadata":meta or {}}

def syn(nid,sid,w,dist=1,nm=2,adapt=None):
    return {"neuron_id":nid,"synapse_id":sid,"type":"postsynaptic","distance_to_hillock":dist,
            "potential":0.0,"u_i":{"info":float(w),"plast":0.0,"adapt":adapt or [0.3]*nm}}
def term(nid,tid=TERM,nm=2,mod=None):
    # mod defaults to ZERO so neuron->neuron neuromodulation can be ON BY DEFAULT without silently
    # modulating every existing circuit. Set it to drive the TARGET's M_vector, which the model uses to
    # shift the target's threshold and refractory period (neuron.py: r = r_base + w_r.M,
    # t_ref = homeostatic + w_tref.M). That is the substrate's gain-control channel.
    return {"neuron_id":nid,"terminal_id":tid,"type":"presynaptic","distance_from_hillock":1,
            "u_o":{"info":1.0,"mod":list(mod) if mod is not None else [0.0]*nm},"u_i_retro":1.0}
def conn(src,tgt,sid,stid=TERM):
    return {"source_neuron":src,"source_terminal":stid,"target_neuron":tgt,"target_synapse":sid,"properties":{}}
def ext(nid,sid):
    return {"target_neuron":nid,"target_synapse":sid,"info":0.0,"mod":[0.0,0.0]}

def build(neurons,syns,conns,exts,nm=2,fix_num_inputs=True):
    if fix_num_inputs:
        cnt={}
        for s in syns:
            if s["type"]=="postsynaptic": cnt[s["neuron_id"]]=cnt.get(s["neuron_id"],0)+1
        for n in neurons: n["params"]["num_inputs"]=cnt.get(n["id"],1)
    cfg={"metadata":{},"global_params":{"num_neuromodulators":nm,"num_inputs":1},
         "simulation_params":{"max_history":400},"neurons":neurons,"synaptic_points":syns,
         "connections":conns,"external_inputs":exts}
    f=tempfile.NamedTemporaryFile("w",suffix=".json",delete=False,dir="/tmp"); json.dump(cfg,f); f.close()
    return f.name

def load(path, graded=True, neuron_class=None):
    # GradedNeuron is a strict subclass: with no graded_* metadata it behaves EXACTLY as Neuron, so it
    # is safe as the network-wide class and only opted-in cells change. ``neuron_class`` permits
    # the same safe pattern for a strict experimental subclass.
    if neuron_class is None:
        neuron_class=GradedNeuron if graded else Neuron
    net=NetworkConfig.load_network_config(path,neuron_class=neuron_class)
    core=NNCore(); core.neural_net=net; core.set_log_level("CRITICAL")
    return net,core

def simulate(path, T, drives=None, mod=None, probe_ids=None, reset_each=False):
    """drives: dict tick-> list of (nid,sid,amp). mod: fn(t, neurons_by_id)->None to set M_vectors.
    Returns dict nid-> spike train (list of 0/1) for probe_ids (default all)."""
    net,core=load(path); nb={nid:nu for nid,nu in net.network.neurons.items()}
    pids=probe_ids or list(nb.keys()); out={i:[] for i in pids}
    drives=drives or {}
    for t in range(T):
        if mod: mod(t,nb)
        for (nid,sid,amp) in drives.get(t,[]): net.set_external_input(nid,sid,amp)
        core.do_tick()
        for i in pids: out[i].append(int(nb[i].O>0))
    return out

def simulate_S(path, T, drives=None, mod=None, probe_ids=None, hold=None):
    """ANALOG readout: returns dict nid-> list of graded membrane S per tick (not spikes).
    Use with high-threshold neurons that never fire, so S is a continuous computed quantity.
    drives: dict tick-> list of (nid,sid,amp). hold(t)->list of (nid,sid,amp) for sustained drive.
    mod: fn(t, neurons_by_id)->None to set M_vectors."""
    net,core=load(path); nb={nid:nu for nid,nu in net.network.neurons.items()}
    pids=probe_ids or list(nb.keys()); out={i:[] for i in pids}; drives=drives or {}
    for t in range(T):
        if mod: mod(t,nb)
        for (nid,sid,amp) in drives.get(t,[]): net.set_external_input(nid,sid,amp)
        if hold:
            for (nid,sid,amp) in hold(t): net.set_external_input(nid,sid,amp)
        core.do_tick()
        for i in pids: out[i].append(float(nb[i].S))
    return out
