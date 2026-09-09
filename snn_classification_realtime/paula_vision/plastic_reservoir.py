"""STRUCTURE LEARNING (user-approved fork): does CLASS-ROUTED reward-modulated plasticity build
task-relevant reservoir features the RANDOM reservoir lacks? Each reservoir neuron has a preferred
class; on a class-c sample, preferred-c neurons get reward (M_reward driven -> reward_hebb nm>1),
so their causally-active INPUT synapses potentiate toward class-c input patterns (rh_decay bounded).
Efficient reward: set M_vector[reward] DIRECTLY each tick for preferred neurons (= target/gamma so
after the in-tick EMA decay, plasticity in 5.E sees ~target). Measure separability of the SAME
reservoir BEFORE (random) vs AFTER training -> controls for architecture. Honest test vs prior
basin-generalization-negative (unsupervised plasticity doesn't build class structure)."""
import argparse, json
import numpy as np
from neuron.nn_core import NNCore
from neuron.network_config import NetworkConfig
from neuron.neuron import Neuron
from snn_classification_realtime.paula_vision.temporal_multiclass import make
from snn_classification_realtime.paula_vision.separability_probe import ridge_acc, lda_acc

INCH = [100,200,300,400,500,600,700,120]

def build(out, chanout, N=400, rec_p=0.1, rec_scale=1.2, in_scale=3.0, kappa=1.0,
          eta=0.03, rh_decay=1.0, nclass=4, gamma=0.9, seed=0):
    rng = np.random.RandomState(seed); ids=set()
    while len(ids)<N: ids.update(int(x) for x in rng.randint(0,2**35,N))
    ids=list(ids)[:N]
    neurons=[];sp=[];conns=[];ext=[];chan={c:[] for c in INCH};pref={}
    rec_src=[[s for s in range(N) if s!=t and rng.random()<rec_p] for t in range(N)]
    for idx,nid in enumerate(ids):
        n_in=8; nsyn=n_in+len(rec_src[idx]); pc=int(rng.randint(nclass)); pref[nid]=pc
        params={"num_inputs":nsyn,"num_neuromodulators":2,"eta_post":eta,"eta_retro":0.0,
                "r_base":float(rng.uniform(0.3,0.6)),"b_base":float(rng.uniform(1.0,1.5)),"c":5,
                "lambda_param":20,"p":1.0,"delta_decay":0.9,"beta_avg":0.99,"gamma":[gamma,gamma],
                "w_r":[0,0],"w_b":[0,0],"w_tref":[0,0],"plasticity_mode":"reward_hebb",
                "rh_decay":rh_decay,"nm_plasticity_kappa":kappa,"nm_reward_index":1,"nm_stress_index":0}
        neurons.append({"id":nid,"params":params,"metadata":{"layer":0,"pref":pc}})
        for s,c in enumerate(INCH):
            w=in_scale*rng.randn()
            sp.append({"neuron_id":nid,"synapse_id":s,"type":"postsynaptic","distance_to_hillock":2,"potential":0.0,"u_i":{"info":float(w),"plast":0.0,"adapt":[0.5,0.5]}})
            ext.append({"target_neuron":nid,"target_synapse":s,"info":0.0,"mod":[0.0,0.0]}); chan[c].append((nid,s,float(w)))
        for j,src in enumerate(rec_src[idx]):
            sid=n_in+j
            sp.append({"neuron_id":nid,"synapse_id":sid,"type":"postsynaptic","distance_to_hillock":3,"potential":0.0,"u_i":{"info":float(rec_scale*rng.randn()),"plast":0.0,"adapt":[0.0,0.0]}})
            conns.append({"source_neuron":ids[src],"source_terminal":0,"target_neuron":nid,"target_synapse":sid,"properties":{}})
        sp.append({"neuron_id":nid,"terminal_id":0,"type":"presynaptic","distance_from_hillock":3,"u_o":{"info":1.0,"mod":[0.2,0.2]},"u_i_retro":1.0})
    cfg={"metadata":{"N":N,"plastic":True},"global_params":{"eta_post":eta,"eta_retro":0.0,"num_neuromodulators":2,"num_inputs":8},"simulation_params":{"max_history":500},"neurons":neurons,"synaptic_points":sp,"connections":conns,"external_inputs":ext}
    json.dump(cfg,open(out,"w")); json.dump({str(k):v for k,v in chan.items()},open(chanout,"w"))
    json.dump({str(k):v for k,v in pref.items()},open(out+".pref","w"))
    print(f"built plastic reservoir N={N} kappa={kappa} eta={eta} rh_decay={rh_decay}", flush=True)

def load(net_path, chan_path):
    net=NetworkConfig.load_network_config(net_path,neuron_class=Neuron)
    core=NNCore(); core.neural_net=net; core.set_log_level("CRITICAL")
    chan={int(k):v for k,v in json.load(open(chan_path)).items()}
    pref={int(k):v for k,v in json.load(open(net_path+".pref")).items()}
    neurons=list(net.network.neurons.values())
    return net,core,chan,neurons,pref

def evaluate(net,core,chan,neurons,T,K,neval,seed,gain=12.0):
    trials=make(neval,90,seed); N=len(neurons); X=[];y=[]
    for ev,lab in trials:
        net.reset_simulation(); core.state.current_tick=0; net.current_tick=0
        bt={}
        for (t,c) in ev: bt.setdefault(t,[]).append(c)
        win=np.zeros((K,N));cnt=np.zeros(K)
        for t in range(T):
            if t in bt:
                for c in bt[t]:
                    for (nid,sid,w) in chan.get(c,[]): net.set_external_input(nid,sid,gain)
            core.do_tick(); k=min(K-1,t*K//T); win[k]+=np.array([nu.S for nu in neurons]);cnt[k]+=1
        win/=np.maximum(cnt[:,None],1); X.append(win.ravel()); y.append(lab)
    X=np.array(X);y=np.array(y);n2=len(X)//2
    return ridge_acc(X[:n2],y[:n2],X[n2:],y[n2:]), lda_acc(X[:n2],y[:n2],X[n2:],y[n2:])

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--net",required=True); ap.add_argument("--chan",required=True)
    ap.add_argument("--T",type=int,default=120); ap.add_argument("--Mtarget",type=float,default=1.0)
    ap.add_argument("--epochs",type=int,default=4); ap.add_argument("--ntrain",type=int,default=500)
    ap.add_argument("--gm",type=float,default=0.9); ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--gain",type=float,default=12.0); ap.add_argument("--differential",action="store_true")
    a=ap.parse_args()
    net,core,chan,neurons,pref=load(a.net,a.chan)
    K=6; G=a.gain
    # firing-rate sanity (healthy ~1-4 spikes/neuron/sample)
    net.reset_simulation()
    r0,l0=evaluate(net,core,chan,neurons,a.T,K,800,333,G)
    print(f"[plastic] BEFORE training (random, frozen-eval): ridge={r0:.3f} lda={l0:.3f} (chance 0.25)", flush=True)
    byclass={c:[nu for nu in neurons if pref.get(nu.id)==c] for c in range(4)}
    Mset=a.Mtarget/a.gm
    trials=make(a.ntrain,90,a.seed)
    for ep in range(a.epochs):
        for ev,lab in trials:
            net.reset_simulation(); core.state.current_tick=0; net.current_tick=0
            rew=byclass[lab]; stress=[nu for c in range(4) if c!=lab for nu in byclass[c]]; bt={}
            for (t,c) in ev: bt.setdefault(t,[]).append(c)
            for t in range(a.T):
                for nu in rew: nu.M_vector[1]=Mset          # REWARD own-class -> LTP
                if a.differential:
                    for nu in stress: nu.M_vector[0]=Mset    # STRESS other-class -> nm->0 -> decay-LTD
                if t in bt:
                    for c in bt[t]:
                        for (nid,sid,w) in chan.get(c,[]): net.set_external_input(nid,sid,G)
                core.do_tick()
        wm=np.mean([abs(nu.postsynaptic_points[s].u_i.info) for nu in neurons for s in range(8)])
        r_ep,l_ep=evaluate(net,core,chan,neurons,a.T,K,400,333,G)
        print(f"  epoch {ep}: mean|w_in|={wm:.3f}  sep(ridge/lda)={r_ep:.3f}/{l_ep:.3f}", flush=True)
    for nu in neurons: nu.params.eta_post=0.0
    r1,l1=evaluate(net,core,chan,neurons,a.T,K,800,333,G)
    print(f"[plastic] AFTER training (frozen-eval): ridge={r1:.3f} lda={l1:.3f}", flush=True)
    print(f"[plastic] PLASTICITY DELTA: ridge={r1-r0:+.3f} lda={l1-l0:+.3f} (>0 => structure learning WORKS)", flush=True)
    print("@@@PLASTIC DONE@@@", flush=True)
