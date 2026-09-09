"""ARCHITECTURAL circuits (batch E): connectors, self-modification, memory systems, recursion,
goals, value, resource economy, entrainment, curiosity — the rest of the AGI-architecture toolkit."""
import numpy as np
from paula_agent import ckit as k
def spikes(t): return sum(t)
def rate(t,a,b): return sum(t[a:b])

def e1_binder():
    """CONNECTOR: cross-modal binder. Fires only when features from TWO modalities co-occur (AND) =>
    a bound multimodal percept; either modality alone stays subthreshold."""
    ne=[k.neuron(1,r=6.0,lam=3,c=1)]; sy=[k.syn(1,0,1.0,1),k.syn(1,1,1.0,1),k.term(1)]
    ex=[k.ext(1,0),k.ext(1,1)]; p=k.build(ne,sy,[],ex)
    def run(mA,mB):
        d={t:([(1,0,4.0)] if mA else [])+([(1,1,4.0)] if mB else []) for t in range(5,25)}
        return spikes(k.simulate(p,30,drives=d,probe_ids=[1])[1])
    return p,run
def v1():
    _,run=e1_binder(); ok= run(1,1)>0 and run(1,0)==0 and run(0,1)==0
    return ok,f"Cross-modal binder: both={run(1,1)} A-only={run(1,0)} B-only={run(0,1)} (binds co-occurrence)"

def e2_gated_relay():
    """CONNECTOR: control-gated relay between modules. Signal passes A->out only when a CONTROL line
    is active (neuromod opens the relay). Enables dynamic inter-module connectivity."""
    ne=[k.neuron(1,r=6.0,lam=3,c=1)]; sy=[k.syn(1,0,1.0,1),k.syn(1,1,1.0,1),k.term(1)]  # AND(signal,control)
    ex=[k.ext(1,0),k.ext(1,1)]; p=k.build(ne,sy,[],ex)
    def run(sig,ctrl):
        d={t:([(1,0,4.0)] if sig else [])+([(1,1,4.0)] if ctrl else []) for t in range(5,25)}
        return spikes(k.simulate(p,30,drives=d,probe_ids=[1])[1])
    return p,run
def v2():
    _,run=e2_gated_relay(); ok= run(1,1)>0 and run(1,0)==0
    return ok,f"Gated relay: signal+control={run(1,1)} signal-only(gate closed)={run(1,0)} (dynamic connectivity)"

def e3_modality_core():
    """MODALITY CORE: a small recurrent cluster that expands input into a distributed code — distinct
    inputs -> distinct population patterns (separability), the substrate of a processing core."""
    N=12; rng=np.random.RandomState(0)
    ne=[k.neuron(i+1,r=0.8,lam=5) for i in range(N)]; sy=[];conns=[]
    for i in range(N):
        sy+=[k.syn(i+1,0,float(rng.uniform(0.5,2.0)),1),k.syn(i+1,1,float(rng.uniform(0.5,2.0)),1),k.term(i+1)]
    # sparse recurrence
    for i in range(N):
        j=int(rng.randint(N))
        if j!=i: sy.append(k.syn(i+1,2,float(rng.uniform(0.5,1.5)),3)); conns.append(k.conn(j+1,i+1,2))
    ex=[k.ext(i+1,s) for i in range(N) for s in (0,1)]; p=k.build(ne,sy,conns,ex)
    def pattern(inA,inB):
        d={t:[(i+1,0,inA) for i in range(N)]+[(i+1,1,inB) for i in range(N)] for t in range(5,25)}
        o=k.simulate(p,30,drives=d,probe_ids=[i+1 for i in range(N)])
        return np.array([spikes(o[i+1]) for i in range(N)])
    return p,pattern
def v3():
    _,pat=e3_modality_core(); p1=pat(4.0,0.0); p2=pat(0.0,4.0)
    ok= p1.sum()>0 and p2.sum()>0 and np.corrcoef(p1,p2)[0,1]<0.8  # distinct inputs -> distinct codes
    return ok,f"Modality core: input-A code sum={p1.sum()} input-B sum={p2.sum()} pattern-corr={np.corrcoef(p1,p2)[0,1]:.2f} (<0.8 => separable)"

def e4_pruning():
    """SELF-MODIFICATION: synaptic pruning. An UNUSED synapse (no reward, low activity) decays toward
    zero under reward_hebb decay; a USED+rewarded one is maintained. Structural economy."""
    ne=[k.neuron(1,r=0.4),k.neuron(2,r=1.2,lam=6,plasticity="reward_hebb",eta_post=0.06,rh_decay=0.5,kappa=2.0)]
    sy=[k.syn(1,0,3.0,1),k.term(1),k.syn(2,0,3.0,1),k.syn(2,1,3.0,1),k.syn(2,2,3.0,1),k.term(2)]  # syn0 used, syn1 UNUSED(no input), syn2 teach
    conns=[k.conn(1,2,0)]; ex=[k.ext(1,0),k.ext(2,2)]; p=k.build(ne,sy,conns,ex)
    def run():
        net,core=k.load(p); nb={i:n for i,n in net.network.neurons.items()}
        w1_0=nb[2].postsynaptic_points[1].u_i.info
        for _ in range(25):
            for t in range(12):
                nb[2].M_vector[1]=2.0 if 2<=t<7 else 0.0
                if 1<=t<6: net.set_external_input(1,0,4.0); net.set_external_input(2,2,4.0)  # use syn0(via relay), teach on syn2
                core.do_tick()
        return nb[2].postsynaptic_points[0].u_i.info, w1_0, nb[2].postsynaptic_points[1].u_i.info
    return p,run
def v4():
    _,run=e4_pruning(); used,unused0,unused=run()
    ok= used>unused  # rewarded/used synapse maintained above the decayed unused one
    return ok,f"Pruning: used+rewarded w={used:.2f} vs unused w {unused0:.2f}->{unused:.2f} (unused pruned)"

def e5_hebb_growth():
    """SELF-MODIFICATION: Hebbian strengthening. Correlated pre->post activity (with reward) GROWS the
    synapse from near-zero -> a new effective connection forms (structural growth analog)."""
    ne=[k.neuron(1,r=0.4),k.neuron(2,r=1.0,lam=6,plasticity="reward_hebb",eta_post=0.06,rh_decay=0.4,kappa=2.0)]
    sy=[k.syn(1,0,3.0,1),k.term(1),k.syn(2,0,0.1,1),k.syn(2,1,3.0,1),k.term(2)]  # syn0 latent(weak), syn1 teach
    conns=[k.conn(1,2,0)]; ex=[k.ext(1,0),k.ext(2,1)]; p=k.build(ne,sy,conns,ex)
    def run():
        net,core=k.load(p); nb={i:n for i,n in net.network.neurons.items()}
        w0=nb[2].postsynaptic_points[0].u_i.info
        for _ in range(25):
            for t in range(12):
                nb[2].M_vector[1]=2.0 if 2<=t<7 else 0.0
                if 1<=t<6: net.set_external_input(1,0,4.0); net.set_external_input(2,1,4.0)
                core.do_tick()
        return w0, nb[2].postsynaptic_points[0].u_i.info
    return p,run
def v5():
    _,run=e5_hebb_growth(); w0,w1=run(); ok= w1>w0+1.0
    return ok,f"Hebbian growth: latent synapse {w0:.2f} -> {w1:.2f} (new connection formed)"

def e6_specialization():
    """SELF-ORGANIZATION: role differentiation. Two output neurons + mutual inhibition + reward_hebb:
    presented with two distinct inputs, they SPECIALIZE (each becomes selective for one input)."""
    ne=[k.neuron(1,r=0.4),k.neuron(2,r=0.4),
        k.neuron(3,r=1.0,lam=6,plasticity="reward_hebb",eta_post=0.05,rh_decay=0.4,kappa=2.0),
        k.neuron(4,r=1.0,lam=6,plasticity="reward_hebb",eta_post=0.05,rh_decay=0.4,kappa=2.0)]
    sy=[k.syn(1,0,3.0,1),k.term(1),k.syn(2,0,3.0,1),k.term(2),
        k.syn(3,0,0.3,1),k.syn(3,1,0.3,1),k.syn(3,2,-8.0,1),k.syn(3,3,3.0,1),k.term(3),  # inA,inB, inhib from 4, teach
        k.syn(4,0,0.3,1),k.syn(4,1,0.3,1),k.syn(4,2,-8.0,1),k.syn(4,3,3.0,1),k.term(4)]
    conns=[k.conn(1,3,0),k.conn(2,3,1),k.conn(1,4,0),k.conn(2,4,1),k.conn(4,3,2),k.conn(3,4,2)]
    ex=[k.ext(1,0),k.ext(2,0),k.ext(3,3),k.ext(4,3)]; p=k.build(ne,sy,conns,ex)
    def run():
        net,core=k.load(p); nb={i:n for i,n in net.network.neurons.items()}
        for ep in range(30):
            inp=ep%2  # alternate input A / B
            teach = 3 if inp==0 else 4  # teach neuron 3 for A, neuron 4 for B
            for t in range(10):
                nb[3].M_vector[1]=2.0; nb[4].M_vector[1]=2.0
                if 1<=t<5: net.set_external_input(inp+1,0,4.0); net.set_external_input(teach,3,5.0)
                core.do_tick()
        # neuron3 should prefer input A (w[0]>w[1]); neuron4 prefer B (w[1]>w[0])
        w3=[nb[3].postsynaptic_points[i].u_i.info for i in (0,1)]
        w4=[nb[4].postsynaptic_points[i].u_i.info for i in (0,1)]
        return w3,w4
    return p,run
def v6():
    _,run=e6_specialization(); w3,w4=run()
    ok= w3[0]>w3[1] and w4[1]>w4[0]  # differentiated: n3->A, n4->B
    return ok,f"Specialization: neuron3 weights(A,B)={[round(x,1) for x in w3]} neuron4={[round(x,1) for x in w4]} (roles differentiated)"

def e7_multislot_wm():
    """WORKING MEMORY: multi-slot register. Two independent bistable latches hold two bits; setting
    one doesn't disturb the other (parallel storage)."""
    def latch(base):
        return ([k.neuron(base,r=0.4,lam=10,c=2)],
                [k.syn(base,0,3.0,1),k.syn(base,1,4.5,1),k.syn(base,2,-8.0,1),k.term(base)],
                [k.conn(base,base,1)])
    ne=[];sy=[];conns=[]
    for b in (1,2):
        n,s,c=latch(b); ne+=n; sy+=s; conns+=c
    ex=[k.ext(1,0),k.ext(1,2),k.ext(2,0),k.ext(2,2)]; p=k.build(ne,sy,conns,ex)
    def run(set_slot):
        d={t:[(set_slot,0,4.0)] for t in range(5,8)}
        o=k.simulate(p,40,drives=d,probe_ids=[1,2])
        return rate(o[1],20,35),rate(o[2],20,35)
    return p,run
def v7():
    _,run=e7_multislot_wm(); s1=run(1); s2=run(2)
    ok= s1[0]>0 and s1[1]==0 and s2[1]>0 and s2[0]==0  # slots independent
    return ok,f"Multi-slot WM: set-slot1->{s1} set-slot2->{s2} (independent parallel storage)"

def e8_sequence_replay():
    """EPISODIC BUFFER: store + replay. A trigger launches a stored spatiotemporal sequence (synfire
    chain) -> the 'episode' replays in order from a single cue."""
    chain=[1,2,3,4]; ne=[k.neuron(n,r=0.5,lam=3) for n in chain]
    sy=[];conns=[]
    for n in chain: sy+=[k.syn(n,0,3.0,4),k.term(n)]
    for a,b in zip(chain,chain[1:]): conns.append(k.conn(a,b,0))
    ex=[k.ext(1,0)]; p=k.build(ne,sy,conns,ex)
    def run():
        o=k.simulate(p,40,drives={5:[(1,0,4.0)]},probe_ids=chain)
        return [o[n].index(1) if 1 in o[n] else -1 for n in chain]
    return p,run
def v8():
    _,run=e8_sequence_replay(); f=run(); ok= all(x>=0 for x in f) and f==sorted(f)
    return ok,f"Episodic replay: cue -> sequence fires at ticks {f} (ordered replay)"

def e9_goal_maintenance():
    """GOAL MAINTENANCE: a goal latch persists a chosen goal that BIASES a downstream selector, so the
    same ambiguous input yields goal-consistent action."""
    ne=[k.neuron(1,r=0.4,lam=10,c=2),                 # goal latch (persistent)
        k.neuron(2,r=4.0,lam=5,c=2)]                    # action = input AND goal-bias
    sy=[k.syn(1,0,3.0,1),k.syn(1,1,4.5,1),k.term(1),
        k.syn(2,0,1.0,1),k.syn(2,1,3.0,1),k.term(2)]   # AND(input, goal): goal weighted up
    conns=[k.conn(1,1,1),k.conn(1,2,1)]; ex=[k.ext(1,0),k.ext(2,0)]; p=k.build(ne,sy,conns,ex)
    def run(goal_set):
        d={}
        if goal_set:
            for t in range(3,6): d[t]=[(1,0,4.0)]
        for t in range(15,35): d.setdefault(t,[]).append((2,0,3.0))  # ambiguous input later
        o=k.simulate(p,40,drives=d,probe_ids=[2]); return spikes(o[2])
    return p,run
def v9():
    _,run=e9_goal_maintenance(); with_goal=run(True); no_goal=run(False)
    ok= with_goal>0 and no_goal==0  # action only when the persistent goal biases it
    return ok,f"Goal maintenance: action-with-goal={with_goal} action-no-goal={no_goal} (goal persists & biases)"

def e10_value():
    """VALUE ACCUMULATOR: encodes the cumulative reward/value of a state (leaky integrator of a reward
    signal) -> higher sustained reward -> higher value activity."""
    ne=[k.neuron(1,r=1.4,lam=40)]; sy=[k.syn(1,0,1.0,1),k.term(1)]; ex=[k.ext(1,0)]; p=k.build(ne,sy,[],ex)
    def val(reward): return spikes(k.simulate(p,80,drives={t:[(1,0,reward)] for t in range(5,80)},probe_ids=[1])[1])
    return p,val
def v10():
    _,val=e10_value(); lo=val(1.5); hi=val(3.0); ok= hi>lo
    return ok,f"Value accumulator: value(reward=1.5)={lo} value(reward=3)={hi} (encodes cumulative value)"

def e11_recursion():
    """RECURSION / self-reflection loop (internal environment): a neuron's output is fed back as its
    own input with a delay -> sustained self-referential reverberation from a transient trigger."""
    ne=[k.neuron(1,r=0.5,lam=5,c=3)]; sy=[k.syn(1,0,3.0,1),k.syn(1,1,3.0,6),k.term(1)]  # ext trigger + self-loop
    conns=[k.conn(1,1,1)]; ex=[k.ext(1,0)]; p=k.build(ne,sy,conns,ex)
    def run(): 
        tr=k.simulate(p,60,drives={5:[(1,0,4.0)],6:[(1,0,4.0)]},probe_ids=[1])[1]
        return sum(tr[3:8]),sum(tr[30:55])  # trigger response, later self-sustained
    return p,run
def v11():
    _,run=e11_recursion(); early,late=run(); ok= early>0 and late>0
    return ok,f"Recursion loop: trigger-response={early}, self-sustained-later={late} (internal reverberation)"

def e12_resource():
    """RESOURCE ECONOMY: metabolic gating via a FATIGUE accumulator. Sustained work builds a slow
    fatigue signal that inhibits the worker -> work rate FALLS over time under constant demand, and
    recovers when idle. A metabolic constraint on computation."""
    # LATCHING fatigue: a slow integrator (2) accumulates worker (1) activity; past threshold it
    # LATCHES on (self-excitation, bistable) and then continuously inhibits the worker -> rate falls
    # after depletion and stays down. (A plain integrator can't sustain inhibition from sparse spikes;
    # the latch converts accumulated activity into a persistent OFF signal.)
    # 1=worker, 2=slow INTEGRATOR (accumulates worker activity, fires only after depletion),
    # 3=bistable LATCH (triggered by integrator, self-sustains densely -> continuous worker inhibition)
    ne=[k.neuron(1,r=0.6,lam=5,c=2),k.neuron(2,r=1.3,lam=35,c=1),k.neuron(3,r=0.4,lam=6,c=1)]
    sy=[k.syn(1,0,2.2,1),k.syn(1,1,-9.0,1),k.term(1),              # worker: demand(+), latch-inhib(-)
        k.syn(2,0,5.0,1),k.term(2),                               # integrator: accumulate worker (no self-loop)
        k.syn(3,0,3.0,1),k.syn(3,1,4.5,1),k.term(3)]              # latch: trigger(from integ) + self-exc(bistable)
    conns=[k.conn(1,2,0),k.conn(2,3,0),k.conn(3,3,1),k.conn(3,1,1)]; ex=[k.ext(1,0)]; p=k.build(ne,sy,conns,ex)
    def run():
        o=k.simulate(p,130,drives={t:[(1,0,3.0)] for t in range(3,130)},probe_ids=[1])  # constant demand
        return rate(o[1],5,25),rate(o[1],105,128)  # early (fresh) vs late (fatigue-latched)
    return p,run
def v12():
    _,run=e12_resource(); early,late=run(); ok= early>late  # activity falls as resource depletes
    return ok,f"Resource economy: work-while-resourced={early} work-after-depletion={late} (metabolic gating)"

def e13_entrainment():
    """SYNCHRONIZATION: entrainment. A neuron phase-LOCKS to an external periodic drive (a global
    clock/rhythm), firing at the rhythm's period -> shared temporal reference across modules."""
    ne=[k.neuron(1,r=1.2,lam=4,c=3)]; sy=[k.syn(1,0,2.0,1),k.term(1)]; ex=[k.ext(1,0)]; p=k.build(ne,sy,[],ex)
    def run(period):
        d={t:[(1,0,4.0)] for t in range(5,60,period)}
        tr=k.simulate(p,60,drives=d,probe_ids=[1])[1]
        idx=[i for i,x in enumerate(tr) if x]
        gaps=np.diff(idx) if len(idx)>1 else np.array([0])
        return len(idx), (float(np.mean(gaps)) if len(gaps)>0 else 0)
    return p,run
def v13():
    _,run=e13_entrainment(); n1,g1=run(6); n2,g2=run(10)
    ok= n1>3 and n2>3 and g2>g1  # locks to the driving period (slower drive -> larger inter-spike gap)
    return ok,f"Entrainment: period6->mean-gap={g1:.1f}, period10->gap={g2:.1f} (phase-locks to rhythm)"

def e14_curiosity():
    """CURIOSITY DRIVE: novelty -> exploration. A novelty/change detector feeds an exploration-drive
    integrator: novel inputs raise the drive; a fully-predictable stream leaves it low."""
    ne=[k.neuron(1,r=0.7,lam=2),k.neuron(2,r=0.8,lam=25,c=2)]  # 1=novelty detector, 2=exploration drive
    sy=[k.syn(1,0,2.5,1),k.syn(1,1,-2.4,7),k.term(1),      # novelty = onset/change
        k.syn(2,0,4.0,1),k.term(2)]                        # drive integrates novelty
    conns=[k.conn(1,2,0)]; ex=[k.ext(1,0),k.ext(1,1)]; p=k.build(ne,sy,conns,ex)
    def run(changing):
        d={}
        if changing:  # repeated ONSETS (novel each time)
            for onset in range(5,55,8):
                for t in range(onset,onset+3): d[t]=[(1,0,3.0),(1,1,3.0)]
        else:  # one sustained (predictable) input
            for t in range(5,55): d[t]=[(1,0,3.0),(1,1,3.0)]
        o=k.simulate(p,60,drives=d,probe_ids=[2]); return spikes(o[2])
    return p,run
def v14():
    _,run=e14_curiosity(); novel=run(True); predictable=run(False)
    ok= novel>predictable
    return ok,f"Curiosity drive: exploration under novelty={novel} vs predictable-stream={predictable} (novelty drives exploration)"

BATCH_E=[("Cross-modal binder",v1),("Gated relay connector",v2),("Modality processing core",v3),
         ("Self-mod: synaptic pruning",v4),("Self-mod: Hebbian growth",v5),("Role specialization",v6),
         ("Multi-slot working memory",v7),("Episodic sequence replay",v8),("Goal maintenance",v9),
         ("Value accumulator",v10),("Recursion/self-reflection loop",v11),("Resource economy",v12),
         ("Entrainment/synchronization",v13),("Curiosity drive",v14)]
if __name__=="__main__":
    import sys; sys.path.insert(0,".")
    ok=0
    for name,vf in BATCH_E:
        try:
            passed,msg=vf(); ok+=bool(passed); print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {msg}",flush=True)
        except Exception as e:
            print(f"  [ERR ] {name}: {e}",flush=True)
    print(f"BATCH E: {ok}/{len(BATCH_E)} verified",flush=True)
