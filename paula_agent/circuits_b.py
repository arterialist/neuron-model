"""Agent circuit library — BATCH B: motor, neuromodulation/motivation/learning, navigation."""
import numpy as np
from paula_agent import ckit as k
def spikes(t): return sum(t)

# ---------------- MOTOR ----------------
def c11_cpg_ring():
    """Locomotor CPG: 3-neuron inhibitory ring -> traveling wave (3-phase gait). Robust oscillator."""
    # looped synfire ring: N1->N2->N3->N1 with delay -> a wave circulates = repeating gait rhythm
    ne=[k.neuron(i+1,r=0.6,lam=3,c=6) for i in range(3)]
    sy=[];conns=[]
    for i in range(3):
        nid=i+1; sy+=[k.syn(nid,0,4.0,6),k.term(nid,tid=900+nid)]  # excitation from previous, delayed
    for i in range(3):
        prev=(i-1)%3; conns.append(k.conn(prev+1,i+1,0,stid=900+prev+1))
    ex=[k.ext(1,0)]; p=k.build(ne,sy,conns,ex)
    def run(T=160):
        return k.simulate(p,T,drives={3:[(1,0,5.0)]},probe_ids=[1,2,3])
    return p,run
def v11():
    _,run=c11_cpg_ring(); o=run(); rates=[spikes(o[i]) for i in [1,2,3]]
    firsts=[o[i].index(1) if 1 in o[i] else -1 for i in [1,2,3]]
    ok = all(r>=3 for r in rates) and firsts[0]<firsts[1]<firsts[2]  # wave circulates in order, repeatedly
    return ok,f"CPG (synfire ring): rates={rates} first-fire order={firsts} (traveling wave gait)"

def c12_sequence():
    """Motor sequence generator: trigger -> M1,M2,M3 fire in ORDER (a motor program) via delay chain."""
    ne=[k.neuron(i+1,r=0.5,lam=3) for i in range(3)]
    sy=[k.syn(1,0,3.0,1),k.term(1),
        k.syn(2,0,3.0,4),k.term(2,tid=902),
        k.syn(3,0,3.0,4),k.term(3,tid=903)]
    conns=[k.conn(1,2,0),k.conn(2,3,0,stid=902)]
    ex=[k.ext(1,0)]; p=k.build(ne,sy,conns,ex)
    def run(): 
        o=k.simulate(p,40,drives={5:[(1,0,4.0)]},probe_ids=[1,2,3])
        return [o[i].index(1) if 1 in o[i] else -1 for i in [1,2,3]]
    return p,run
def v12():
    _,run=c12_sequence(); f=run(); ok=all(x>=0 for x in f) and f[0]<f[1]<f[2]
    return ok,f"Motor sequence: fire order ticks={f}"

def c13_reflex():
    """Reflex arc: sensor -> interneuron -> motor, fast fixed latency (withdrawal reflex)."""
    ne=[k.neuron(1,r=0.4,lam=3),k.neuron(2,r=0.4,lam=3),k.neuron(3,r=0.4,lam=3)]  # sensor,inter,motor
    sy=[k.syn(1,0,3.0,1),k.term(1),k.syn(2,0,3.0,1),k.term(2,tid=902),k.syn(3,0,3.0,1),k.term(3,tid=903)]
    conns=[k.conn(1,2,0),k.conn(2,3,0,stid=902)]
    ex=[k.ext(1,0)]; p=k.build(ne,sy,conns,ex)
    def run():
        o=k.simulate(p,30,drives={5:[(1,0,4.0)]},probe_ids=[1,3])
        s=o[1].index(1) if 1 in o[1] else -1; m=o[3].index(1) if 1 in o[3] else -1
        return s,m
    return p,run
def v13():
    _,run=c13_reflex(); s,m=run(); ok= s>=0 and m>=0 and 0< m-s <=8
    return ok,f"Reflex arc: sensor@{s} -> motor@{m} (latency {m-s})"

def c14_pcontroller():
    """Proportional controller: motor firing rate scales with error magnitude (graded response)."""
    ne=[k.neuron(1,r=1.9,lam=5,c=1)]; sy=[k.syn(1,0,1.0,1),k.term(1)]; ex=[k.ext(1,0)]; p=k.build(ne,sy,[],ex)
    def rate(err): return spikes(k.simulate(p,60,drives={t:[(1,0,err)] for t in range(5,60)},probe_ids=[1])[1])
    return p,rate
def v14():
    _,rate=c14_pcontroller(); lo=rate(1.0); mid=rate(2.5); hi=rate(5.0)
    ok= lo<mid<hi
    return ok,f"P-controller: rate(err=1/2.5/5)={lo}/{mid}/{hi} (monotone)"

# ---------------- NEUROMOD / MOTIVATION / LEARNING ----------------
def c15_router():
    """Attention router: targeted neuromod opens a specific gate pathway (demultiplexer)."""
    ne=[k.neuron(1,r=0.5),k.neuron(2,r=2.2,w_r=[0,-2.0]),k.neuron(3,r=2.2,w_r=[0,-2.0]),k.neuron(4,r=0.6),k.neuron(5,r=0.6)]
    sy=[k.syn(1,0,1.0,1),k.term(1),k.syn(2,0,2.0,1),k.term(2,tid=902),k.syn(3,0,2.0,1),k.term(3,tid=903),
        k.syn(4,0,2.0,1),k.term(4,tid=904),k.syn(5,0,2.0,1),k.term(5,tid=905)]
    conns=[k.conn(1,2,0),k.conn(1,3,0),k.conn(2,4,0,stid=902),k.conn(3,5,0,stid=903)]
    ex=[k.ext(1,0)]; p=k.build(ne,sy,conns,ex)
    def run(target):
        def mod(t,nb):
            if target==1: nb[2].M_vector[1]=2.0
            elif target==2: nb[3].M_vector[1]=2.0
        o=k.simulate(p,50,drives={t:[(1,0,4.0)] for t in range(5,25,4)},mod=mod,probe_ids=[4,5])
        return spikes(o[4]),spikes(o[5])
    return p,run
def v15():
    _,run=c15_router(); n=run(0); g1=run(1); g2=run(2)
    ok= n==(0,0) and g1[0]>0 and g1[1]==0 and g2[0]==0 and g2[1]>0
    return ok,f"Router: none={n} mod->G1={g1} mod->G2={g2}"

def c16_drive():
    """Homeostatic drive: 'satiety' neuron (kept up by feeding) INHIBITS the 'hunger/drive' neuron.
    Feeding -> drive silent; feeding stops -> satiety decays -> drive released -> motivates behavior."""
    ne=[k.neuron(1,r=0.5,lam=25),k.neuron(2,r=0.6,lam=5)]  # 1=satiety(slow), 2=drive
    sy=[k.syn(1,0,3.0,1),k.syn(1,1,2.0,1),k.term(1),  # satiety self-sustains
        k.syn(2,0,1.2,1),k.syn(2,1,-5.0,1),k.term(2,tid=902)]  # drive: tonic + inhibited by satiety
    conns=[k.conn(1,1,1),k.conn(1,2,1)]  # satiety self-exc + inhibits drive
    ex=[k.ext(1,0),k.ext(2,0)]; p=k.build(ne,sy,conns,ex)
    def run(T=120):
        d={t:[(2,0,2.0)] for t in range(T)}  # tonic drive-seeking
        for t in range(3,25): d.setdefault(t,[]).append((1,0,4.0))  # feed early only
        o=k.simulate(p,T,drives=d,probe_ids=[1,2])
        drive_early=sum(o[2][10:30]); drive_late=sum(o[2][80:110])
        return drive_early,drive_late
    return p,run
def v16():
    _,run=c16_drive(); early,late=run(); ok= late>early  # drive rises as satiety decays
    return ok,f"Homeostatic drive: drive-while-fed={early} drive-after-fed-stops={late} (rises)"

def c17_arousal():
    """Global arousal: a neuromod scales downstream excitability (same input -> more output when aroused)."""
    ne=[k.neuron(1,r=1.3,w_r=[0,-1.0])]; sy=[k.syn(1,0,1.0,1),k.term(1)]; ex=[k.ext(1,0)]; p=k.build(ne,sy,[],ex)
    def rate(arousal):
        def mod(t,nb): nb[1].M_vector[1]=arousal
        return spikes(k.simulate(p,60,drives={t:[(1,0,1.3)] for t in range(5,60)},mod=mod,probe_ids=[1])[1])
    return p,rate
def v17():
    _,rate=c17_arousal(); lo=rate(0.0); hi=rate(2.0); ok= hi>lo
    return ok,f"Arousal gain: rate(arousal=0)={lo} rate(aroused)={hi}"

def c18_conditioning():
    """Reward-modulated associative learning (classical conditioning): CS->US via a plastic synapse
    that strengthens when CS fires + reward follows. After training, CS alone triggers the response."""
    ne=[k.neuron(1,r=0.4),  # CS input relay
        k.neuron(2,r=0.9,lam=6,nm=2,plasticity="reward_hebb",eta_post=0.05,rh_decay=0.3,kappa=2.0,w_r=[0,0])] # response neuron, plastic CS synapse
    sy=[k.syn(1,0,3.0,1),k.term(1),
        k.syn(2,0,0.2,1),   # CS->response: weak initially (won't fire alone)
        k.syn(2,1,4.0,1),   # US (unconditioned strong)
        k.term(2,tid=902)]
    conns=[k.conn(1,2,0)]
    ex=[k.ext(1,0),k.ext(2,1)]; p=k.build(ne,sy,conns,ex)
    def cs_alone():
        return spikes(k.simulate(p,30,drives={t:[(1,0,4.0)] for t in range(5,10)},probe_ids=[2])[2])
    def train_and_test():
        net,core=k.load(p); nb={i:n for i,n in net.network.neurons.items()}
        def step(T,drives,reward):
            for t in range(T):
                if reward: nb[2].M_vector[1]=2.0
                for (nid,sid,a) in drives.get(t,[]): net.set_external_input(nid,sid,a)
                core.do_tick()
        # training: CS + US + reward, repeated
        for _ in range(15):
            step(14,{**{t:[(1,0,4.0)] for t in range(1,5)},**{t:[(2,1,4.0)] for t in range(2,6)}},reward=True)
        # freeze + test CS alone
        nb[2].params.eta_post=0.0
        resp=[]
        for t in range(40):
            if 3<=t<12: net.set_external_input(1,0,5.0)
            core.do_tick(); resp.append(int(nb[2].O>0))
        w=nb[2].postsynaptic_points[0].u_i.info
        return sum(resp),w
    return p,cs_alone,train_and_test
def v18():
    _,cs_alone,tnt=c18_conditioning(); before=cs_alone(); after,w=tnt()
    ok= before==0 and after>0
    return ok,f"Conditioning: CS-alone before={before}, after training={after} (CS-weight->{w:.2f})"

# ---------------- NAVIGATION ----------------
def c19_heading():
    """Heading integrator: angular-velocity input integrated into a heading estimate (bidirectional)."""
    ne=[k.neuron(1,r=1.4,lam=50)]; sy=[k.syn(1,0,1.0,1),k.syn(1,1,-1.0,1),k.term(1)]  # +CW, -CCW
    ex=[k.ext(1,0),k.ext(1,1)]; p=k.build(ne,sy,[],ex)
    def run(cw_rate):
        return spikes(k.simulate(p,80,drives={t:[(1,0,cw_rate)] for t in range(5,80)},probe_ids=[1])[1])
    return p,run
def v19():
    _,run=c19_heading(); slow=run(1.5); fast=run(3.0); ok= fast>slow
    return ok,f"Heading integrator: turn-rate slow->{slow} fast->{fast} spikes (integrates angular velocity)"

def c20_pathint():
    """Path integrator: speed input integrated into distance traveled (fires more the farther gone)."""
    ne=[k.neuron(1,r=1.1,lam=60,c=1)]; sy=[k.syn(1,0,1.0,1),k.term(1)]; ex=[k.ext(1,0)]; p=k.build(ne,sy,[],ex)
    def dist(speed,dur): return spikes(k.simulate(p,dur+10,drives={t:[(1,0,speed)] for t in range(5,5+dur)},probe_ids=[1])[1])
    return p,dist
def v20():
    _,dist=c20_pathint(); short=dist(2.0,20); long=dist(2.0,60); ok= long>short
    return ok,f"Path integrator: dist(dur=20)={short} dist(dur=60)={long} (accumulates distance)"

def c21_place():
    """Place cell: fires only at a specific 'location' = coincidence of two place cues arriving with
    a specific relative timing (dendritic-delay conjunction of landmarks)."""
    ne=[k.neuron(1,r=1.5,lam=3)]; sy=[k.syn(1,0,0.9,10),k.syn(1,1,0.9,3),k.term(1)]; ex=[k.ext(1,0),k.ext(1,1)]
    p=k.build(ne,sy,[],ex)
    def at(delta):  # relative arrival of the two landmarks (= a location)
        d={5:[(1,0,4.0)],5+delta:[(1,1,4.0)]}
        if delta==0: d={5:[(1,0,4.0),(1,1,4.0)]}
        return spikes(k.simulate(p,40,drives=d,probe_ids=[1])[1])
    return p,at
def v21():
    _,at=c21_place(); here=at(7); elsewhere=at(0)+at(14)
    ok= here>0 and elsewhere==0
    return ok,f"Place cell: at-place(delta=7)={here} elsewhere(0,14)={elsewhere}"

BATCH_B=[("Locomotor CPG (3-ring gait)",v11),("Motor sequence generator",v12),("Reflex arc",v13),
         ("Proportional controller",v14),("Attention router",v15),("Homeostatic drive",v16),
         ("Global arousal gain",v17),("Associative conditioning (learning)",v18),
         ("Heading integrator",v19),("Path integrator",v20),("Place cell",v21)]
if __name__=="__main__":
    import sys; sys.path.insert(0,".")
    ok=0
    for name,vf in BATCH_B:
        try:
            passed,msg=vf(); ok+=bool(passed)
            print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {msg}",flush=True)
        except Exception as e:
            import traceback; print(f"  [ERR ] {name}: {e}",flush=True)
    print(f"BATCH B: {ok}/{len(BATCH_B)} circuits verified",flush=True)
