"""Agent circuit library — BATCH A: sensory, memory, decision. Each build_* returns a net path;
each verify runs it and asserts the intended behavior. PAULA as platform (dendritic delays,
inhibition, thresholds, self-excitation)."""
import numpy as np
from paula_agent import ckit as k

def spikes(train): return sum(train)

# ---------------- SENSORY ----------------
def c1_reichardt():
    """Direction-selective motion detector (rightward cell R, leftward cell L)."""
    dL,dS,w=12,3,0.9
    ne=[k.neuron(1,r=1.4,lam=4),k.neuron(2,r=1.4,lam=4)]
    sy=[k.syn(1,0,w,dL),k.syn(1,1,w,dS),k.term(1),
        k.syn(2,0,w,dL),k.syn(2,1,w,dS),k.term(2,tid=901)]
    ex=[k.ext(1,0),k.ext(1,1),k.ext(2,0),k.ext(2,1)]
    p=k.build(ne,sy,[],ex)
    def present(direction,v,t0=5):
        tA,tB=(t0,t0+v) if direction=="right" else (t0+v,t0)
        dr={tA:[(1,0,4.0),(2,1,4.0)], tB:[(1,1,4.0),(2,0,4.0)]}
        # merge same-tick
        d={}
        for t,lst in dr.items(): d.setdefault(t,[]).extend(lst)
        o=k.simulate(p,60,drives=d,probe_ids=[1,2]); return spikes(o[1]),spikes(o[2])
    return p,present
def v1():
    _,present=c1_reichardt(); R_r,L_r=present("right",9); R_l,L_l=present("left",9)
    ok = R_r>0 and L_r==0 and L_l>0 and R_l==0
    return ok,f"Reichardt: right->R={R_r},L={L_r} | left->R={R_l},L={L_l}"

def c2_looming():
    """Looming detector: fires only on synchronous MASS activation (approaching object), not sparse."""
    N=8
    ne=[k.neuron(1,r=6.5,lam=2)]  # high threshold: needs ~6 coincident inputs
    sy=[k.syn(1,i,1.0,1) for i in range(N)]+[k.term(1)]
    ex=[k.ext(1,i) for i in range(N)]
    p=k.build(ne,sy,[],ex)
    def drive(n_active, spread):
        d={}
        for i in range(n_active): d.setdefault(5+i*spread,[]).append((1,i,4.0))
        return spikes(k.simulate(p,40,drives=d,probe_ids=[1])[1])
    return p,drive
def v2():
    _,drive=c2_looming(); loom=drive(8,0); sparse=drive(3,0); spread=drive(8,4)
    ok = loom>0 and sparse==0 and spread==0
    return ok,f"Looming: mass-synchronous={loom} sparse={sparse} spread-out={spread}"

def c3_localizer():
    """Jeffress place-code: bank of coincidence detectors tuned to different input time-differences."""
    banks=[(6,3),(9,3),(12,3),(15,3),(18,3)]; peaks=[]
    for j,(dX,dY) in enumerate(banks):
        ne=[k.neuron(1,r=2.1,lam=2)]; sy=[k.syn(1,0,0.9,dX),k.syn(1,1,0.9,dY),k.term(1)]; ex=[k.ext(1,0),k.ext(1,1)]
        p=k.build(ne,sy,[],ex); curve=[]
        for D in range(0,22):
            d={5:[(1,0,4.0)],5+D:[(1,1,4.0)]}
            if D==0: d={5:[(1,0,4.0),(1,1,4.0)]}
            curve.append(spikes(k.simulate(p,50,drives=d,probe_ids=[1])[1]))
        peaks.append(int(np.argmax(curve)) if max(curve)>0 else -1)
    return peaks
def v3():
    peaks=c3_localizer(); mono=all(peaks[i]<peaks[i+1] for i in range(len(peaks)-1))
    return mono,f"Localizer place-code peaks (should increase): {peaks}"

def c4_edge():
    """Center-surround edge detector: center excitatory, surround inhibitory -> fires for edge, not uniform."""
    ne=[k.neuron(1,r=0.8,lam=4)]
    sy=[k.syn(1,0,2.0,1),k.syn(1,1,-2.0,1),k.term(1)]  # 0=center(+), 1=surround(-)
    ex=[k.ext(1,0),k.ext(1,1)]; p=k.build(ne,sy,[],ex)
    def stim(center,surround):
        d={}
        for t in range(5,20,3):
            row=[]
            if center: row.append((1,0,3.0))
            if surround: row.append((1,1,3.0))
            d[t]=row
        return spikes(k.simulate(p,40,drives=d,probe_ids=[1])[1])
    return p,stim
def v4():
    _,stim=c4_edge(); edge=stim(True,False); uniform=stim(True,True)
    ok= edge>0 and uniform< edge
    return ok,f"Edge(center-surround): edge(center only)={edge} uniform(center+surround)={uniform}"

def c5_novelty():
    """Change/novelty detector: responds to input ONSET, habituates to sustained (via delayed self-inhibition)."""
    ne=[k.neuron(1,r=0.7,lam=2)]
    sy=[k.syn(1,0,2.5,1),k.syn(1,1,-2.4,7),k.term(1)]  # same input -> fast exc (syn0) + DELAYED inhib (syn1)
    ex=[k.ext(1,0),k.ext(1,1)]; conns=[]
    p=k.build(ne,sy,conns,ex)
    def run(sustained_ticks):
        d={t:[(1,0,3.0),(1,1,3.0)] for t in range(5,5+sustained_ticks)}
        tr=k.simulate(p,5+sustained_ticks+5,drives=d,probe_ids=[1])[1]
        early=sum(tr[5:12]); late=sum(tr[15:25]) if len(tr)>25 else 0
        return early,late
    return p,run
def v5():
    _,run=c5_novelty(); early,late=run(30)
    ok= early>0 and late< max(1,early)  # responds at onset, habituates
    return ok,f"Novelty: onset-response={early} sustained-late-response={late} (habituates)"

# ---------------- MEMORY ----------------
def c6_latch():
    """Working-memory bistable latch: brief SET -> persistent self-sustained firing; RESET -> off."""
    ne=[k.neuron(1,r=0.4,lam=10,c=2)]
    sy=[k.syn(1,0,3.0,1),k.syn(1,1,4.5,1),k.syn(1,2,-8.0,1),k.term(1)]  # 0=SET,1=self-exc,2=RESET
    ex=[k.ext(1,0),k.ext(1,2)]; conns=[k.conn(1,1,1)]  # self-excitation
    p=k.build(ne,sy,conns,ex)
    def run(set_t=5, reset_t=None, T=60):
        d={t:[(1,0,4.0)] for t in range(set_t,set_t+3)}
        if reset_t:
            for t in range(reset_t,reset_t+3): d.setdefault(t,[]).append((1,2,5.0))
        tr=k.simulate(p,T,drives=d,probe_ids=[1])[1]
        after_set=sum(tr[15:25]); after_reset=sum(tr[reset_t+5:reset_t+15]) if reset_t else None
        return after_set,after_reset
    return p,run
def v6():
    _,run=c6_latch(); held,_=run(); _,after_reset=run(reset_t=30)
    ok= held>0 and (after_reset==0)
    return ok,f"Latch: persists-after-SET={held} activity-after-RESET={after_reset}"

def c7_delaybuffer():
    """Delay-line memory buffer: signal appears at output N ticks after input (short-term memory)."""
    chain=[1,2,3,4]; ne=[k.neuron(n,r=0.4,lam=3) for n in chain]
    sy=[]; conns=[]
    for n in chain:
        sy+=[k.syn(n,0,3.0,3),k.term(n,tid=900+n)]
    for a,b in zip(chain,chain[1:]):
        conns.append(k.conn(a,b,0,stid=900+a))
    ex=[k.ext(1,0)]; p=k.build(ne,sy,conns,ex)
    def run():
        o=k.simulate(p,40,drives={5:[(1,0,4.0)]},probe_ids=chain)
        firsts=[ (o[n].index(1) if 1 in o[n] else -1) for n in chain]
        return firsts
    return p,run
def v7():
    _,run=c7_delaybuffer(); firsts=run()
    ok= all(f>=0 for f in firsts) and all(firsts[i]<firsts[i+1] for i in range(len(firsts)-1))
    return ok,f"Delay-buffer: first-spike tick per stage (increasing) = {firsts}"

# ---------------- DECISION ----------------
def c8_wta():
    """Winner-take-all: N neurons, mutual inhibition, strongest input wins (only 1 fires)."""
    N=3; ne=[k.neuron(i+1,r=0.6,lam=8) for i in range(N)]
    sy=[]; conns=[]
    for i in range(N):
        nid=i+1; sy+=[k.syn(nid,0,2.0,1)]  # input
        for j in range(N):
            if j!=i: sy.append(k.syn(nid,1+j,-16.0,1))  # inhibition from others
        sy.append(k.term(nid,tid=900+nid))
    for i in range(N):
        for j in range(N):
            if j!=i: conns.append(k.conn(j+1,i+1,1+j,stid=900+j+1))
    ex=[k.ext(i+1,0) for i in range(N)]; p=k.build(ne,sy,conns,ex)
    def run(inputs):
        d={}
        for t in range(5,25,2):
            for i,a in enumerate(inputs): d.setdefault(t,[]).append((i+1,0,a))
        o=k.simulate(p,40,drives=d,probe_ids=[1,2,3]); return [spikes(o[i]) for i in [1,2,3]]
    return p,run
def v8():
    _,run=c8_wta(); out=run([3.0,4.5,2.0])
    win=int(np.argmax(out)); others=sum(out)-out[win]
    ok= win==1 and out[1]>0 and others<=out[1]*0.4
    return ok,f"WTA: spikes per unit (input 3/4.5/2)={out}; winner={win}"

def c9_accumulator():
    """Evidence accumulator (drift-diffusion): integrates evidence, fires (decides) at threshold;
    stronger evidence -> faster decision."""
    ne=[k.neuron(1,r=1.15,lam=40,c=2)]  # long lambda = integrator
    sy=[k.syn(1,0,1.0,1),k.term(1)]; ex=[k.ext(1,0)]; p=k.build(ne,sy,[],ex)
    def run(evidence,T=200):
        tr=k.simulate(p,T,drives={t:[(1,0,evidence)] for t in range(5,T)},probe_ids=[1])[1]
        return tr.index(1) if 1 in tr else -1  # decision tick
    return p,run
def v9():
    _,run=c9_accumulator(); tw=run(2.5); ts=run(1.2)
    ok= tw>0 and ts>0 and tw<ts  # strong evidence decides sooner
    return ok,f"Accumulator: decision tick strong-ev={tw} weak-ev={ts} (strong sooner)"

def c10_twochoice():
    """Two-choice cross-inhibition decision: two accumulators mutually inhibit -> the one with more
    evidence wins and suppresses the other."""
    ne=[k.neuron(1,r=1.4,lam=30),k.neuron(2,r=1.4,lam=30)]
    sy=[k.syn(1,0,1.0,1),k.syn(1,1,-3.0,1),k.term(1),
        k.syn(2,0,1.0,1),k.syn(2,1,-3.0,1),k.term(2,tid=902)]
    conns=[k.conn(1,2,1),k.conn(2,1,1,stid=902)]
    ex=[k.ext(1,0),k.ext(2,0)]; p=k.build(ne,sy,conns,ex)
    def run(evA,evB,T=140):
        d={t:[(1,0,evA),(2,0,evB)] for t in range(5,T)}
        o=k.simulate(p,T,drives=d,probe_ids=[1,2]); return spikes(o[1]),spikes(o[2])
    return p,run
def v10():
    _,run=c10_twochoice(); a,b=run(2.2,1.4)
    ok= a>0 and a>b*2
    return ok,f"Two-choice: A-evidence>B -> A={a} spikes, B={b} spikes"

BATCH_A=[("Reichardt motion detector",v1),("Looming detector",v2),("Jeffress localizer",v3),
         ("Center-surround edge detector",v4),("Novelty/change detector",v5),
         ("Working-memory latch",v6),("Delay-line buffer",v7),("Winner-take-all",v8),
         ("Evidence accumulator",v9),("Two-choice decision",v10)]
if __name__=="__main__":
    import sys; sys.path.insert(0,".")
    ok=0
    for name,vf in BATCH_A:
        try:
            passed,msg=vf(); ok+=passed
            print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {msg}",flush=True)
        except Exception as e:
            print(f"  [ERR ] {name}: {e}",flush=True)
    print(f"BATCH A: {ok}/{len(BATCH_A)} circuits verified",flush=True)
