"""Agent circuit library — BATCH C: behavioral arbitration + a COMPOSITE sensorimotor agent that
chains circuits into emergent adaptive behavior (the 'complex behavior' capstone)."""
import numpy as np
from paula_agent import ckit as k
def spikes(t): return sum(t)

def c22_arbiter():
    """Subsumption arbiter: two behaviors compete; ESCAPE has PRIORITY over SEEK (escape inhibits
    seek's output when active). Higher-priority behavior suppresses lower when both are triggered."""
    ne=[k.neuron(1,r=0.5),k.neuron(2,r=0.5)]  # 1=escape-cmd, 2=seek-cmd
    sy=[k.syn(1,0,2.0,1),k.term(1),
        k.syn(2,0,2.0,1),k.syn(2,1,-12.0,1),k.term(2)]  # seek inhibited by escape
    conns=[k.conn(1,2,1)]  # escape -> inhibits seek
    ex=[k.ext(1,0),k.ext(2,0)]; p=k.build(ne,sy,conns,ex)
    def run(escape_on,seek_on):
        d={}
        for t in range(5,25,3):
            row=[]
            if escape_on: row.append((1,0,3.0))
            if seek_on: row.append((2,0,3.0))
            d[t]=row
        o=k.simulate(p,40,drives=d,probe_ids=[1,2]); return spikes(o[1]),spikes(o[2])
    return p,run
def v22():
    _,run=c22_arbiter(); seek_only=run(False,True); both=run(True,True)
    ok= seek_only[1]>0 and both[0]>0 and both[1] <= seek_only[1]*0.3  # escape wins, seek suppressed
    return ok,f"Arbiter: seek-alone={seek_only} both-triggered={both} (escape overrides seek)"

def c23_startle():
    """Startle/escape trigger: a looming/sudden input evokes a fast high-gain BURST (all-or-none escape)."""
    ne=[k.neuron(1,r=3.2,lam=3,c=1),k.neuron(2,r=0.35,lam=5,c=1)]  # 1=detector(high thr), 2=burst generator
    sy=[k.syn(1,0,3.0,1),k.term(1),
        k.syn(2,0,3.0,1),k.syn(2,1,5.0,1),k.term(2)]  # burst neuron self-excites for a burst
    conns=[k.conn(1,2,0),k.conn(2,2,1)]
    ex=[k.ext(1,0)]; p=k.build(ne,sy,conns,ex)
    def run(strength):
        o=k.simulate(p,40,drives={t:[(1,0,strength)] for t in range(5,9)},probe_ids=[2]); return spikes(o[2])
    return p,run
def v23():
    _,run=c23_startle(); strong=run(4.0); weak=run(0.8)
    ok= strong>=3 and weak< strong  # strong stimulus -> escape burst
    return ok,f"Startle: strong-stim burst={strong} spikes, weak-stim={weak} (all-or-none escape)"

def c24_gainfield():
    """Gain modulation (coordinate transform primitive): a neuromod scales a neuron's INPUT gain, so
    the output = input x gain (multiplicative-like) -> basis for sensorimotor coordinate transforms."""
    ne=[k.neuron(1,r=1.7,lam=6,c=2,w_r=[0,-0.9])]  # modulator lowers threshold => amplifies input effect
    sy=[k.syn(1,0,1.0,1),k.term(1)]; ex=[k.ext(1,0)]; p=k.build(ne,sy,[],ex)
    def out(inp,gain):
        def mod(t,nb): nb[1].M_vector[1]=gain
        return spikes(k.simulate(p,60,drives={t:[(1,0,inp)] for t in range(5,60)},mod=mod,probe_ids=[1])[1])
    return p,out
def v24():
    _,out=c24_gainfield()
    g0=out(1.4,0.0); g1=out(1.4,1.2); hi_i=out(1.9,0.6); lo_i=out(1.2,0.6)
    ok= g1>g0 and hi_i>lo_i  # output scales with gain (fixed input) AND with input (fixed gain)
    return ok,f"Gain field: gain 0->1 gives {g0}->{g1}; at fixed gain input 0.9->1.6 gives {lo_i}->{hi_i}"

def c25_agent():
    """COMPOSITE AGENT (complex behavior): a predator-escape vs food-approach creature.
    PREDATOR sensor -> ESCAPE command (priority). FOOD sensor -> APPROACH command. Arbiter: escape
    suppresses approach. Each command drives a MOTOR output. Emergent behavior: forages toward food,
    but escapes (overriding foraging) when a predator appears."""
    # neurons: 1=predator sensor, 2=food sensor, 3=escape cmd, 4=approach cmd, 5=escape motor, 6=approach motor
    ne=[k.neuron(1,r=0.5),k.neuron(2,r=0.5),k.neuron(3,r=0.6),k.neuron(4,r=0.6),k.neuron(5,r=0.5),k.neuron(6,r=0.5)]
    sy=[k.syn(1,0,3.0,1),k.term(1),
        k.syn(2,0,3.0,1),k.term(2),
        k.syn(3,0,2.5,1),k.term(3),                    # escape cmd from predator
        k.syn(4,0,2.5,1),k.syn(4,1,-7.0,1),k.term(4),  # approach cmd from food, inhibited by escape
        k.syn(5,0,2.5,1),k.term(5),                    # escape motor
        k.syn(6,0,2.5,1),k.term(6)]                    # approach motor
    conns=[k.conn(1,3,0),k.conn(2,4,0),k.conn(3,4,1),k.conn(3,5,0),k.conn(4,6,0)]
    ex=[k.ext(1,0),k.ext(2,0)]; p=k.build(ne,sy,conns,ex)
    def episode(predator,food):
        d={}
        for t in range(5,30,3):
            row=[]
            if food: row.append((2,0,3.0))
            if predator and 12<=t<24: row.append((1,0,3.0))  # predator appears mid-episode
            d[t]=row
        o=k.simulate(p,40,drives=d,probe_ids=[5,6]); return spikes(o[5]),spikes(o[6])  # escape-motor, approach-motor
    return p,episode
def v25():
    _,ep=c25_agent()
    food_only=ep(False,True)      # should approach
    predator=ep(True,True)        # predator appears -> escape, approach suppressed
    ok= food_only[0]==0 and food_only[1]>0 and predator[0]>0 and predator[1] < food_only[1]
    return ok,(f"COMPOSITE AGENT: food-only (escape,approach)={food_only} | "
               f"predator+food={predator} -> escapes & suppresses foraging")

BATCH_C=[("Subsumption arbiter",v22),("Startle/escape burst",v23),("Gain-field modulation",v24),
         ("COMPOSITE sensorimotor agent",v25)]
if __name__=="__main__":
    import sys; sys.path.insert(0,".")
    ok=0
    for name,vf in BATCH_C:
        try:
            passed,msg=vf(); ok+=bool(passed); print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {msg}",flush=True)
        except Exception as e:
            import traceback; print(f"  [ERR ] {name}: {e}\n{traceback.format_exc()[-300:]}",flush=True)
    print(f"BATCH C: {ok}/{len(BATCH_C)} circuits verified",flush=True)
