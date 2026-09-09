"""ARCHITECTURAL TIER (batch F) — the multi-core / recursive layer from the spec: cross-modal
integration across processing cores, a re-entrant internal-environment loop, and a global-workspace
broadcast. The higher-order composition above single circuits."""
import numpy as np
from paula_agent import ckit as k
def spikes(t): return sum(t)
def rate(t,a,b): return sum(t[a:b])

def f1_multicore_bind():
    """MULTI-CORE cross-modal integration: two cores (X=vision, Y=audio) each produce a feature; a
    CONNECTOR binds them into a unified percept that requires BOTH cores (congruent bimodal) — the
    modality-processor + connector motif from the spec."""
    # 1=coreX out, 2=coreY out, 3=binder (AND across cores)
    ne=[k.neuron(1,r=0.4,c=1),k.neuron(2,r=0.4,c=1),k.neuron(3,r=3.2,lam=8,c=1)]
    sy=[k.syn(1,0,3.0,1),k.term(1),k.syn(2,0,3.0,1),k.term(2),
        k.syn(3,0,2.6,1),k.syn(3,1,2.6,1),k.term(3)]
    conns=[k.conn(1,3,0),k.conn(2,3,1)]; ex=[k.ext(1,0),k.ext(2,0)]; p=k.build(ne,sy,conns,ex)
    def run(x,y):
        d={t:([(1,0,4.0)] if x else [])+([(2,0,4.0)] if y else []) for t in range(4,26)}
        return spikes(k.simulate(p,30,drives=d,probe_ids=[3])[3])
    return p,run
def v1():
    _,run=f1_multicore_bind(); both=run(1,1); x=run(1,0); y=run(0,1)
    ok= both>0 and x==0 and y==0
    return ok,f"Multi-core binding: bimodal-congruent={both} vision-only={x} audio-only={y} (unified percept needs both cores)"

def f2_reentry():
    """RE-ENTRANT LOOP (internal environment): a bound percept re-enters the cores top-down and
    SUSTAINS itself after the stimulus is gone (recurrent self-maintenance = a working percept /
    chained recursion). Transient bimodal input -> persistent bound activity."""
    # 3=percept (bound), self-sustaining via re-entry with delay
    ne=[k.neuron(3,r=0.5,lam=6,c=3)]
    sy=[k.syn(3,0,2.2,1),k.syn(3,1,3.5,1),k.syn(3,2,3.5,4),k.term(3)]  # X-drive, Y-drive, re-entry(delayed)
    conns=[k.conn(3,3,2)]; ex=[k.ext(3,0),k.ext(3,1)]; p=k.build(ne,sy,conns,ex)
    def run():
        d={t:[(3,0,4.0),(3,1,4.0)] for t in range(4,9)}  # transient bimodal trigger
        tr=k.simulate(p,55,drives=d,probe_ids=[3])[3]
        return sum(tr[4:12]),sum(tr[30:52])  # trigger, sustained-after
    return p,run
def v2():
    _,run=f2_reentry(); trig,sustained=run(); ok= trig>0 and sustained>0
    return ok,f"Re-entrant loop: percept@trigger={trig}, sustained-after-stimulus={sustained} (internal-environment self-maintenance)"

def f3_global_workspace():
    """GLOBAL WORKSPACE broadcast: a winning percept is broadcast to MANY downstream modules at once
    (one->all ignition), so the whole system shares access to it — the LLGC/broadcast motif."""
    # 1=percept source -> broadcasts to modules 2,3,4 simultaneously
    ne=[k.neuron(1,r=0.5)]+[k.neuron(i,r=0.6) for i in (2,3,4)]
    sy=[k.syn(1,0,3.0,1),k.term(1)]+[s for i in (2,3,4) for s in (k.syn(i,0,2.5,1),k.term(i))]
    conns=[k.conn(1,i,0) for i in (2,3,4)]; ex=[k.ext(1,0)]; p=k.build(ne,sy,conns,ex)
    def run(active):
        d={t:[(1,0,4.0)] for t in range(4,20)} if active else {}
        o=k.simulate(p,26,drives=d,probe_ids=[2,3,4]); return [spikes(o[i]) for i in (2,3,4)]
    return p,run
def v3():
    _,run=f3_global_workspace(); on=run(True); off=run(False)
    ok= all(m>0 for m in on) and all(m==0 for m in off)  # broadcast reaches ALL modules
    return ok,f"Global workspace: percept-ON modules={on} percept-OFF={off} (broadcast ignites all modules)"

def f4_crossmodal_conflict():
    """CROSS-CORE CONFLICT RESOLUTION: two cores propose different percepts; a supervisor picks the
    stronger (WTA across cores) and suppresses the loser -> a single coherent global interpretation."""
    ne=[k.neuron(1,r=0.6,lam=6),k.neuron(2,r=0.6,lam=6)]
    sy=[k.syn(1,0,2.0,1),k.syn(1,1,-9.0,1),k.term(1),k.syn(2,0,2.0,1),k.syn(2,1,-9.0,1),k.term(2)]
    conns=[k.conn(1,2,1),k.conn(2,1,1)]; ex=[k.ext(1,0),k.ext(2,0)]; p=k.build(ne,sy,conns,ex)
    def run(sX,sY):
        d={t:[(1,0,sX),(2,0,sY)] for t in range(4,26)}
        o=k.simulate(p,30,drives=d,probe_ids=[1,2]); return spikes(o[1]),spikes(o[2])
    return p,run
def v4():
    _,run=f4_crossmodal_conflict(); a,b=run(3.2,1.6)
    ok= a>0 and a>b*2
    return ok,f"Cross-core conflict: coreX(strong)={a} vs coreY(weak)={b} (single coherent interpretation)"

BATCH_F=[("Multi-core cross-modal binding",v1),("Re-entrant internal-environment loop",v2),
         ("Global-workspace broadcast",v3),("Cross-core conflict resolution",v4)]
if __name__=="__main__":
    import sys; sys.path.insert(0,".")
    ok=0
    for name,vf in BATCH_F:
        try:
            passed,msg=vf(); ok+=bool(passed); print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {msg}",flush=True)
        except Exception as e:
            print(f"  [ERR ] {name}: {e}",flush=True)
    print(f"BATCH F: {ok}/{len(BATCH_F)} verified",flush=True)
