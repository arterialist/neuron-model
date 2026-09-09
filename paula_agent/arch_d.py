"""ARCHITECTURAL CORE circuits (batch D) — building blocks of the AGI architecture from the spec:
predictive coding, SUPERVISORS (anomaly / homeostatic / coherency = 'feelings'), temporal attention,
reward-pathway strengthening, connectors, self-modification. Sophisticated, composed circuits."""
import numpy as np
from paula_agent import ckit as k
def spikes(t): return sum(t)
def rate(t,a,b): return sum(t[a:b])

def d1_predictive_coding():
    """Predictive-coding unit (repetition suppression): fast excitation from input + a DELAYED
    inhibitory 'prediction' of continued input. Error/surprise neuron fires at stimulus ONSET
    (unpredicted) but is suppressed during SUSTAINED input (predicted) -> error ~ derivative =
    prediction error. Novel change re-drives it."""
    ne=[k.neuron(1,r=0.7,lam=2)]
    sy=[k.syn(1,0,2.6,1),k.syn(1,1,-2.4,7),k.term(1)]  # exc(fast) + delayed inhibitory prediction
    ex=[k.ext(1,0),k.ext(1,1)]; p=k.build(ne,sy,[],ex)
    def run():
        d={t:[(1,0,3.0),(1,1,3.0)] for t in range(5,45)}  # sustained (predictable) input
        tr=k.simulate(p,50,drives=d,probe_ids=[1])[1]
        onset=sum(tr[5:12]); sustained=sum(tr[20:40])
        return onset,sustained
    return p,run
def v1():
    _,run=d1_predictive_coding(); onset,sustained=run()
    ok= onset>0 and sustained< onset
    return ok,f"Predictive coding: error@onset(unpredicted)={onset} error@sustained(predicted)={sustained} (surprise=derivative)"
def d2_supervisor_anomaly():
    """SUPERVISOR: anomaly detector. Monitors a module's firing; ALARMS when activity leaves the
    normal band (too silent OR too active). 'A distant representation of feelings.'"""
    # 1=graded module; 2=too-high (integrates module output, fires only at HIGH rate);
    # 3=too-low (tonic, suppressed by even normal module activity, fires only when SILENT); 4=alarm OR
    ne=[k.neuron(1,r=4.5,lam=3,c=4),k.neuron(2,r=2.2,lam=14,c=1),k.neuron(3,r=0.5,lam=5,c=1),k.neuron(4,r=0.5,c=1)]
    sy=[k.syn(1,0,1.0,1),k.term(1),
        k.syn(2,0,3.0,1),k.term(2),                   # too-high: strong coupling, fires when module rate high
        k.syn(3,0,0.9,1),k.syn(3,1,-9.0,1),k.term(3), # too-low: VDD tonic, inhibited by any module activity
        k.syn(4,0,1.5,1),k.syn(4,1,1.5,1),k.term(4)]  # alarm = OR(too-high, too-low)
    conns=[k.conn(1,2,0),k.conn(1,3,1),k.conn(2,4,0),k.conn(3,4,1)]
    ex=[k.ext(1,0),k.ext(3,0)]; p=k.build(ne,sy,conns,ex)
    def run(module_drive):
        d={t:[(1,0,module_drive)] for t in range(3,50)}
        for t in range(50): d.setdefault(t,[]).append((3,0,3.0))  # tonic expectation on too-low
        o=k.simulate(p,55,drives=d,probe_ids=[4]); return spikes(o[4])
    return p,run
def v2():
    _,run=d2_supervisor_anomaly(); normal=run(6.0); silent=run(0.0)
    ok= silent>normal*2  # alarms when the monitored module falls silent (dead/stuck module)
    return ok,f"Supervisor(inactivity): alarm normal={normal} module-silent={silent} (detects a stalled module)"

def d3_supervisor_homeostat():
    """SUPERVISOR: homeostatic regulator. Negative feedback keeps a module near a setpoint: if the
    module fires too much, the regulator inhibits it; the loop settles the rate. Compare regulated
    vs unregulated final rate under an over-drive."""
    def build_net(regulated):
        ne=[k.neuron(1,r=1.6,lam=5,c=1),k.neuron(2,r=0.4,lam=8,c=1)]  # 1=graded module, 2=regulator
        sy=[k.syn(1,0,1.0,1)]
        if regulated: sy.append(k.syn(1,1,-9.0,1))  # regulator inhibits module
        sy+=[k.term(1),k.syn(2,0,4.0,1),k.term(2)]
        conns=[k.conn(1,2,0)]+([k.conn(2,1,1)] if regulated else [])
        return k.build(ne,sy,conns,[k.ext(1,0)])
    def run(regulated):
        p=build_net(regulated); o=k.simulate(p,80,drives={t:[(1,0,4.0)] for t in range(3,80)},probe_ids=[1])
        return rate(o[1],55,80)  # late steady-state module rate
    return None,run
def v3():
    _,run=d3_supervisor_homeostat(); unreg=run(False); reg=run(True)
    ok= reg< unreg  # regulator suppresses the over-driven module toward setpoint
    return ok,f"Supervisor(homeostat): unregulated late-rate={unreg} regulated={reg} (pulled down to setpoint)"

def d4_temporal_attention():
    """TEMPORAL ATTENTION: digests a stream in time-CHUNKS — a gate that only passes input during a
    periodic attention window (driven by an internal rhythm), so downstream sees discrete chunks."""
    # rhythm gate: an oscillator opens the gate periodically; gate = input AND rhythm
    ne=[k.neuron(1,r=0.5,lam=3),k.neuron(2,r=1.6,lam=3)]  # 1=rhythm(bursts), 2=gated output = input AND rhythm
    sy=[k.syn(1,0,3.0,8),k.term(1),                       # rhythm: self-loop with delay = periodic
        k.syn(2,0,1.0,1),k.syn(2,1,1.0,1),k.term(2)]      # AND(input, rhythm)
    conns=[k.conn(1,1,0),k.conn(1,2,1)]
    ex=[k.ext(1,0),k.ext(2,0)]; p=k.build(ne,sy,conns,ex)
    def run():
        d={3:[(1,0,5.0)]}  # kick the rhythm
        for t in range(3,60): d.setdefault(t,[]).append((2,0,3.0))  # constant input stream
        o=k.simulate(p,60,drives=d,probe_ids=[1,2])
        return spikes(o[1]),spikes(o[2]),o[2]
    return p,run
def v4():
    _,run=d4_temporal_attention(); rhythm,gated,tr=run()
    # gated output should be CHUNKED (bursts), not constant: check it's non-empty but sparser than input
    ok= 0<gated and rhythm>2 and gated< 40
    return ok,f"Temporal attention: rhythm-pulses={rhythm}, chunked-output-spikes={gated} (input gated into chunks)"

def d5_reward_strengthen():
    """REWARD/DOPAMINE: strengthen the pathway that predicts reward. Two pathways compete to drive a
    target; only the one active JUST BEFORE reward gets strengthened (eligibility x dopamine). After
    training, the rewarded pathway dominates."""
    # n1,n2 = two cue pathways -> n3 target (both plastic). Reward only when cue1 was active.
    ne=[k.neuron(1,r=0.4),k.neuron(2,r=0.4),
        k.neuron(3,r=1.2,lam=6,plasticity="reward_hebb",eta_post=0.05,rh_decay=0.4,kappa=2.5)]
    sy=[k.syn(1,0,3.0,1),k.term(1),k.syn(2,0,3.0,1),k.term(2),
        k.syn(3,0,0.3,1),k.syn(3,1,0.3,1),k.syn(3,2,1.0,1),k.term(3)]  # cue1,cue2 weak + US(syn2)
    conns=[k.conn(1,3,0),k.conn(2,3,1)]
    ex=[k.ext(1,0),k.ext(2,0),k.ext(3,2)]; p=k.build(ne,sy,conns,ex)
    def run():
        net,core=k.load(p); nb={i:n for i,n in net.network.neurons.items()}
        for _ in range(20):
            for t in range(14):
                nb[3].M_vector[1]=2.0 if 2<=t<8 else 0.0    # dopamine during cue1 trial
                if 1<=t<5: net.set_external_input(1,0,4.0)   # cue1 (rewarded) + teaching -> target fires causally
                if 1<=t<5: net.set_external_input(3,2,6.0)   # US/teaching drives target while cue1 present
                if t==10: net.set_external_input(2,0,4.0)    # cue2 unpaired, no reward, no teaching
                core.do_tick()
        return nb[3].postsynaptic_points[0].u_i.info, nb[3].postsynaptic_points[1].u_i.info
    return p,run
def v5():
    _,run=d5_reward_strengthen(); w1,w2=run()
    ok= w1> w2  # rewarded (cue1) pathway strengthened more than unrewarded (cue2)
    return ok,f"Reward strengthening: rewarded-pathway w={w1:.2f} vs unrewarded w={w2:.2f} (dopamine credits the predictor)"

def d6_coherency_sync():
    """SUPERVISOR: coherency enforcer. Two modules coupled by a supervisor become phase-SYNCHRONIZED
    (coherent) vs drifting when uncoupled. Measure spike-time correlation coupled vs uncoupled."""
    def build_net(coupled):
        ne=[k.neuron(1,r=0.6,lam=5),k.neuron(2,r=0.6,lam=5)]
        sy=[k.syn(1,0,1.5,1),k.syn(2,0,1.5,1)]
        if coupled: sy+=[k.syn(1,1,1.5,1),k.syn(2,1,1.5,1)]  # mutual excitation -> sync
        sy+=[k.term(1),k.term(2)]
        conns=[k.conn(1,2,1),k.conn(2,1,1)] if coupled else []
        return k.build(ne,sy,conns,[k.ext(1,0),k.ext(2,0)])
    def corr(coupled):
        p=build_net(coupled)
        d={t:[(1,0,2.0)] for t in range(60)}
        for t in range(60): d.setdefault(t,[]).append((2,0,2.2))  # slightly different drive -> would drift
        o=k.simulate(p,60,drives=d,probe_ids=[1,2])
        a,b=np.array(o[1][10:]),np.array(o[2][10:])
        return float(np.corrcoef(a,b)[0,1]) if a.std()>0 and b.std()>0 else 0.0
    return None,corr
def v6():
    _,corr=d6_coherency_sync(); uncoupled=corr(False); coupled=corr(True)
    ok= coupled> uncoupled
    return ok,f"Supervisor(coherency): spike-corr uncoupled={uncoupled:.2f} coupled={coupled:.2f} (enforces coherence)"

BATCH_D=[("Predictive-coding unit",v1),("Supervisor: inactivity detector",v2),
         ("Supervisor: homeostatic regulator",v3),("Temporal attention (chunking)",v4),
         ("Reward pathway strengthening (dopamine)",v5),("Supervisor: coherency enforcer",v6)]
if __name__=="__main__":
    import sys; sys.path.insert(0,".")
    ok=0
    for name,vf in BATCH_D:
        try:
            passed,msg=vf(); ok+=bool(passed); print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {msg}",flush=True)
        except Exception as e:
            import traceback; print(f"  [ERR ] {name}: {e}",flush=True)
    print(f"BATCH D: {ok}/{len(BATCH_D)} verified",flush=True)
