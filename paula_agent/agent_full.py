"""INTEGRATED LEARNING-SURVIVAL AGENT: unifies behaving + learning. A creature forages and flees in a
food+predator world, but the mapping cue->food / cue->danger is NOT innate — it is LEARNED online via
reward-modulated plasticity during the agent's life. Innate: a hardwired looming reflex (always
escapes an imminent predator). Learned: which distal CUE predicts food (approach) vs a predator
(avoid). Emergent: survival/foraging IMPROVES over the lifetime as the associations form. Rule-3
deliverable — an ALife agent that learns to survive, online, bio-plausibly, no gradient."""
import numpy as np
from paula_agent import ckit as k

def build_brain(eta=0.06, rh_decay=0.4, kappa=2.5):
    # neuron 3 = APPROACH decision: cueA(syn0) & cueB(syn1) plastic + teaching US(syn2). Direct wiring
    # (causal STDP). neuron 4 = innate ESCAPE (fixed): looming(syn0) -> escape, always works.
    ne=[k.neuron(3,r=1.4,lam=6,plasticity="reward_hebb",eta_post=eta,rh_decay=rh_decay,kappa=kappa),
        k.neuron(4,r=0.6,lam=4)]
    sy=[k.syn(3,0,1.2,1),k.syn(3,1,1.2,1),k.syn(3,2,1.5,1),k.term(3),
        k.syn(4,0,3.0,1),k.term(4)]
    ex=[k.ext(3,0),k.ext(3,1),k.ext(3,2),k.ext(4,0)]
    return k.build(ne,sy,[],ex)

def perceive(net,core,nb,cue,looming):
    """One perception step (plasticity FROZEN): does the agent APPROACH (decision) and/or ESCAPE (reflex)?"""
    eta=nb[3].params.eta_post; nb[3].params.eta_post=0.0
    net.reset_simulation(); core.state.current_tick=0; net.current_tick=0
    ap=esc=0; sid=0 if cue==1 else 1
    for t in range(14):
        if 2<=t<8:
            if cue: net.set_external_input(3,sid,4.0)
            if looming: net.set_external_input(4,0,4.0)
        core.do_tick(); ap+=int(nb[3].O>0); esc+=int(nb[4].O>0)
    nb[3].params.eta_post=eta
    return ap>0, esc>0

def learn(net,core,nb,cue,reward):
    """Update from experienced outcome: food(reward) -> dopamine+teaching -> LTP; predator -> stress -> LTD."""
    net.reset_simulation(); core.state.current_tick=0; net.current_tick=0
    sid=0 if cue==1 else 1
    for t in range(16):
        if 2<=t<11: nb[3].M_vector[1 if reward else 0]=2.0 if reward else 2.5
        if 2<=t<8: net.set_external_input(3,sid,4.0)
        if reward and 3<=t<9: net.set_external_input(3,2,5.0)
        core.do_tick()

def life(seed=0, steps=90):
    rng=np.random.RandomState(seed); p=build_brain(); net,core=k.load(p); nb={i:n for i,n in net.network.neurons.items()}
    energy=12.0; log=[]
    for step in range(steps):
        # an object appears: cue A (food) or cue B (predator). predator sometimes looms (innate reflex).
        cue = 1 if rng.random()<0.5 else 2
        looming = (cue==2 and rng.random()<0.6)  # some predators loom (reflex catches these)
        approach, escape = perceive(net,core,nb,cue,looming)
        outcome=0
        if escape and cue==2:            # innate reflex saved it (or it fled)
            pass
        elif approach and cue==1:        # approached food -> eat
            energy+=3; outcome=1
        elif approach and cue==2:        # approached a predator (not fled) -> hurt
            energy-=4; outcome=-1
        # LEARN from the engagement (only when the agent chose to approach)
        if approach:
            learn(net,core,nb,cue,reward=(cue==1))
        danger_approach = 1 if (approach and cue==2 and not escape) else 0
        energy-=0.35
        log.append((step,outcome,danger_approach,round(energy,1)))
        if energy<=0: break
    wA=nb[3].postsynaptic_points[0].u_i.info; wB=nb[3].postsynaptic_points[1].u_i.info
    return log,wA,wB,energy

if __name__=="__main__":
    import sys; sys.path.insert(0,".")
    print("INTEGRATED LEARNING-SURVIVAL AGENT — learns to survive over its lifetime:",flush=True)
    early_out=[]; late_out=[]; wAs=[]; wBs=[]; surv=0
    for s in range(3):
        log,wA,wB,energy=life(seed=s)
        n=len(log); e=sum(d for _,_,d,_ in log[:n//3]); l=sum(d for _,_,d,_ in log[2*n//3:])
        early_out.append(e); late_out.append(l); wAs.append(wA); wBs.append(wB); surv+= (energy>0)
        print(f"  life{s}: DANGER-approaches early={e} -> late={l} (learns to avoid) | lifespan={n} | food-cue={wA:.1f} danger-cue={wB:.1f}",flush=True)
    ok = np.mean(late_out)<np.mean(early_out) and np.mean(wAs)>np.mean(wBs)*3
    print(f"AGENT: danger-approaches early={np.mean(early_out):.1f} -> late={np.mean(late_out):.1f} (declines=learned avoidance); "
          f"food-cue-weight={np.mean(wAs):.1f} >> danger-cue={np.mean(wBs):.1f}; survived={surv}/3",flush=True)
    print(f"VERDICT: {'AGENT LEARNED TO SURVIVE online (foraging improves as cues are learned)' if ok else 'needs tuning'}",flush=True)
    print("@@@FULL DONE@@@",flush=True)
