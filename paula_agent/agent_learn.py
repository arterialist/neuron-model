"""LEARNING AGENT: closes the loop between the verified reward-plasticity circuits and behavior.
The world has two arbitrary cues; cue A marks FOOD, cue B marks a PREDATOR — but the agent does NOT
know this initially. Plastic cue->approach synapses (reward_hebb): approaching cue A yields food ->
dopamine -> LTP (learn to approach A); approaching cue B yields a predator -> stress -> the
A-approach path is credited while B-approach is not, so over episodes the agent LEARNS to approach A
and avoid B. Verified by a LEARNING CURVE: performance improves early->late, and cueA-weight >
cueB-weight after learning."""
import numpy as np
from paula_agent import ckit as k

def build_brain(eta=0.06, rh_decay=0.4, kappa=2.5):
    # 3=APPROACH decision neuron; cue A/B drive its plastic synapses DIRECTLY (syn0/syn1); syn2=teaching US
    ne=[k.neuron(3,r=1.4,lam=6,plasticity="reward_hebb",eta_post=eta,rh_decay=rh_decay,kappa=kappa)]
    sy=[k.syn(3,0,1.2,1),k.syn(3,1,1.2,1),k.syn(3,2,1.5,1),k.term(3)]  # cueA,cueB plastic (explore) + US
    ex=[k.ext(3,0),k.ext(3,1),k.ext(3,2)]
    return k.build(ne,sy,[],ex)

def trial(net, core, nb, cue):
    """Present a cue with plasticity FROZEN; return whether APPROACH neuron fired (agent's choice)."""
    sid=0 if cue==1 else 1
    eta=nb[3].params.eta_post; nb[3].params.eta_post=0.0
    net.reset_simulation(); core.state.current_tick=0; net.current_tick=0
    fired=0
    for t in range(14):
        if 2<=t<8: net.set_external_input(3,sid,4.0)
        core.do_tick(); fired+=int(nb[3].O>0)
    nb[3].params.eta_post=eta
    return fired>0

def learn_step(net, core, nb, cue, reward):
    """Re-experience the cue WITH its outcome. FOOD(cue A): dopamine + teaching US -> LTP.
    PREDATOR(cue B): stress -> LTD (the moderate cue weight fires the neuron causally, stress makes
    the update net-negative -> that cue->approach synapse decays)."""
    sid=0 if cue==1 else 1
    net.reset_simulation(); core.state.current_tick=0; net.current_tick=0
    for t in range(16):
        if 2<=t<11:
            if reward: nb[3].M_vector[1]=2.0   # dopamine
            else:      nb[3].M_vector[0]=2.5   # stress
        if 2<=t<8:  net.set_external_input(3,sid,4.0)          # cue direct to decision neuron
        if reward and 3<=t<9: net.set_external_input(3,2,5.0)  # teaching US overlaps cue -> causal LTP (as in conditioning)
        core.do_tick()

def run(seed=0, episodes=40):
    rng=np.random.RandomState(seed); p=build_brain()
    net,core=k.load(p); nb={i:n for i,n in net.network.neurons.items()}
    hist=[]
    for ep in range(episodes):
        food=0; caught=0
        for _ in range(6):
            cue = 1 if rng.random()<0.5 else 2   # cue A(food) or B(predator)
            approach = trial(net,core,nb,cue)
            reward = (cue==1)  # approaching A = food; B = predator
            if approach and cue==1: food+=1
            if approach and cue==2: caught+=1
            # LEARN from the experienced outcome (only when the agent engaged the cue)
            if approach:
                learn_step(net,core,nb,cue,reward=reward)
        hist.append((food,caught))
    wA=nb[3].postsynaptic_points[0].u_i.info; wB=nb[3].postsynaptic_points[1].u_i.info
    return hist,wA,wB

if __name__=="__main__":
    import sys; sys.path.insert(0,".")
    print("LEARNING AGENT — learns which cue is food vs predator over episodes:",flush=True)
    allearly=[]; alllate=[]; wAs=[]; wBs=[]
    for s in range(3):
        hist,wA,wB=run(seed=s)
        early=np.mean([h[0]-h[1] for h in hist[:8]])   # net gain (food-caught) early
        late=np.mean([h[0]-h[1] for h in hist[-8:]])   # late
        allearly.append(early); alllate.append(late); wAs.append(wA); wBs.append(wB)
        print(f"  seed{s}: net-gain early={early:+.2f} -> late={late:+.2f} | learned weights cueA={wA:.2f} cueB={wB:.2f}",flush=True)
    ok = np.mean(alllate)>np.mean(allearly) and np.mean(wAs)>np.mean(wBs)
    print(f"AGENT LEARNING: early net-gain={np.mean(allearly):+.2f} -> late={np.mean(alllate):+.2f}; "
          f"cueA-weight={np.mean(wAs):.2f} > cueB-weight={np.mean(wBs):.2f}",flush=True)
    print(f"VERDICT: {'AGENT LEARNED cue->outcome associations online' if ok else 'needs tuning'}",flush=True)
    print("@@@LEARN DONE@@@",flush=True)
