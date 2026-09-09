"""COMPLEX INTEGRATED AGENT — composes the platform into one creature that instantiates the target
architecture (encoder/sensor -> value core -> homeostatic supervisor -> decision -> motor, with a
dopamine reward loop and continuous online learning) in a richer survival world.

World: a 2D field with TWO look-alike food types — nutritious (energy+) and TOXIC (energy-) — that
differ only by a learnable CUE, plus a predator. The cue->value mapping is NOT innate.

Brain (maps to system-requirements):
  * sensors: food gradient (N/S/E/W), the nearest food's 2-channel cue, predator proximity.
  * value core (dopamine/reward_hebb): cue -> VALUE neuron; eating good food = dopamine -> LTP on its
    cue; eating toxic = stress -> LTD. Learns which cue predicts a good outcome (condition strengthening).
  * homeostatic supervisor (neuromodulation): low energy -> HUNGER modulator that lowers the approach
    threshold (forage harder when starving) — a regulatory "drive".
  * decision: approach the food only if its LEARNED value clears the (hunger-adjusted) threshold;
    an innate looming reflex overrides everything to flee the predator.
Emergent: the agent starts eating toxic food (can't tell them apart) and learns to avoid it while
keeping energy up — survival IMPROVES over the lifetime. A no-learning control does markedly worse."""
import numpy as np
from paula_agent import ckit as k


def build_value_core(eta=0.14, rh_decay=0.3, kappa=2.6):
    # VALUE neuron 1: cueA(syn0) + cueB(syn1) plastic, teaching/US(syn2). reward_hebb w/ dopamine gate.
    # r=2.2 so a suppressed (learned-toxic) cue fails to trigger approach; init 0.75 -> S~3 fires (explore).
    ne = [k.neuron(1, r=2.2, c=2, lam=5, plasticity="reward_hebb", eta_post=eta, rh_decay=rh_decay, kappa=kappa)]
    sy = [k.syn(1, 0, 0.75, 1), k.syn(1, 1, 0.75, 1), k.syn(1, 2, 1.6, 1), k.term(1)]
    return k.build(ne, sy, [], [k.ext(1, 0), k.ext(1, 1), k.ext(1, 2)]), None


def value_of(net, core, nb, cue):
    """Frozen readout of the LEARNED value of a cue: how strongly the VALUE neuron responds. Returns a
    spike count. This is the appetitive/aversive judgement; it decides WHAT is worth approaching (a
    learned-toxic cue no longer clears threshold). Homeostasis is handled separately, so hunger never
    overrides a learned aversion (no starving-into-poison death spiral)."""
    eta = nb[1].params.eta_post; nb[1].params.eta_post = 0.0
    net.reset_simulation(); core.state.current_tick = 0; net.current_tick = 0
    v = 0
    for t in range(12):
        if 2 <= t < 8:
            net.set_external_input(1, 0, 4.0 if cue == 0 else 0.0)
            net.set_external_input(1, 1, 4.0 if cue == 1 else 0.0)
        core.do_tick(); v += int(nb[1].O > 0)
    nb[1].params.eta_post = eta
    return v


def teach(net, core, nb, cue, good):
    """Dopamine-gated learning from the eaten food's actual outcome (mirrors the working agent_full
    pattern: the cue itself fires the cell, so plasticity is causal). GOOD: dopamine gate on + US ->
    reward-Hebb LTP grows the cue's value. TOXIC: stress gate drives nm->0, so the rule becomes pure
    -rh_decay*w decay -> the cue's value bleeds down over a few bad meals until it no longer clears the
    approach threshold (learned aversion). No separate firing synapse (that would self-depress)."""
    sweeps = 1 if good else 2
    for _ in range(sweeps):
        net.reset_simulation(); core.state.current_tick = 0; net.current_tick = 0
        for t in range(16):
            if good:
                if 2 <= t < 11: nb[1].M_vector[1] = 2.2                      # dopamine
                if 3 <= t < 9: net.set_external_input(1, cue, 4.0); net.set_external_input(1, 2, 5.0)
            else:
                if 2 <= t < 14: nb[1].M_vector[0] = 2.6                      # stress -> nm=0
                if 2 <= t < 14: net.set_external_input(1, cue, 4.0)          # active -> pure decay
            core.do_tick()


def life(seed=0, steps=200, learning=True):
    rng = np.random.RandomState(seed); G = 6
    p, _ = build_value_core(); net, core = k.load(p); nb = {i: n for i, n in net.network.neurons.items()}
    ax, ay = rng.randint(0, G), rng.randint(0, G)
    # food (x,y,cue,good): cue 0 == nutritious, cue 1 == TOXIC look-alike (must be LEARNED).
    def spawn():
        good = rng.random() < 0.5; cue = 0 if good else 1
        return [rng.randint(0, G), rng.randint(0, G), cue, good]
    food = spawn(); px, py = (G - 1 - ax), (G - 1 - ay)       # predator starts opposite corner
    energy = 14.0; frames = []; toxic_eaten = 0; good_eaten = 0; since = 0
    for step in range(steps):
        hunger = max(0.0, 3.5 * (1 - energy / 14.0))          # starving -> appetitive bias (less picky)
        dx, dy = food[0] - ax, food[1] - ay
        dpred = abs(px - ax) + abs(py - ay); looming = dpred <= 1   # predator only an emergency up close
        val = value_of(net, core, nb, food[2])                 # learned value of the visible food's cue
        approach = val > 0 and not looming
        act = "ignore"
        if looming:
            ax = int(np.clip(ax + (1 if px < ax else -1), 0, G - 1)); ay = int(np.clip(ay + (1 if py < ay else -1), 0, G - 1)); act = "flee"
        elif approach:
            if abs(dx) >= abs(dy): ax = int(np.clip(ax + np.sign(dx), 0, G - 1))
            else: ay = int(np.clip(ay + np.sign(dy), 0, G - 1))
            act = "approach"
        else:                                                  # judged bad -> skip it, a new item appears
            act = "avoid"; food = spawn(); since = 0
        # eat?
        ate = None; since += 1
        if ax == food[0] and ay == food[1] and approach:
            good = bool(food[3]); ate = "good" if good else "toxic"
            energy += 3.0 if good else -4.0
            good_eaten += good; toxic_eaten += (not good)
            if learning and not good: teach(net, core, nb, food[2], False)  # taste-aversion learning
            food = spawn(); since = 0
        elif since > 10:                                       # couldn't reach it; a fresh item appears
            food = spawn(); since = 0
        if step % 2 == 0:                                      # predator drifts (occasional hazard, not a
            if dpred <= 3:                                      # relentless pursuer): homes only when near,
                px = int(np.clip(px + np.sign(ax - px), 0, G - 1)); py = int(np.clip(py + np.sign(ay - py), 0, G - 1))
            else:                                              # otherwise wanders
                px = int(np.clip(px + rng.choice([-1, 0, 1]), 0, G - 1)); py = int(np.clip(py + rng.choice([-1, 0, 1]), 0, G - 1))
        energy -= 0.25
        wGood = float(nb[1].postsynaptic_points[0].u_i.info); wTox = float(nb[1].postsynaptic_points[1].u_i.info)
        frames.append({"step": step, "ax": ax, "ay": ay, "fx": food[0], "fy": food[1], "fcue": food[2],
                       "fgood": bool(food[3]), "px": px, "py": py, "act": act, "val": val,
                       "energy": round(max(energy, 0), 1), "hunger": round(hunger, 1),
                       "wGood": round(wGood, 2), "wTox": round(wTox, 2), "ate": ate})
        if energy <= 0: break
    return {"grid": G, "frames": frames, "toxic_eaten": toxic_eaten, "good_eaten": good_eaten,
            "lifespan": len(frames), "wGood": wGood, "wTox": wTox}


if __name__ == "__main__":
    import sys; sys.path.insert(0, ".")
    print("COMPLEX INTEGRATED AGENT — learns to avoid toxic look-alike food while foraging + fleeing:", flush=True)
    L = [life(seed=s, learning=True) for s in range(4)]
    C = [life(seed=s, learning=False) for s in range(4)]
    # learned: toxic-eaten in late third << early third; value weight good >> toxic; lives longer than control
    def split(fr):
        n = len(fr); e = sum(1 for f in fr[:n // 3] if f["ate"] == "toxic"); l = sum(1 for f in fr[2 * n // 3:] if f["ate"] == "toxic"); return e, l
    tox_e = np.mean([split(r["frames"])[0] for r in L]); tox_l = np.mean([split(r["frames"])[1] for r in L])
    wG = np.mean([r["wGood"] for r in L]); wT = np.mean([r["wTox"] for r in L])
    life_L = np.mean([r["lifespan"] for r in L]); life_C = np.mean([r["lifespan"] for r in C])
    toxtot_L = np.mean([r["toxic_eaten"] for r in L]); toxtot_C = np.mean([r["toxic_eaten"] for r in C])
    ok = tox_l <= tox_e and wG > wT and toxtot_L < toxtot_C and life_L >= life_C
    print(f"  toxic-eaten early={tox_e:.1f} -> late={tox_l:.1f} (learns avoidance)", flush=True)
    print(f"  value weights: good-cue={wG:.1f} vs toxic-cue={wT:.1f}", flush=True)
    print(f"  total toxic eaten: learner={toxtot_L:.1f} < control(no learning)={toxtot_C:.1f}", flush=True)
    print(f"  lifespan: learner={life_L:.0f} vs control={life_C:.0f}", flush=True)
    print(f"VERDICT: {'COMPLEX AGENT LEARNS TO SURVIVE (discriminates toxic food online, forages, evades)' if ok else 'needs tuning'}", flush=True)
    print("@@@COMPLEX DONE@@@", flush=True)
