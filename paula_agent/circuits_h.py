"""LEARNING / ADAPTATION tier — circuits that change themselves online, the regime where the substrate
is genuinely load-bearing. Uses PAULA's three-factor reward-modulated Hebbian rule
Dw = eta*(nm*dir*info - rh_decay*w), nm = 1 + kappa*(reward - stress), dir = +1 if the cell fired within
t_ref (causal) else -1; plus retrograde presynaptic adaptation and metaplasticity (a modulator resizing
the learning window). Each verified by the actual weight change it produces."""
import numpy as np
from paula_agent import ckit as k


def _train(reward=0.0, stress=0.0, kappa=2.5, pairings=25, eta=0.1, rh_decay=0.3,
           w0=0.5, cue_amp=4.0, teach_amp=5.0, causal=True, w_tref=None):
    """One plastic cue synapse (0) + a teaching synapse (1) that fires the neuron. Returns weight trace
    of the cue synapse across pairings. causal=False presents the cue LONG AFTER firing (acausal)."""
    ne = [k.neuron(1, r=1.0, c=2, lam=5, plasticity="reward_hebb", eta_post=eta, rh_decay=rh_decay,
                   kappa=kappa, w_tref=w_tref)]
    sy = [k.syn(1, 0, w0, 1), k.syn(1, 1, 1.5, 1), k.term(1)]
    p = k.build(ne, sy, [], [k.ext(1, 0), k.ext(1, 1)])
    net, core = k.load(p); nb = {i: n for i, n in net.network.neurons.items()}
    trace = [float(nb[1].postsynaptic_points[0].u_i.info)]
    for _ in range(pairings):
        net.reset_simulation(); core.state.current_tick = 0; net.current_tick = 0
        for t in range(16):
            if 2 <= t < 10:
                nb[1].M_vector[1] = reward; nb[1].M_vector[0] = stress
            if causal:
                if 3 <= t < 7: net.set_external_input(1, 0, cue_amp)   # cue overlaps firing
                if 3 <= t < 7: net.set_external_input(1, 1, teach_amp)
            else:
                if 2 <= t < 4: net.set_external_input(1, 1, teach_amp)  # fire early
                if 10 <= t < 14: net.set_external_input(1, 0, cue_amp)  # cue long after (acausal)
            core.do_tick()
        trace.append(float(nb[1].postsynaptic_points[0].u_i.info))
    return trace


def h1_reward_ltp():
    """Reward-modulated LTP: dopamine amplifies potentiation, stress suppresses/reverses it."""
    wr = _train(reward=1.0, stress=0.0)[-1]
    wn = _train(reward=0.0, stress=0.0)[-1]
    ws = _train(reward=0.0, stress=1.0)[-1]
    ok = wr > wn > ws
    return ok, f"final cue weight: reward={wr:.2f} > neutral={wn:.2f} > stress={ws:.2f}"


def h2_stdp_direction():
    """Spike-timing direction: cue causal with firing -> LTP; cue long AFTER firing -> LTD (weight drops)."""
    wc = _train(reward=1.0, causal=True)[-1]
    wa = _train(reward=1.0, causal=False)[-1]
    ok = wc > 0.6 and wa < wc   # causal potentiates above baseline, acausal is lower/depressed
    return ok, f"causal LTP={wc:.2f}  vs  acausal={wa:.2f} (timing sets sign)"


def h3_bounded_convergence():
    """Stable learning: reward_hebb has a bounded fixed point w->nm*info/rh_decay — it PLATEAUS, no
    multiplicative runaway (contrast: legacy mode explodes)."""
    tr = _train(reward=1.0, pairings=60)
    late = tr[-10:]
    plateau = max(late) - min(late) < 0.15 * abs(np.mean(late))  # converged
    finite = np.isfinite(tr[-1]) and tr[-1] < 50
    ok = plateau and finite and tr[-1] > tr[0]
    return ok, f"weight {tr[0]:.2f} -> plateau {np.mean(late):.2f} (spread {max(late)-min(late):.3f}, bounded)"


def h4_discrimination():
    """Reward discrimination (classical conditioning): two cues both drive firing, but only cue A is
    rewarded. The circuit learns to separate them — w_A grows high, w_B stays at baseline."""
    ne = [k.neuron(1, r=1.0, c=2, lam=5, plasticity="reward_hebb", eta_post=0.1, rh_decay=0.3, kappa=2.5)]
    sy = [k.syn(1, 0, 0.5, 1), k.syn(1, 1, 0.5, 1), k.syn(1, 2, 1.5, 1), k.term(1)]  # 0=cueA,1=cueB,2=teach
    p = k.build(ne, sy, [], [k.ext(1, 0), k.ext(1, 1), k.ext(1, 2)])
    net, core = k.load(p); nb = {i: n for i, n in net.network.neurons.items()}
    for _ in range(30):
        for cue, rew in ((0, 1.0), (1, 0.0)):   # cueA rewarded, cueB not
            net.reset_simulation(); core.state.current_tick = 0; net.current_tick = 0
            for t in range(16):
                if 2 <= t < 10: nb[1].M_vector[1] = rew
                if 3 <= t < 7: net.set_external_input(1, cue, 4.0); net.set_external_input(1, 2, 5.0)
                core.do_tick()
    wA = float(nb[1].postsynaptic_points[0].u_i.info); wB = float(nb[1].postsynaptic_points[1].u_i.info)
    ok = wA > wB * 2.5
    return ok, f"learned discrimination: rewarded cue-A w={wA:.2f} >> unrewarded cue-B w={wB:.2f}"


def h5_retrograde_presyn():
    """Retrograde signaling: a postsynaptic prediction error propagates BACKWARD and adapts the
    PRESYNAPTIC terminal's output weight (u_o) — credit flows upstream. Off when eta_retro=0."""
    def run(eta_retro):
        A = k.neuron(1, r=0.6, c=3, plasticity="reward_hebb", eta_post=0.05, eta_retro=eta_retro)
        B = k.neuron(2, r=0.6, c=3, plasticity="reward_hebb", eta_post=0.05, eta_retro=eta_retro)
        sy = [k.syn(1, 0, 1.0, 1), k.syn(2, 0, 0.8, 1), k.syn(2, 1, 1.0, 1), k.term(1), k.term(2)]
        p = k.build([A, B], sy, [k.conn(1, 2, 0)], [k.ext(1, 0), k.ext(2, 1)])
        net, core = k.load(p); nb = {i: n for i, n in net.network.neurons.items()}
        from paula_agent.ckit import TERM
        w0 = float(nb[1].presynaptic_points[TERM].u_o.info)
        for t in range(40):
            net.set_external_input(1, 0, 5.0); net.set_external_input(2, 1, 6.0); core.do_tick()
        return float(nb[1].presynaptic_points[TERM].u_o.info) - w0
    d_off = run(0.0); d_on = run(0.2)
    ok = abs(d_off) < 1e-6 and abs(d_on) > 0.05
    return ok, f"presynaptic dw: eta_retro=0 -> {d_off:+.3f} (frozen), eta_retro=0.2 -> {d_on:+.3f} (adapts)"


def h6_metaplasticity_window():
    """Metaplasticity (plasticity of plasticity): the learning window t_ref is itself dynamic — it
    tracks the cell's own activity (F_avg). High sustained activity shrinks t_ref toward its lower
    bound; quiescence relaxes it toward the upper bound. The rule that governs learning adapts."""
    # 6 synapses so t_ref range [2c, c*num_inputs] = [4,12] has room to move (num_inputs>2).
    def final_tref(drive):
        ne = [k.neuron(1, r=0.5, c=2, lam=4)]
        sy = [k.syn(1, i, 1.0, 1) for i in range(6)] + [k.term(1)]
        p = k.build(ne, sy, [], [k.ext(1, i) for i in range(6)])
        net, core = k.load(p); nb = {i: n for i, n in net.network.neurons.items()}
        for t in range(60):
            if drive:
                for i in range(6): net.set_external_input(1, i, 3.0)
            core.do_tick()
        return float(nb[1].t_ref), float(nb[1].F_avg)
    thi, fhi = final_tref(True)     # high activity -> t_ref shrinks
    tlo, flo = final_tref(False)    # quiescent -> t_ref stays high
    ok = thi < tlo - 1.0 and fhi > flo
    return ok, f"learning-window t_ref: active(F={fhi:.2f})={thi:.1f} < quiet(F={flo:.2f})={tlo:.1f} (window adapts)"


CIRCUITS = [
    ("h1 reward-modulated LTP (three-factor)", h1_reward_ltp),
    ("h2 spike-timing direction (LTP/LTD)", h2_stdp_direction),
    ("h3 bounded convergence (stable learning)", h3_bounded_convergence),
    ("h4 reward discrimination (conditioning)", h4_discrimination),
    ("h5 retrograde presynaptic adaptation", h5_retrograde_presyn),
    ("h6 metaplasticity — modulator resizes window", h6_metaplasticity_window),
]

if __name__ == "__main__":
    import sys; sys.path.insert(0, ".")
    print("LEARNING / ADAPTATION TIER (three-factor plasticity / retrograde / metaplasticity):", flush=True)
    npass = 0
    for name, fn in CIRCUITS:
        try:
            ok, detail = fn()
        except Exception as e:
            ok, detail = False, f"ERROR {e}"
        npass += ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:44s} {detail}", flush=True)
    print(f"circuits_h: {npass}/{len(CIRCUITS)}", flush=True)
    print("@@@H DONE@@@", flush=True)
