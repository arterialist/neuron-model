"""NON-SPIKE COMPUTATION tier — PAULA is not only about spikes. With a high firing threshold a PAULA
neuron never fires and its membrane S becomes a continuous ANALOG variable: S is a leaky integrator of
the dendritic current, dS=(1/lambda)(-S+I_t), steady-state S=sum_i w_i x_i, time-constant lambda, with
each synapse's contribution delayed by its distance_to_hillock and attenuated by delta_decay^distance.
That yields a family of graded/dendritic/neuromodulatory primitives that spike-timing circuits ignore:
linear combiners, tunable filters, integrators, analog subtraction, dendritic FIR kernels, a derivative
(high-pass), and neuromodulatory gain control. Each verified by reading graded S (or firing rate)."""
import numpy as np
from paula_agent import ckit as k

HI = 1e6  # firing threshold so high the neuron is a pure analog unit (never spikes)


def g1_weighted_sum():
    """Analog linear combiner: S -> w0*x0 + w1*x1 (exact additive superposition)."""
    n = [k.neuron(1, r=HI, lam=5, delta_decay=1.0)]
    sy = [k.syn(1, 0, 1.5, 1), k.syn(1, 1, 0.5, 1), k.term(1)]
    p = k.build(n, sy, [], [k.ext(1, 0), k.ext(1, 1)])
    a = k.simulate_S(p, 40, hold=lambda t: [(1, 0, 2.0)], probe_ids=[1])[1][-1]
    b = k.simulate_S(p, 40, hold=lambda t: [(1, 1, 4.0)], probe_ids=[1])[1][-1]
    ab = k.simulate_S(p, 40, hold=lambda t: [(1, 0, 2.0), (1, 1, 4.0)], probe_ids=[1])[1][-1]
    # a~1.5*2=3, b~0.5*4=2, ab~5 (superposition)
    ok = abs(a - 3.0) < 0.2 and abs(b - 2.0) < 0.2 and abs(ab - (a + b)) < 0.15
    return ok, f"S(x0)={a:.2f} S(x1)={b:.2f} S(both)={ab:.2f}~sum={a+b:.2f}"


def g2_lowpass():
    """Tunable low-pass filter: the 63% rise-time equals lambda (larger lambda = smoother/slower)."""
    def tau(lam):
        n = [k.neuron(1, r=HI, lam=lam, delta_decay=1.0)]
        p = k.build(n, [k.syn(1, 0, 1.0, 1), k.term(1)], [], [k.ext(1, 0)])
        S = k.simulate_S(p, 6 * lam, hold=lambda t: [(1, 0, 3.0)], probe_ids=[1])[1]
        ss = S[-1]
        return next((i for i, v in enumerate(S) if v >= 0.63 * ss), -1)
    t1, t2 = tau(3), tau(30)
    ok = 2 <= t1 <= 4 and 27 <= t2 <= 33
    return ok, f"tau(lambda=3)={t1}  tau(lambda=30)={t2}  (tau==lambda)"


def g3_integrator():
    """Leaky integrator/accumulator: brief pulses ACCUMULATE in S, then S decays when input stops."""
    n = [k.neuron(1, r=HI, lam=40, delta_decay=1.0)]
    p = k.build(n, [k.syn(1, 0, 1.0, 1), k.term(1)], [], [k.ext(1, 0)])
    # 5 short pulses (2 ticks on / 3 off), then silence
    def drv(t):
        return [(1, 0, 6.0)] if (t % 5 < 2 and t < 25) else []
    S = k.simulate_S(p, 60, hold=drv, probe_ids=[1])[1]
    after1 = S[4]; peak = max(S[:25]); tail = S[-1]
    ok = peak > after1 * 2 and tail < peak * 0.6  # accumulates above one pulse, then decays
    return ok, f"S after 1 pulse={after1:.2f} -> peak(accumulated)={peak:.2f} -> decayed tail={tail:.2f}"


def g4_difference():
    """Analog subtraction: excitatory synapse minus inhibitory synapse -> S = x_exc - x_inh (signed)."""
    n = [k.neuron(1, r=HI, lam=5, delta_decay=1.0)]
    sy = [k.syn(1, 0, 1.0, 1), k.syn(1, 1, -1.0, 1), k.term(1)]
    p = k.build(n, sy, [], [k.ext(1, 0), k.ext(1, 1)])
    hi = k.simulate_S(p, 40, hold=lambda t: [(1, 0, 5.0), (1, 1, 2.0)], probe_ids=[1])[1][-1]
    lo = k.simulate_S(p, 40, hold=lambda t: [(1, 0, 2.0), (1, 1, 5.0)], probe_ids=[1])[1][-1]
    ok = abs(hi - 3.0) < 0.3 and abs(lo + 3.0) < 0.3  # +3 and -3
    return ok, f"S(5-2)={hi:.2f}  S(2-5)={lo:.2f}  (graded signed difference)"


def g5_dendritic_fir():
    """Dendritic FIR temporal filter: one impulse fed to 3 synapses at distances 1/6/12 produces three
    delayed graded bumps in S at those lags -> a programmable temporal convolution kernel."""
    n = [k.neuron(1, r=HI, lam=2, delta_decay=1.0)]
    sy = [k.syn(1, 0, 1.0, 1), k.syn(1, 1, 1.0, 6), k.syn(1, 2, 1.0, 12), k.term(1)]
    p = k.build(n, sy, [], [k.ext(1, 0), k.ext(1, 1), k.ext(1, 2)])
    # single-tick impulse at t=1 on all three synapses
    S = k.simulate_S(p, 30, drives={1: [(1, 0, 8.0), (1, 1, 8.0), (1, 2, 8.0)]}, probe_ids=[1])[1]
    # find local maxima; expect bumps near t=1+1, 1+6, 1+12
    peaks = [i for i in range(1, len(S) - 1) if S[i] > S[i - 1] and S[i] >= S[i + 1] and S[i] > 0.3]
    near = lambda lag: any(abs(pp - lag) <= 2 for pp in peaks)
    ok = near(2) and near(7) and near(13)
    return ok, f"impulse-response peaks at ticks {peaks} (taps expected ~2,7,13)"


def g6_neuromod_gain():
    """Neuromodulatory gain control: a modulator M raises the firing threshold (w_r*M) so the SAME input
    yields a graded, monotonically DECREASING firing rate -> analog gain/volume knob via neuromodulation."""
    # w_r[0]=+0.5 : M[0] raises r -> harder to fire -> lower rate
    def rate(m):
        n = [k.neuron(1, r=0.6, lam=6, c=2, w_r=[0.5, 0.0])]
        p = k.build(n, [k.syn(1, 0, 1.0, 1), k.term(1)], [], [k.ext(1, 0)])
        out = k.simulate(p, 40, drives={t: [(1, 0, 3.0)] for t in range(40)},
                         mod=lambda t, nb: nb[1].M_vector.__setitem__(0, m), probe_ids=[1])
        return sum(out[1])
    r0, r1, r2 = rate(0.0), rate(2.0), rate(5.0)
    ok = r0 > r1 > r2 and r2 <= r0  # monotone gate down
    return ok, f"firing rate: M=0 ->{r0}, M=2 ->{r1}, M=5 ->{r2} (modulator dials gain down)"


def g7_derivative():
    """Analog high-pass / derivative: S_fast - S_slow (fast vs slow leaky integrator on the same input)
    responds to CHANGES, not steady level -> a graded edge/onset detector in the analog domain."""
    nf = [k.neuron(1, r=HI, lam=2, delta_decay=1.0)]
    ns = [k.neuron(2, r=HI, lam=30, delta_decay=1.0)]
    pf = k.build(nf, [k.syn(1, 0, 1.0, 1), k.term(1)], [], [k.ext(1, 0)])
    ps = k.build(ns, [k.syn(2, 0, 1.0, 1), k.term(2)], [], [k.ext(2, 0)])
    step = lambda t: [(1, 0, 5.0)] if t >= 5 else []
    steps = lambda t: [(2, 0, 5.0)] if t >= 5 else []
    Sf = k.simulate_S(pf, 60, hold=step, probe_ids=[1])[1]
    Ss = k.simulate_S(ps, 60, hold=steps, probe_ids=[2])[2]
    diff = [f - s for f, s in zip(Sf, Ss)]
    peak_i = int(np.argmax(diff))
    ok = max(diff) > 1.5 and diff[peak_i] > diff[-1] * 3 and 5 <= peak_i <= 20  # transient at the edge, decays
    return ok, f"onset transient peak={max(diff):.2f} at t={peak_i}, settles to {diff[-1]:.2f} (change-detector)"


CIRCUITS = [
    ("g1 weighted-sum (linear combiner)", g1_weighted_sum),
    ("g2 tunable low-pass filter", g2_lowpass),
    ("g3 leaky integrator/accumulator", g3_integrator),
    ("g4 analog difference (excit-inhib)", g4_difference),
    ("g5 dendritic FIR temporal filter", g5_dendritic_fir),
    ("g6 neuromodulatory gain control", g6_neuromod_gain),
    ("g7 analog derivative (change detector)", g7_derivative),
]

if __name__ == "__main__":
    import sys; sys.path.insert(0, ".")
    print("NON-SPIKE COMPUTATION TIER (graded S / dendritic / neuromodulatory):", flush=True)
    npass = 0
    for name, fn in CIRCUITS:
        ok, detail = fn()
        npass += ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:40s} {detail}", flush=True)
    print(f"circuits_g: {npass}/{len(CIRCUITS)}", flush=True)
    print("@@@G DONE@@@", flush=True)
