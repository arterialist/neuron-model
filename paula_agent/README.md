# PAULA Agent Circuit Library

A library of **70 hand-designed, behavior-verified circuits** built on the PAULA neuron model,
composed into **5 working agents** — an entire agent nervous system / cognitive architecture,
treating PAULA as a computational platform (**not only spikes**: graded/analog membrane computation,
dendritic integration, multi-dimensional neuromodulation, and online plasticity).

**Verify everything in one command:**
```bash
PYTHONPATH=. python paula_agent/run_all.py
# LIBRARY 70/70 circuits · AGENTS 5/5 · TOTAL 75/75
```

Every circuit is a real PAULA network (neurons, dendritic synapses with delays, connections,
neuromodulators) run in the simulator with an automated test asserting its intended input→output
behavior — not a plausibility claim.

## Layout

| file | contents |
|---|---|
| `ckit.py` | construction kit: `neuron / syn / term / conn / ext / build / simulate` helpers |
| `circuits_a.py` | **Sensory + memory + decision (10)** — Reichardt motion, looming, Jeffress localizer, edge, novelty; WM-latch, delay-buffer; WTA, evidence-accumulator, two-choice |
| `circuits_b.py` | **Motor + neuromod + navigation (11)** — CPG (synfire ring), motor-sequence, reflex arc, P-controller; attention-router, homeostatic-drive, arousal, associative-conditioning; heading/path integrators, place-cell |
| `circuits_c.py` | **Arbitration + composite (4)** — subsumption arbiter, startle, gain-field, composite agent |
| `arch_d.py` | **Architectural core (6)** — predictive-coding; **supervisors** (inactivity / homeostatic / coherency); temporal-attention chunking; dopamine reward-strengthening |
| `arch_e.py` | **Connectors / self-mod / memory (14)** — cross-modal binder, gated relay, modality core; Hebbian growth, pruning, role-specialization; multi-slot WM, episodic replay, goal maintenance, value, recursion loop, resource-economy (fatigue), entrainment, curiosity |
| `arch_f.py` | **Multi-core tier (4)** — cross-modal binding across cores, re-entrant internal-environment loop, global-workspace broadcast, cross-core conflict resolution |
| `logic.py` | **Spiking logic (8)** — AND/OR/NOT/NAND/NOR/XOR/XNOR/IMPLY (threshold logic, De Morgan) |
| `circuits_g.py` | **Non-spike computation (7)** — analog weighted-sum, tunable low-pass filter, leaky integrator, signed subtraction, dendritic FIR temporal filter, neuromodulatory gain, analog derivative (all read graded `S`, high threshold so the cell never fires) |
| `circuits_h.py` | **Learning / adaptation (6)** — reward-modulated LTP, spike-timing LTP/LTD, bounded convergence, reward discrimination, retrograde presynaptic adaptation, activity-dependent metaplasticity |
| `agent_complex.py` | **complex integrated agent** — analog value core + three-factor taste-aversion learning + predator reflex + homeostatic foraging; out-survives a no-learning twin |
| `agent_sim.py` | behaving agent (forage + flee; causal control) |
| `agent_learn.py` | learns cue→outcome associations online (reward-modulated plasticity) |
| `agent_full.py` | integrated learning-survival agent (learns to avoid danger over its lifetime) |
| `agent_2d.py` | 2D spatial navigation agent (WTA direction selection + escape) |
| `run_all.py` | master regression runner |

Maps onto the target system architecture (`../../system-requirements,-architectural-framework,-and-high-level-notes.md`):
interface/sensors → temporal attention → modality cores + connectors → memory/prediction → decision →
motor, with **supervisors** and a **neuromodulatory economy** as the regulatory overlay, and a
**recursive internal environment** (re-entrant loop / global workspace).

## Design primitives

- **Dendritic delay** — a synapse's `distance_to_hillock` is an exact propagation delay (soma peaks
  at `tx + distance`); the basis for coincidence detection, motion, sequences.
- **Threshold coincidence / logic** — a neuron fires on a specific count of coincident inputs.
- **Inhibitory motifs** — WTA, lateral inhibition, subsumption, oscillators.
- **Neuromodulatory gating** — per-neuron `w_r` sensitivity lets a modulator open/close a neuron;
  targetable to specific cells (routing, attention, arousal).
- **Three-factor reward plasticity** (`reward_hebb`) — dopamine/stress-gated Hebbian learning.

## Calibration lessons (why circuits work)

- With **sustained** drive, membrane S → `w·HI`, so an **AND** needs threshold *between* one-input
  and two-input levels (~6, not ~2).
- Internal **spike-train** signals are weaker than sustained external lines → boost internal weights,
  use strong inhibition (−10…−16) for gating/inverting.
- **Plasticity is timing-critical**: `reward_hebb` LTP requires the target to fire *causally*
  (dir=+1) during/after the potentiated input; an extra hop that delays the input past firing
  inverts the sign (→ LTD). A **teaching/US signal** that fires the target causally fixes it.
- **De Morgan** for robust inverting logic: `NAND = OR(NOT a, NOT b)`, `XNOR = OR(AND, NOR)`.
- **Delayed inhibition** for oscillators / novelty / adaptation.
- A leaky **integrator** converges to the *average* input (it can't accumulate unboundedly); an
  "accumulate-then-fire" delay comes from the ramp time, and sustaining an OFF state needs a
  **bistable latch**, not a plain integrator (see `resource-economy`).
- Use a **uniform terminal id** (kit default 900) so presynaptic terminals don't collide with
  postsynaptic synapse ids in the neuron's `distances` dict.
- **No per-synapse credit assignment.** Plasticity uses a **neuron-global** signal — the reward gate
  (`nm` from `M_vector`) and the causal/acausal `direction` come from the *neuron's* firing, and every
  active synapse gets that same signal (scaled only by its own input and current weight). You cannot
  potentiate one synapse and depress another in the *same* activation. Discriminate inputs by
  **separating them in time or reward-context** (activate cue A in reward trials, cue B in others) —
  never by expecting the rule to credit the "responsible" synapse. (Only *active* synapses update.)
- **Analog regime**: set the threshold very high (`r≈1e6`) so the cell never fires and `S` is a
  continuous computed value — a leaky integrator with time-constant `λ`, steady state `Σ wᵢxᵢ`.

## Scope

Hand-designed (not learned) circuits; several are neuromorphic classics (Reichardt 1956, Jeffress
1948, Marder half-centers, drift-diffusion) realized in this substrate. The point is the **platform**:
sensing, memory, decision, action, motivation, learning, and self-regulation all composable in one
bio-plausible neuron model, and composing into agents that behave and learn online with no gradient.
