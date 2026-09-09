# Foveated PAULA/ALERM perception — research log

A fixed-resolution retina (k×k) resampled from a larger image at a movable
position, with an ALERM supervisor injecting m0/m1 to destabilize (saccade) or
crystallize (fixate) the eye's motion. Klinokinesis (C. elegans run/pirouette)
ported to visual foveation. Teacher-present: the teacher knows the class and
injects m0/m1; the perception substrate only ever sees the scalar modulators
("am I doing the right thing?"), never the class — so no class leakage.

Ablation ladder: P0 static crop · P1 saliency policy · P2a diffusion · P2b
klinotaxis · P3 teacher-present decoder loop. Storage budget ~22 GB, so no
large [T,N] recording: experiments compute metrics in-process and save compact
JSON only.

---

## Design decisions (settled with user)

- **Fixed retina, moving eye.** Perception net is built at retina resolution
  `(C, k, k)` and the image is *resampled* into it. Never crop-feed a full
  32×32 network (that leaves 3/4 of its inputs permanently silent).
- **Supervisor = the regulating layer** above perception; folds in the drive.
  Reads perception's free-energy proxy, injects m0/m1 down (volume
  transmission), emits the motor command.
- **Motor = population code**, not scalar S. Velocity from opponent pairs
  `vx=g(S_r−S_l), vy=g(S_d−S_u)` or a population vector. m0 sets exploration
  *temperature* (Langevin), drift sources the *direction*.
- **Decoder reads the continuous stream** (no frame splicing — that breaks the
  stateful LIF). "Settled" = adaptive readout time and/or m0/m1 as extra
  decoder channels.
- **Object-selectivity is learned, not intrinsic.** The dark-room problem:
  naive FE-minimization fixates on the flattest patch. Reward must be
  *structured* stability = settledness × informativeness; and the reliable
  object signal comes from the teacher/decoder, since "object" ≝ "region from
  which the class is decodable."

---

## EXP1 — reward–saliency correlation (the crux: does label-free structured
## stability track the object?)

**Method.** Slide the fovea over a position grid; at each position reset the
net, hold the patch, run a settle window, record reward = settledness ×
informativeness. Correlate the reward map with image edge-energy saliency and a
center prior. Untrained perception net (125 neurons: conv k4s2 f2 → k3s2 f3
over 1×16×16). CIFAR-10 grayscale.

**Findings (chronological — includes a corrected false positive).**

| run | images | settle | gain | reward~saliency | note |
|-----|-------:|-------:|-----:|----------------:|------|
| smoke | 2 | 6 | 0.1 | **+0.87** | FALSE POSITIVE (n=2 fluke) |
| main | 12 | 15 | 0.1 | +0.04 (±0.59) | ≈ chance; 50% images positive |
| settle sweep | 8 | 2–40 | 0.1 | −0.05…−0.07 | flat/negative |
| **gain sweep** | 8 | 8 | 0.03–0.3 | +0.20 | **participation = 0.000 (net SILENT)** |
| gain sweep | 8 | 8 | 3.0 | +0.25 | participation 0.149 (spiking regime) |
| settle sweep | 10 | 4–60 | **3.0** | **+0.11…+0.13** | stable, correct regime |

**Two bugs found by measuring instead of assuming:**
1. **n=2 gave +0.87; n≥10 gives ~+0.12.** The first "great" result was sampling
   luck. Always ≥10 images.
2. **At gain 0.1 the fovea net never spikes** (participation 0.000). Every early
   run measured sub-threshold membrane noise, not dynamics. The net enters its
   sparse spiking regime (~15% active, matching the paper's 5–10%) only around
   gain 3.0. **Regime must be set by participation, not by the drive/threshold
   ratio calibrator alone.**

**Result in the correct regime (gain 3.0, n=10):**
- reward ~ saliency: **+0.11 to +0.13** (weakly positive, stable across settle
  time; decays to +0.075 by 60 ticks).
- The reward is **informativeness-dominated** (info~saliency ≈ reward~saliency
  to 3 decimals). Informativeness = std(S) has a mild contrast bias.
- **settledness ~ saliency is noise** (−0.32…+0.12, no trend). Settling into an
  attractor carries *no* object signal on an untrained net — expected, since
  class-specific attractors are a product of local learning over exposures,
  which a cold 60-tick run has not produced.

**Interpretation.** H1 (label-free structured stability is object-selective) is
**weakly supported at best (~+0.12), and only via a contrast bias, not via
attractor settling.** This vindicates the user's objection: nothing in the
untrained substrate strongly selects the object. +0.12 is the **baseline any
learned policy must beat** — and it means P2a/P2b (label-free rungs) will be
weak, and the object-selectivity has to come from a **trained perception net
and/or the teacher-present decoder loop (P3)**. This is exactly the design the
user chose, so the ladder proceeds with P3 as the load-bearing rung.

---

## Decisions / next steps

1. **Regime gate first, always.** Any perception net must be run at a gain that
   yields ~5–15% participation before any correlation is meaningful. Add a
   participation check to the experiment harness (currently manual).
2. **Trained-net re-test of EXP1.** Repeat the scan with a perception net whose
   local plasticity has been exposed to CIFAR patches (does settledness become
   object-selective once attractors form?). This isolates "intrinsic dynamics"
   from "learned attractors." Cheap-ish; do before building P3.
3. **P2a diffusion rollout** as the control: expect near-null object-seeking
   (baseline ≈ +0.12), quantify it, don't expect it to win.
4. **P3 teacher-present loop** is where object-selectivity should appear;
   bootstrap: diffusion → collect small foveated set → train decoder v0 →
   teacher (m1 if class-correct) shapes fixations → re-collect. Measure per-class
   fixation-on-object and accuracy vs P2a. The P2a→P3 gap = how much
   object-seeking is *learned from the task* vs intrinsic.
5. **Habituation** (user's attention insight): once P3 exists, add a novelty/
   habituation term so a fixated percept, once predicted, releases the eye —
   turning single fixations into scanpaths.

## Harness (implemented, tested)

`foveation/`: `fovea.py` (retina + IOR), `perception.py` (fovea-sized net +
per-tick state + m0/m1 volume broadcast), `signals.py` (free-energy probe +
stability→m0/m1), `saliency.py` (edge-energy + partial-correlation metrics),
`experiment_scan.py` (EXP1), `experiment_settle_sweep.py` (EXP1b). Results in
`foveation_results/` (compact JSON).

---

## Wave 2 — EXP1c, EXP2-deep, EXP4 glow, animations

### EXP1c — local adaptation did not confer object-selective settling
Short unsupervised adaptation (3 epochs) grew mean efficacy +623% (the
multiplicative variance amplifier) but reward~saliency barely moved
(+0.081→+0.095); settledness stayed negative. The 6× weight growth shifts the
drive regime (net saturates at fixed gain). Conclusion: crude adaptation at
fixed gain doesn't help; regime must be re-calibrated as weights grow, and/or
learning must run with homeostatic gain control maintained. Weights per se are
not the lever (user's point) — regime is.

### EXP4 — retinotopic glow map: the big positive
Scan fovea, paint perception activity back to image space. In the GRADED regime
(gain 1.0, 10px fovea, single conv), the **informativeness glow traces object
contours**: info-glow ~ edge = **+0.60 mean, up to +0.96** on clean images;
firing-rate glow +0.34. Visually the object lights up and background goes black,
with no labels — a V1-like, label-free figure/ground signal.

**This overturns the earlier "label-free signal is only +0.12" conclusion.** That
+0.12 came from (a) the clock-saturated gain-3 regime, (b) a 16px fovea, and
(c) multiplying informativeness by *settledness* (which is noise) — the product
diluted the strong informativeness signal. Corrected recipe: **graded regime
(gain ~1) + small fovea + informativeness alone.**

### EXP2-deep — the settled regime IS input-distinct; EXP2 measured it wrong
LOO-NN label separability (chance 0.20), fixed center fovea, per-neuron vectors:

| config | mean-S | firing-rate | t_ref |
|--------|-------:|------------:|------:|
| gain1 grayscale | 0.07 | 0.20 | 0.30 |
| gain3 grayscale | 0.08 | **0.40** | 0.32 |
| gain1 RGB       | 0.08 | **0.40** | 0.28 |

Findings:
- **mean-S is ≤ chance everywhere** — my original EXP2 signature was S-heavy, so
  it hid the signal. Region-distinctness ~0 for the same reason (mean vector),
  while the glow (variance) shows huge object/background contrast.
- **firing-rate separates at 2× chance (0.40)** with RGB or gain-3; grayscale +
  gain-1 is the weakest corner (0.20) — so **grayscale does dilute the class
  code; color helps** (user's hypothesis confirmed).
- **t_ref ~0.30 above chance in every regime** — the homeostatic window is the
  most reliably input-distinct scalar feature.
- **~21% of neurons are individually label-selective** (ANOVA) — not input-
  invariant.
- **Separability is flat over 100→1000 ticks** — the input-specific regime forms
  by ~t100 and holds; longer settling adds no class info. Spiking is clock-like
  (ISI CV ~0.001) in all regimes at ~8% participation.
- Attractor-space check: 5 labels settle at distinct <S> (2.6/3.5/5.0/11.4/11.8);
  <t_ref> pinned near its ceiling (~119) at this participation.

### Animations (mp4, ffmpeg)
- `anim/glow_anim_*.mp4` — 5 panels: input+fovea box · edge · info-glow
  accumulating · rate-glow accumulating · info-glow overlaid on input. The
  object emerges from black as the fovea scans.
- `anim/attractor_settling.mp4` — spike raster + live (<S>,<t_ref>) phase-plane
  trajectories, one per label, tracing input-specific paths.

## PIVOT — the informativeness glow is the fovea drive
The label-free object signal we couldn't find in "settledness" is sitting in
**informativeness** (spatial spread of perception activity), and it is strong
(+0.6..+0.96) in the graded regime. This reframes the ladder:
- **Drive = climb the informativeness glow**; lock in (fixate) where glow is
  high. This is a concrete, strong, label-free klinotaxis signal — P2b's drift
  term is `∇(informativeness)`, estimable through small test movements.
- The teacher (P3) then only has to *refine* fixation for class-decodability on
  cluttered cases, on top of a drive that already finds figure/ground for free.
- Open question to nail next: does the glow localize the *object* or merely
  *high-contrast texture*? Needs the cluttered-background classes (frog/deer/
  bird on foliage) scored explicitly.

### EXP10 — glow: object or texture? (mask-free, per-class)
Discriminators split clean {plane,car,ship,truck} vs cluttered {bird,cat,deer,
dog,frog,horse}, gain 1, 10px fovea, n=4/class:

- glow~edge: highly class-variable (car +0.72, dog +0.73, but deer −0.02,
  horse −0.12); clean +0.45 vs cluttered +0.36.
- glow centre-of-mass ≈ edge centre-of-mass (~0.07), near image centre for BOTH
  groups — glow sits where edges are, and CIFAR edges are centre-biased.
- glow Gini (0.29) ≈ footprint-blurred-edge Gini (0.29-0.33): glow is NOT more
  concentrated than a blurred edge map — it does not select a subset of edges.
- centre_frac ~0.25 = uniform (centre gets only its area-share of glow).
- **No clean-vs-cluttered difference on any structural metric.**

**Read: the glow is a bottom-up edge/contrast (saliency) map blurred by the
fovea footprint — NOT an object detector.** The earlier r=+0.96 "object glows"
cases were clean backgrounds where the object simply IS the dominant edge
structure. On cluttered images the background texture lights up too. This
confirms the object-vs-texture worry.

**Reframe (not a failure):** glow = bottom-up saliency (V1 / superior colliculus
first-pass), the teacher/decoder = top-down object identity. Standard two-stage
visual attention: glow proposes contrast hotspots as saccade targets; the
teacher (P3) says "this is the object, lock in" vs "keep looking". Matches the
teacher-present design. So P2b rides the glow gradient for exploration; P3
supplies object-selectivity.

Caveats: n=4/class is noisy (deer/horse negatives may be noise); no true object
masks (COM/Gini/centre are proxies); CIFAR at 32×32 may be too low-res for a
sharp object/texture split — STL-10 (96×96) or masks would test this properly.

---

## Wave 3 — the CIFAR "cracking" question: does scaling PAULA help?

User's gating condition: don't move on from CIFAR unless it's TRULY out of reach
for a small bio-plausible net OR a **big network of PAULA** (untested). Also:
"the signal slowly fades as it propagates through layers." Two experiments.

### Reference ceiling (conventional, for context only)
kNN/logreg/CNN on CIFAR-10, 45 epochs. Grayscale: pixel-kNN 0.223, logistic
0.237, tiny-CNN(67k) 0.60, small-CNN(281k) 0.643. RGB: 0.258 / 0.321 / 0.617 /
0.692. So a 281k-param CNN caps ~64% grayscale / 69% RGB — color is worth ~5pts,
and 10→45 epochs barely moved it. (User: conventional-model size is beside the
point; kept only as a yardstick.)

> **CORRECTION (directive #1):** the EXP11/EXP12 probes were NOT frozen. Local
> plasticity was ON throughout (eta_post/eta_retro nonzero; reset preserves
> efficacies), so the substrate was doing UNSUPERVISED online learning across the
> image sequence — just with no teacher/task signal. So the 0.21 pixel-kNN
> ceiling holds *despite* unsupervised plasticity. Read "untrained/frozen" below
> as "unsupervised-plasticity-on, no teacher." This strengthens the conclusion:
> unsupervised local learning alone does not build class structure — the missing
> ingredient is the TEACHER-shaped reward (P3), not "turn learning on."

### EXP11 — PAULA substrate capacity + the depth-fade (unsupervised plasticity on)
Full-image PAULA nets, escalating width/depth, EACH held in its spiking regime
by **per-layer firing-threshold homeostasis** (r_base auto-tuned per layer to
hit ~0.1 participation — the principled version of the user's "careful threshold
tuning"). Read settled activity [rate_all|rate_late|S_late|t_ref_late], decode
kNN(cosine)+logreg over PCA-80. Grayscale, 15 train / 8 test per class (chance
0.10, ±0.033). Readout = ALL neurons and LAST layer only.

| arch | layers | neurons | kNN (all) | kNN (last layer) |
|------|-------:|--------:|----------:|-----------------:|
| w2 | 1 | 450  | 0.212 | 0.212 |
| w4 | 1 | 900  | 0.212 | 0.212 |
| w8 | 1 | 1800 | 0.200 | 0.200 |
| d1 | 1 | 900  | 0.212 | 0.212 |
| d2 | 2 | 1194 | 0.113 | 0.100 |
| d3 | 3 | 1394 | 0.100 | 0.113 |

Three findings:
1. **Width does nothing.** 450→1800 neurons, decodability flat ~0.21. Untrained
   extra filters are redundant random projections of the same input; PCA folds
   them together.
2. **Single-layer ceiling ≈ raw-pixel kNN.** 0.21 ≈ reference grayscale pixel-kNN
   (0.223). The untrained substrate is essentially a spiking re-encoding of the
   downsampled image — it keeps pixel-level class info, nothing more.
3. **Depth fades the signal to chance, and threshold homeostasis does NOT fix
   it.** 2–3 layers → last-layer decodability = chance (0.10). Homeostasis
   restored FIRING (every layer ~0.085 participation) but the deep layers fire
   NOISE. Confirms the user's warning exactly: equalizing participation is
   necessary-but-not-sufficient — it fixes activity, not information.

Nuance/honesty: the all-neuron collapse for d2/d3 is partly probe DILUTION
(StandardScaler makes ~1000s of deep noise-dims equal-variance to layer-0 signal,
so PCA-80 wastes components) — layer 0 alone still has ~0.21. The load-bearing,
non-artifact claim is the LAST-LAYER readout = chance: **untrained deep conv
layers carry no class signal.** Caveats: untrained/frozen probe (a LEARNED deep
net could differ — the fade result is about RANDOM deep stacks); kNN/logreg on
settled activity is a lower bound vs the trained temporal decoder; grayscale
gain-1 is the weak corner (RGB gave 2× separability in EXP2-deep).

**Reframing.** The probe answers "does a bigger UNTRAINED substrate hold more
class info?" → No, not by width (redundant) or depth (fades). So the real
classifier's capacity comes from LEARNING (local plasticity shaping filters), not
substrate size. The gap to 95% is not "add neurons" territory. Levers that match
the finding: (a) skip/residual composition (builder has shortcut_specs) so the
pixel signal survives to deep layers; (b) local plasticity in deep layers WITH
per-layer homeostasis maintained, so deep layers build signal instead of
scrambling it. The true "big LEARNED micro-brain" test needs learning enabled +
re-probe — the frozen probe cannot see it.

Harness: `experiment_reference_ceiling.py`, `experiment_paula_capacity.py`
(per-layer participation, uniform vs homeostatic threshold, all/last readout).
perception.py gained `layer_of_pos`, `set_layer_threshold`, `scale_layer_threshold`.

### Delay mechanism (discovered while building EXP12)
Propagation delay currently lives in two places, neither a rich spatial embedding:
- **Axonal/network:** `delay = randint(MIN,MAX)` per signal with **MIN=MAX=1** →
  every inter-neuron hop is exactly 1 tick. Stored axon-terminal distances are
  IGNORED for timing.
- **Dendritic:** real & per-connection — a synapse reaches the hillock after
  `distance_to_hillock` ticks (2–8 as built) AND is attenuated by
  `delta_decay**distance` (0.96^d). So distance COUPLES delay + attenuation.
So "the spatial aspect" today = 2–8 ticks of dendritic delay with coupled decay,
1 tick everywhere else. EXP12 sweeps this.

### EXP12 — propagation-delay sweep (d2 arch, participation held by homeostasis)
871-neuron 2-layer net; each policy re-calibrated to ~0.1 participation so DELAY
is the only variable. Readouts: settled decodability, temporal (6-bin) decoda-
bility, participation-ratio dimensionality. Grayscale, 15/8 per class (chance
0.10 ±0.033).

| policy | dendrite | dd | axon | kNN set | kNN tmp | PR |
|--------|----------|---:|------|--------:|--------:|---:|
| min1 | =1 | 0.96 | 1 | 0.163 | 0.138 | 7.0 |
| current | 2–8 | 0.96 | 1 | 0.100 | 0.113 | 7.2 |
| const8 | =8 | 0.96 | 1 | 0.100 | 0.113 | 7.0 |
| wide16 | 1–16 | 0.96 | 1 | 0.138 | 0.163 | 7.0 |
| wide16_nodecay | 1–16 | 1.0 | 1 | 0.150 | 0.113 | 6.8 |
| axon1to10 | 2–8 | 0.96 | 1–10 | 0.163 | 0.113 | 7.1 |

**Result: no measurable effect.** Every cell is within ~2σ of chance and of
every other cell; PR ≈ 7 everywhere (a very low-dim, redundant code — consistent
with the clock-like regime of EXP2-deep). Delay magnitude (1→8→16), attenuation
on/off (dd 0.96 vs 1.0), and axonal jitter (1→10) all leave decodability and
dimensionality unchanged.

**Own the confound:** this was run on the d2 arch, which EXP11 already showed
FADES TO CHANCE. There is no live signal for delays to modulate here, so the null
is underpowered, not conclusive. The clean re-test = delay sweep on the
SINGLE-LAYER net (real ~0.21 signal) with more test samples, asking whether
delays change a representation that actually carries class info. Also: the sim
re-randomizes axonal delay per-signal, so it is temporal JITTER, not a fixed
spatial embedding — a true spatial-delay test needs code that reads the stored
per-connection distances deterministically (polychronization-style). Both are
follow-ups, not done.

Harness: `experiment_delays.py` (JSON delay rewrite + delta_decay override +
axonal constant patch; settled/temporal/PR readouts).

---

## Wave 4 — interactive tools + the slow (weight) state

Directives from the user: (#1) never freeze weights; (#2) an interactive viewer
(drag gaze, continuous run, all neurons individually + dynamics); (#3) small
fovea + blurred periphery; (#4) is PAULA just a high-dim input encoder or does it
form internal representations; (#5) also read out with the trained SNN decoder;
(#6) stay on CIFAR, STL only as smoke; (#7) build the mini-brain.

### Tools shipped
- `retina.py` — multi-resolution retina: small sharp fovea + blurred peripheral
  gist rendered concentric to `(2C, grid, grid)`.
- `viewer.py` — draggable-gaze, continuous-run viewer with per-neuron S grid,
  spike raster, and dynamics (participation/t_ref/efficacy). Learning ON. Has an
  Agg `--record` mode. 10 per-class demo clips in `anim/viewer_demos/`.
- CORRECTION logged: earlier probes were never frozen (plasticity was on) —
  see the correction box above.

### EXP13 — efficacy saturation is a familiarity FLAG, not a memory bank
User observed in the viewer: hold gaze → mean efficacy grows exp→plateau (~1k
ticks); move → grows again; return to a plateaued spot → flat. Reproducible.

Quantified (retina net, 196 neurons, learning on, schedule A,B,A,C,B):
- Reproduced the exp→plateau growth; saturation is FAST (~500–700 ticks) toward a
  shared hard ceiling (~9.98) on `u_i.info`.
- Growth is content-specific in DIRECTION: A vs B delta-efficacy vectors are
  anti-correlated (**corr −0.99** at hold 400; top-30% grown-synapse Jaccard
  0.23) — each patch preferentially grows a different subset FIRST.
- BUT the resampling retina fills the whole input grid regardless of gaze, so
  every input synapse is always driven → holding ANY single patch saturates
  essentially ALL synapses to the ceiling. So the "renewed growth on move" is the
  transient approach-to-ceiling being re-excited by content before full
  saturation; the "flat return" is simply "ceiling already reached."

**Read:** the phenomenon is real but it is a NOVELTY/FAMILIARITY flag, not a
location-addressable memory. Two architectural consequences (both feed the
mini-brain, #7):
1. **A resampling retina is incompatible with spatial memory.** To get
   content-addressable, location-specific traces, the eye must touch DIFFERENT
   substrate inputs at different gaze positions — a topographic/retinotopic input
   (full-image net, or a retina + explicit "where" tag), not a patch resampled to
   fill the grid.
2. **The multiplicative rule saturates globally to a common ceiling**, which
   erases content distinctions over time. A useful weight-memory needs a
   forgetting/decay term, input-dependent ceilings, or competitive normalization
   so synapses don't all max out.

Bearing on #4: the SLOW (weight) state DOES write content-dependent traces
(anti-correlated growth) — so PAULA is more than a stateless encoder — but under
the current unbounded-to-ceiling rule those traces collapse into a familiarity
flag. "Internal representation" in the strong sense needs competition/decay added.
Harness: `experiment_familiarity.py`, perception `efficacy_vector/efficacy_index`.

### EXP14 (#4/#5) — readout comparison (legacy rule)
`experiment_readout.py`: full-image conv layer (topographic), learning on, regime
via homeostasis; readouts = pixel-kNN, PAULA settled kNN/linear, PAULA temporal
**SNN decoder** (snntorch, MPS), SNN on time-SHUFFLED streams, and (added) the
WEIGHT vector. Legacy rule, grayscale, 40/20 per class (chance 0.10):

| readout | acc |
|---|---|
| pixel-kNN baseline | 0.230 |
| PAULA settled kNN | 0.225 |
| PAULA settled linear | 0.230 |
| PAULA temporal SNN (trained, MPS) | 0.230 |
| PAULA temporal SNN (time-shuffled) | 0.175 |

**Verdict (legacy rule):** even the trained temporal SNN decoder cannot exceed the
pixel-kNN ceiling → PAULA is confirmed a high-dim PIXEL-LEVEL encoder, no more.
The shuffle control (0.230 vs 0.175) shows temporal order carries a little
structure, but it nets to pixel level. This is the direct #4 answer under legacy.

---

## Wave 5 — fixing the plasticity rule (multiplicative → error-correcting)

Root cause of EXP13's content-independent saturation (and the C. elegans
"saturates critical pathways" freeze): the legacy update
`Δw = eta·dir·|E|·w·(1−w/10) − eta·0.02·w` keeps the error's MAGNITUDE but drops
its SIGN, and multiplies by `w`, so weights grow past their target
(`info_val≈[0,2]`) to the ceiling ~10, content-independently.

**Fix implemented (opt-in, gated; legacy stays default).** neuron.py gains params
`plasticity_mode`, `lr_error`, `weight_decay_tau`, `weight_baseline`.
- `error_correcting`: `Δw = lr·dir·(info_val − w)` — additive, SIGNED error toward
  the input signal. Equilibrium `w→info_val` ⇒ content-specific, bounded.
- per-tick passive decay pass (all synapses, every tick) for genuine forgetting.
- perception `set_plasticity(...)` selects it at runtime (no JSON change).

**Conceptual resolution (weight vs u_info).** `info_val` = the SIGNAL at the
synapse; `u_i.info` = the WEIGHT (a gain) used as multiplicative throughput in
the forward pass `V_local = info_val·(u_i.info+u_i.plast)`. "Weight is
multiplicative throughput" (forward) and "update is multiplicative" (learning)
are INDEPENDENT — keep the former, make the latter additive/error-correcting.
(`u_i.plast` is a fixed baseline gain, never updated.)

**EXP13 re-run, error_correcting (retina net):** per-patch final efficacy
A/B/C = 1.08 / 1.21 / 1.07 (DISTINCT, content-specific) vs legacy's uniform
9.978. Weights now settle at reproducible per-patch values and restore on return.
Because the resampling retina drives every synapse each view, it behaves as a
WORKING memory (tracks current input, small overwrite on move) — persistent
location memory still needs topographic input. Plot exp13_errorcorrecting.png.

### EXP14b (#4/#5 payoff) — readout under error_correcting + decay
Three arms on the topographic full-image net (grayscale, 40/20 per class, n=200
test so σ≈0.03):

| arm | pixel | setKNN | setLIN | tempSNN | SNNshuf | wKNN | wLIN |
|---|---|---|---|---|---|---|---|
| legacy | 0.230 | 0.225 | 0.230 | 0.230 | 0.175 | **0.300** | 0.235 |
| error_correcting | 0.230 | 0.255 | 0.200 | 0.245 | 0.205 | 0.220 | 0.175 |
| error_correcting+decay(τ200) | 0.230 | 0.245 | 0.170 | 0.235 | 0.190 | 0.205 | 0.175 |

**Verdict: the rule fix did NOT break the 0.23 pixel ceiling.** All activity
readouts (incl. trained temporal SNN on MPS) sit at pixel level under every rule.
So no LOCAL UNSUPERVISED rule makes PAULA build supra-pixel class structure — the
error-correcting rule fixes stability/memory (EXP13) but not classification. One
lead: legacy WEIGHT-kNN 0.300 (~2.3σ over pixels) — the slow plastic state may be
a marginally richer readout; borderline (n=200), legacy-specific, needs
replication. **Conclusion: the ceiling is a LEARNING-SIGNAL problem, not a rule
problem. Next = the teacher-present loop (#7), the only thing that could break
0.23. The rule fix is necessary infrastructure (a teacher loop on legacy would
saturate).**

---

## SESSION 2026-07-07 — Architecture sweep (three proposals) + MNIST self-correction

### The self-correction that reframes everything
The mini-brain consolidation report concluded "a random reservoir can't add
linearly-decodable class info / plasticity saturates" and over-generalized it to
PAULA. **The user corrected this: the default `snn_classification_realtime/`
pipeline — a CONVOLUTIONAL PAULA net + trained SNN classifier — reaches ~95% on
MNIST, far past the ~92% linear-pixel probe.** So PAULA unambiguously builds
beyond-linear structure. The mini-brain's negative result was regime-specific, not
a property of PAULA. The working pipeline and the failing reservoir differ on EVERY
axis: convolutional (local, retinotopic) vs dense scrambling recurrent pool;
trained multi-layer SNN readout vs single online linear decoder; firings+avg_S over
the full set vs mean[S,F_avg,O] of one glimpse; MNIST (~92% linear) vs CIFAR-gray
(~29% linear). Reconciliation: conv beats a linear probe because of the
NONLINEARITY (ReLU/pool), and its gift is a LOCALITY prior — exactly what a dense
reservoir throws away. **Sharpened hypothesis: the mini-brain underperformed
because it abandoned the topology (conv/retinotopic) + readout (trained nonlinear)
that already work.** All three proposals now test this directly.

### Feature-flagged capabilities added (defaults reproduce old behavior)
- **neuron.py (gated, user-approved this session):** new `plasticity_mode="reward_hebb"`:
  Δw = eta_post·nm·dir·info_val − eta_post·rh_decay·w. Content-dependent BOUNDED fixed
  point (w*=nm·dir·info_val/rh_decay); does NOT rail to the legacy content-free ceiling
  and is NOT the error-correcting rule. `nm` = three-factor reward gate (kappa). New
  param `rh_decay=0.1`. Verified: weights stay bounded+spread (mean~0.96, 0.12–1.52) vs
  legacy railing to ~10. Default `legacy_multiplicative` path is byte-identical.
- **retinal_conv.py:** `conv_bank="rich"` (4 DoG scales + 8 oriented Gabors = 12 filters
  7×7) vs "default" (2 DoG + 4 Sobel = 6 filters 5×5).
- **reservoir.py:** `wiring="retinotopic"` — buffer & pool units tiled in 2D retinal
  space, each samples sources within `local_radius` (LOCAL receptive fields preserving
  conv locality); hubs stay global. vs "random" (dense scrambling baseline).
- **minibrain/heads.py (NEW):** `TorchMLPHead` (trained 2-hidden-layer readout ≈ the SNN
  classifier), `AssociationLayer` (external reward-gated three-factor readout layer),
  `ExternalReservoirPlasticity` (oja / bcm / rmhebb applied to u_i.info OUTSIDE neuron.py).
- **core.py MiniBrainConfig flags:** conv_bank, wiring, readout_head, mlp_hidden,
  ext_plasticity, ext_plast_eta, assoc_layer, assoc_dim, plasticity_mode, nm_kappa,
  rh_decay, eta_post_res. All default to current behavior.

### LOCKED run parameters (user-confirmed 2026-07-07)
- **dwell = 500 ticks/image** (memory forms at ~500 ticks — dwell 40 was measuring an
  unsettled substrate). At 500 the F_avg EMA (τ≈1000) moves ~40%/fixation so the rate
  readout is finally meaningful.
- **warmup = 500 ticks** per FRESH net before any measurement (each arm builds its own
  substrate; also a periodic fresh-net probe as counterfactual). [C. elegans stabilizes
  ~5k ticks; the net is never reset between images so it keeps stabilizing over the run.]
- **400 images/arm** (~200k ticks/arm), gray AND color (cifar10_grayscale + cifar10).
- Run 1 (arm,seed) per process, UNIQUE output_dir, OMP_NUM_THREADS=1, 12 cores.

### Experiment matrix (each arm emits static plots + a live-dynamics animation)
- **Exp 1 Encoder×readout separability:** wiring{random,retinotopic,conv} ×
  conv_bank{default,rich} × readout{linear,mlp} × dataset{gray,color}. Does
  conv/retinotopic + MLP recover beyond-linear? (the MNIST lesson)
- **Exp 2 Substrate-plasticity trajectory (long):** {frozen, native reward_hebb+teacher,
  reward_hebb no-teacher, ext oja/bcm/rmhebb, assoc-layer} × dataset, retinotopic wiring.
  Does reward-gated plasticity ACCUMULATE structure in the right topology?
- **Exp 3 Gaze policy (cluttered):** policy{linear,mlp,recurrent} + oracle/random baselines.
- STATUS: 25-arm sweep DONE; spatiotemporal readout DONE; findings artifact published; powered pool run handed to user.

### RESULTS (2026-07-08)
**Static mean-lens (25-arm sweep, dwell 500/warmup 500/400img):** FLAT NULL across
architectures. sep: retino_def 0.226 ≈ random 0.208 > conv 0.170; rich bank ≈ default;
**MLP readout < linear** (0.208→0.170) — a fancier readout of a mean can't recover what
the mean discarded. This flatness is the fingerprint of averaging away a temporal code.
**Plasticity:** native gated `reward_hebb`+teacher gave the HIGHEST plastic frozen-probe
separability 0.194 (vs frozen 0.083–0.139, vs reward_hebb-no-teacher 0.139) — teacher-gated
reward plasticity looks constructive in the retinotopic topology (opposite of old dense-
reservoir saturation). **Gaze:** learned policies (linear/mlp/recurrent) home to 15.9–17.5px
vs random 26.8px but acc ~chance (oracle 0.133 under mean-based memorize); mlp/recurrent did
NOT beat linear.

**Spatiotemporal readout (`exp_temporal.py`, records O/S/t_ref as (ticks,neurons); mean vs
temporal-linear vs GRU vs LSTM on MPS, sliced input/pool/all):** the headline temporal≫mean
did NOT appear overall (most Δ within σ≈0.033). BUT in the recurrent POOL, temporal>mean for
ALL 3 topologies: conv +0.042, random +0.036, retino +0.015 — while the feedforward input
layer shows NO temporal gain. Cleanest possible shape for "representation lives in the
recurrent dynamics," but ~1σ each and RNN-underpowered (280 train seqs) → DIRECTIONAL, not
closed. Topology: best-achieved retino 0.267 > random 0.217 > conv 0.16 (locality mildly wins).

**Powered confirmation (user ran it, 2026-07-09) — REFUTED the pool effect.** `run_powered_pool.py`
(1200 img/arm ×3 archs ×2 seeds, ~360 test, σ≈0.02, MPS GRU/LSTM). Pool Δ (temporal−mean) collapsed:
conv −0.019, random −0.013, retino −0.001 (all ≤0, vs underpowered +0.042/+0.036/+0.015). The ~1σ pool
signal was NOISE. On this task the MEAN IS A SUFFICIENT STATISTIC — trajectory readout (temporal_lin OR
GRU/LSTM) adds nothing over the mean for CIFAR class. temporal_lin (well-powered) also flat → not an
undertrained RNN. This refutes "downstream classifier decodes more label-info from trajectory than mean
HERE"; does NOT refute the dynamical-representation thesis broadly (CIFAR-from-glimpse is ~static; never
forces dynamics to matter for the label). To truly test: dynamics-only stimuli / regime-identity readout /
causal perturbation. Lesson: distrust ~1σ even across 3 conditions; power up first.
Artifact updated to honest negative (powered-refutation). Videos: gallery/{retino,random,conv,gaze_hunt}.mp4
+ arch/plast_{frozen,rhebb_teach}.mp4.

### CRITICAL: representation is SPATIOTEMPORAL (user, 2026-07-07 — from the paper)
Representation lives in the DYNAMICS over time (the dynamical regime / oscillatory
pattern), NOT in a mean over N ticks or a weight snapshot. While active, the net
holds the concept "in mind"; when activity fades the remnant is engraved in weights
+ state vars (t_ref) that let it RE-ENTER the same regime. **Consequence: `_rep` /
`buffer_reps` reading the MEAN of [S,F_avg,O] collapses the very structure that IS
the representation → static-lens LOWER BOUND.** The running sweep's numbers are that
baseline; the load-bearing FOLLOW-UP = a SPATIOTEMPORAL readout on the top archs
(classify the tick-by-tick O(t)/S(t) trajectory: sub-window rate profiles, oscillation/
autocorrelation/spectral features, state-space trajectory, or a temporal SNN/1D-conv/
recurrent classifier). This is also WHY the MNIST pipeline works — its SNN classifier
reads activity DYNAMICS, not a mean. See memory [[paula-representation-spatiotemporal]].

### SESSION 2026-07-09 — the CIFAR ≥50% push (user: "20% is not acceptable")
Goal: get ≥0.50 on CIFAR-10 with the mini-brain, storage-lean (the original 0.50 pipeline
dumped ~25GB; infeasible). Three parts scoped: A (hit ≥0.50), B (settling-ticks sweep:
mean vs dynamics separability vs settle time), C (attractor analysis).

**ENCODER CEILING established first (`exp_scale.py --mode convfeat`, MLP directly on the
retinal-conv feature vector, bypassing the substrate; 6000 train / 2000 test):**
| config | dims | test acc |
|---|---|---|
| color, grid 16 | 9216 | **0.483** |
| color, grid 24 | 20736 | 0.482 |
| grayscale, grid 16 | 3072 | 0.469 |
→ **The retinal-conv front-end + an MLP alone reaches ~0.48 on CIFAR-10** (matches the
original pipeline's ~0.50). grid 16 already saturates the encoder (grid 24 adds nothing).
**REFRAME: the mini-brain's 20% was NEVER a PAULA/encoder limit — the SUBSTRATE + foveation
were DESTROYING the ~0.48 the front-end already extracts.** So the ≥0.50 goal is a
substrate-PRESERVATION problem, not a feature-manufacturing one. The original 0.50 pipeline
used a conv-PAULA substrate (substrate_type="conv"), so the `conv` arm is the one expected to
preserve the signal; random/retinotopic reservoirs test how lossy a generic spiking substrate is.

**Built (all storage-lean, sharded-parallel, resumable, tqdm, user-launchable):**
- `exp_settle.py` (Part A+B): one pass of max_dwell ticks/image; accumulates running
  S/F/O sums snapshotted at checkpoints {25,50,100,200,400,800} → per-checkpoint MEAN feature,
  and per-interval firing counts → DYNAMICS rate-profile feature. Never stores (T×N). ~8MB/shard,
  9 shards/core per arm, arms {conv, reservoir_random, reservoir_retino}. MLP head per checkpoint
  per feature. Answers: does substrate preserve 0.48 (Part A) + does dyn overtake mean w/ settling (Part B).
  Smoke PASSED (conv arm full JSON, 183 conv neurons; reservoir arm validated via attractor smoke).
- `exp_attractor.py` (Part C): records FULL trajectory for a few exemplars/class (tens of MB),
  produces velocity(t) [fixed-point vs limit-cycle], 2D/3D PCA phase portraits, rotating-3D +
  2D-unfold animations, early-vs-late class-separation ratio [does time help], and a re-entry test
  (present A → fade under tonic-only → re-present A; does state return to A's attractor vs B's).

### RESULTS 2026-07-09/10 — Parts B/C/D (user: "make sure it's not the classifier doing real work")

**PART D — PAULA ATTRIBUTION LADDER (`exp_controls.py`, the decisive experiment).** For a fixed
sample set (600 train / 300 test), four representations read with BOTH a regularized LINEAR decoder
and an MLP. LINEAR is load-bearing (an MLP can mask bad features). Numbers (CIFAR-10 color, grid 16):
| arch | pixel L/M | retina L/M | SUBSTRATE L/M | random-proj L/M | N |
|---|---|---|---|---|---|
| conv             | 0.317/0.283 | 0.383/0.297 | **0.230/0.170** | 0.393/0.387 | 183 |
| reservoir_random | 0.317/0.327 | 0.383/0.283 | **0.217/0.213** | 0.383/0.313 | 304 |
| reservoir_retino | 0.317/0.300 | 0.383/0.283 | **0.220/0.187** | 0.383/0.353 | 304 |
Verdicts (identical across all 3): `paula_lifts_linear ≈ −0.16` (substrate 0.22 vs its retina input 0.38)
and `paula_vs_random ≈ −0.16` (substrate vs a dim-matched random tanh projection of the SAME retina
features, which keeps ~0.38). `retina_over_pixel = +0.066` (fixed conv adds a little). `mlp_compensation ≤ 0`
(MLP recovers NOTHING extra from the substrate → info is genuinely gone, not just under-read).
**PLAIN VERDICT: on this task (CIFAR-from-a-foveated-glimpse, FROZEN substrate, mean readout) the PAULA
substrate is NOT load-bearing — it is a LOSSY bottleneck. The class signal is carried by the fixed
retinal convolution; the spiking substrate DESTROYS ~40% of the linearly-decodable class info (0.38→0.22)
and performs WORSE than a random projection of equal width.** So it is NOT the MLP flattering PAULA (MLP
adds nothing on the substrate); and it is NOT merely the 9216→~300 compression (the random 9216→912 proj
kept 0.38) — the spiking dynamics themselves discard class-label-aligned variance.

**PART B — settling sweep (`exp_settle.py`).** Through the substrate, MLP-on-MEAN plateaus ~0.20 at
every settle checkpoint {25..800} (conv 0.14→0.20 by tick 50 then flat; retino/random ~0.20); the
DYNAMICS (binned-rate profile) feature ≈ mean or worse (conv dyn<mean all cps; retino dyn<mean after
cp25; random dyn≈mean, +0.003..+0.05 within noise). Settling doesn't help past ~50 ticks; dynamics
never beats mean. Corroborates Part D and the earlier powered pool-temporal refutation.

**PART C — attractor dynamics (`exp_attractor.py`, the visual deliverable, 6 mp4s).** All 3 substrates:
SUSTAINED dynamics (velocity ratio late/early ≈ 0.98 → no fixed-point relaxation; limit-cycle/chaotic),
and RE-ENTRY / savings WORKS (re-presenting image A returns the state nearer A's attractor than a
different image B's: conv 0.26 vs 0.32, random 0.43 vs 0.96, retino 0.40 vs 1.14 — strongest for retino).
Time marginally increases state-space class-separation in all 3. Artifacts per arch:
attractor_{velocity,pca2d,pca3d,reentry}.png + attractor_{rotate,unfold}.mp4 under attractor/{conv,retino,random}/.

**RECONCILIATION (does NOT rescue PAULA on this task, but bounds the claim).** The substrate has genuine
rich sustained dynamics + real re-entry/savings (Part C) — the dynamical machinery [[paula-representation-spatiotemporal]]
is real. But those dynamics are NOT aligned with CIFAR class labels in any linearly/MLP-decodable way
(Parts B+D). "PAULA has interesting dynamics" ≠ "PAULA helps classify CIFAR from a glimpse." What's
refuted: the FROZEN substrate + mean/trajectory readout adds class separability on THIS task. What's NOT
tested here: a TRAINED substrate (the working MNIST pipeline is trained conv-PAULA + SNN activity
classifier, full-image — a different regime), or tasks where the label lives in the time course. The
levers to make PAULA load-bearing: (a) train/adapt the substrate to the task (plasticity, not frozen LSM),
(b) don't collapse 9216→300 (keep width / retinotopic fan-out), (c) a task whose class IS a dynamical
regime. **Lesson stays: measure attribution with a LINEAR probe + random-projection control before
crediting the neuron model; a good end-to-end number can be entirely the front-end + readout.**

### RESULTS 2026-07-11 — Part E: attractor-as-representation (user idea; `exp_attractor_class.py`)
User idea: the phase portraits *look* class-clustered, so maybe the class lives in the ATTRACTOR
ITSELF (identity/shape/location), not the mean. Test: 5000 imgs (500/label), full image (no fovea,
grid16), settle 500 + observe 500, reservoir_retino RGB. Compact per-image attractor descriptor
(centroid=mean rate N, temporal std N, cov of a Dr=12 random projection = cycle SHAPE, population
power spectrum K=40). ~11h sharded 9-way. Three questions:

**(a) ATTRACTOR TYPE — NOT a clean limit cycle.** Over 50 exemplars: dominant "complex/weak-cycle"
(43/50; 7 quasi-periodic, 0 clean limit-cycle). Spectral flatness 0.69 (HIGH → broadband; a limit
cycle would be <0.2), ~8.9 spectral peaks (no single dominant frequency), largest-Lyapunov ~+0.012
(max 0.018 → weakly positive), autocorr period ~2 (no clean period), recurrence rate 0.09 (low). So
the sustained dynamics are **broadband / aperiodic — weakly chaotic / high-dimensional fluctuation,
NOT a limit cycle** (corrects the earlier hand-wave "limit-cycle/chaotic": it's the chaotic/complex
end, not periodic). Figures: _spectra/_poincare/_recurrence.png.

**(b) WITHIN-CLASS UNIFORMITY — attractors are NOT class-clustered.** Fisher(between/within)=0.455
(within-class spread 1.97 is 2.2x the between-class centroid distance 0.897); silhouette = -0.097
(NEGATIVE → an average attractor is closer to some OTHER class's attractors than its own). Same-class
images do NOT converge to the same/near attractor. The visual "separability" in the earlier small
phase portraits was a low-D PCA / small-sample ILLUSION — it vanishes at 5000 images. Figures:
_classdist.png (10x10, nearly uniform), _embedding.png (classes overlap), _attractors_rotate.mp4.

**(c) CLASSIFY-BY-ATTRACTOR — does NOT beat the mean.** Linear/kNN/nearest-centroid on the descriptor,
ablated: centroid(=mean) 0.253 | std 0.238 | shape 0.179 | spec 0.155 | cen+shape+spec 0.247 |
FULL 0.241. **key delta = full − centroid = −0.012** (retina ref 0.38). The dynamics-specific parts
(shape/spectrum) carry LESS class info than the centroid, not more. Figure: _classify.png.

**VERDICT: the attractor-classification hypothesis is REFUTED for retino-RGB.** Classifying by the
attractor is no better than (slightly worse than) the mean, which is itself lossy vs retina. The
substrate's attractors are complex/aperiodic and NOT organized by class — the apparent visual
clustering was an artifact. Strengthens the Part D "frozen substrate is lossy/not load-bearing"
conclusion: neither the mean NOR the full attractor geometry recovers class structure the substrate's
own retinal input already has (0.38). To make attractors class-specific you'd have to TRAIN/adapt the
substrate (plasticity); the frozen LSM does not. Other 3 arms (retino-gray, random-rgb, random-gray)
NOT yet run; retino was the best-odds arm (strongest re-entry, most visually separable) and it refutes
clearly, so confirming on the weaker arms is low-value. Related [[paula-substrate-lossy-cifar-glimpse]].
