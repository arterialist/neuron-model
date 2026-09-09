# PAULA-only visual cortex — breaking 70% on RGB CIFAR-10

**Owner: this is a standing research direction, not a one-off experiment.** Goal, locked with
the user 2026-07-17:

> A **PAULA-only** circuit (a PAULA network + an external activity decoder) that breaks the
> **70% barrier on RGB CIFAR-10**, where the substrate **demonstrably adds class separability**
> (is not a passthrough) via **biologically-plausible, online, real-time learning**, and whose
> features are **label-free and data-free**.

Everything below serves that sentence. The end goal of the wider program is an *artificial
human*, so **biological plausibility is a first-class constraint**, not decoration.

---

## 0. Definitions, locked with the user

| term | binding definition (do not drift from these) |
|---|---|
| **PAULA-only** | the network is PAULA neurons end-to-end; a decoder reads their *activity* from outside. All computation happens in PAULA dynamics. |
| **feature arms** | BOTH required: **HW** = hardwired (analytic) kernels; **LRN** = features learned online by PAULA itself. |
| **label-free AND data-free** | feature weights come from **analytic construction** (Gabor/DoG — touches no data) or from **online emergence out of the input stream** (shaped in real time, never precomputed offline on a corpus). **This eliminates the k-means/Krotov-on-patches dictionary** validated in `cifar_ceiling/` (69.55%), because that fits an offline dataset. That ceiling result is retained only as an external difficulty reference. |
| **adds separability** | linear separability of a layer's **output** activity must **exceed** that of its **input**, beating a random-projection-of-input null. A passthrough (output ≤ input) is a **failure**, not a partial win. Hard gate. |
| **bio-plausible learning** | local + online + real-time: per-tick, per-synapse rules driven by the stream; the substrate accumulates experience and adapts usefully. **No backprop through the substrate, ever.** |
| **readout ladder** | `torch MLP` (acceptable) → `closed-form / no-gradient decoder` (better) → `PAULA-only classifier` (perfect). All three are "external decoders"; ambition rises down the ladder. |

### Standing constraints (from project memory, honored throughout)
- **No `neuron.py` edit** — it is the holy-grail model shared with C. elegans. All architecture
  lives in the **builder / config / a non-destructive post-processor**. Inhibition uses the
  already-present `MIN_SYNAPTIC_WEIGHT = -100.0`; plasticity is set via `global_params` in the
  net JSON. Any genuine model need is raised as a feature-flagged, default-off, *quantified*
  proposal — not edited in.
- **No `error_correcting` plasticity.** Online learning uses `legacy` (content-free control) and
  `reward_hebb` (content-dependent) only.
- **No fovea.** This is full-frame vision.
- **Measure, don't assert. Distrust the clean result.** The in-vs-out gate exists precisely to
  falsify a flattering end-to-end number.

---

## 1. Why the current net can't do this, precisely (verified, not assumed)

Empirically checked against the built `conv_cifar10_4l_4k2s3k1s2k1s2k1s.json` (the ~62% net):

- **"Filters" are not convolution.** Each `(filter, y, x)` is a neuron whose source lookup
  `prev_coord_to_id[(c, in_y, in_x)]` **does not depend on the filter index**, and whose weights
  are an independent `random.uniform(0.5, 1.5)` draw. Verified: two positions of the same filter
  index have **different** weights. So `filters` = *F independent random projections of the same
  receptive field* — a **wider locally-connected random reservoir**, with **no shared kernel and
  no translation invariance.**
- **No inhibition.** All 10 800 layer-0 weights are positive (min 0.500, max 1.500); no
  competition anywhere, despite the model supporting negative weights.
- **Consequence.** Fixed random projection + trained readout **is reservoir computing**, whose
  known ceiling on natural images is ~45–55% regardless of width. Widening `filters` moves along
  that curve; it does **not** break it. Breaking 70% requires the two priors the net lacks:
  **(i) the same designed/emergent kernel repeated across all positions** (translation
  invariance) and **(ii) competition** (lateral inhibition / normalization).

---

## 2. Architecture — the "PAULA visual cortex"

A retinotopic hierarchy of PAULA neurons, deliberately mirroring the ventral stream (the honest
"artificial human" framing *and* the label-free/data-free feature bank):

```
   RGB 32x32
     │
[L0 Retina]   color-opponent DoG center-surround, ON/OFF rectified     ← analytic, data-free
     │        (R-G, B-Y, lum on/off): the retinal ganglion prior
[L1 V1 simple] Gabor bank: O orientations × S scales × P phases,       ← the key lever
     │        the SAME kernel repeated at every position (transl. inv.)
     │        + lateral inhibition across orientations at each location  ← competition (topology)
[L2 V1 complex] energy/phase pooling over P phases + local space,       ← phase invariance
     │          strided (subsampling)                                    ← the pooling prior
[L3 expansion] PAULA recurrent dynamic expansion                        ← what PAULA is good at
     │          (nonlinear spatiotemporal lift into a separable code)
[decoder]     external activity readout (ladder, §4)
```

L0–L2 are the fixed/emergent visual front-end that supplies the priors the reservoir lacks. L3
is where PAULA's dynamical strength (§ our own `paula-representation-spatiotemporal` finding) is
supposed to *earn its keep* by adding separability — the gate in §3 tests exactly that.

**Wiring mechanism (no builder-code-path change).** Build the topology with the existing
`build_network_config_direct`, then a **non-destructive post-processor** (`wire_designed.py`)
rewrites layer weights and adds lateral connections in the saved config JSON — the same category
of config-level transform already used to set plasticity via `global_params`. Synapse IDs decode
as `(c·k + ky)·k + kx` (verified from the builder), so an analytic kernel `[c, ky, kx]` maps
directly onto `synaptic_points[].u_i.info`. Requires `connectivity: 1.0` on designed layers so
the full kernel is present (asserted).

---

## 3. THE HARD GATE — substrate must add separability (run first, run always)

The single most important instrument. Before any accuracy is believed, and continuously
thereafter:

- **Metric.** Linear separability of a layer's **input** activity vs its **output** activity,
  measured by a **closed-form** probe (ridge / LDA / nearest-class-centroid — *no gradient
  training*, so the probe itself cannot "learn around" a bad representation). Report probe
  accuracy and class margin.
- **Controls (the anti-passthrough battery):**
  1. **random-projection null** — replace the layer with a random linear map to the same output
     dimension. The layer must beat this, or it is adding structure no better than noise.
  2. **identity/passthrough** — output separability ≤ input separability ⇒ **FAIL**.
  3. **shuffled-activity** — destroys within-sample temporal structure; isolates whether the
     *dynamics* (not just the static code) carry the gain.
- **Pre-registered success:** `sep(output) − sep(input) > 0` with a bootstrap CI over seeds
  excluding 0, **and** the layer beats the random-projection null. Applied per layer, so we know
  *which* stage adds separability and which destroys it (recall: an earlier finding had the
  substrate destroying ~40% of glimpse class info — this gate is what would have caught it).

This gate is what makes a 70% result *mean* something rather than being the readout doing all
the work on a passthrough substrate.

---

## 4. Readout ladder (all "external activity decoder")

| rung | decoder | trains? | why it matters |
|---|---|---|---|
| **R0** | torch snntorch MLP (current pipeline) | gradient | upper-bound reference; acceptable but not the goal |
| **R1** | **closed-form linear** — ridge / LDA / nearest-class-centroid on activity features | **no gradient** | "a readout that doesn't require training a torch net." A no-gradient decoder that hits target says the substrate did the work, not the readout. |
| **R2** | **PAULA-only classifier** — a PAULA readout population trained by the substrate's own reward-modulated plasticity (`reward_hebb` + class-contingent reward via the `nm` gate, per `run_supervised_kappa.py`); class = most-active readout population | **online, in-substrate** | "perfect" per the user — the classifier is itself PAULA, learning by local plasticity. |

Headline accuracy is reported at **R1** (closed-form): it is the honest measure of substrate
quality. R0 brackets the ceiling; R2 is the bio-plausible target.

---

## 5. The two feature arms

### Arm HW — hardwired analytic kernels (fixed circuit)
L0 color-opponent DoG + L1 Gabor bank are **analytic and fixed** (Gabor parameters, not data).
This is the "design the circuit" method that made C. elegans work, applied to vision. Tests the
fixed-circuit PAULA-only ceiling. Sweeps: #orientations, #scales, phase pooling on/off,
color-opponent vs grayscale, lateral-inhibition strength. **Data-free by construction.**

### Arm LRN — features emerge online inside PAULA (the ALife arm)
L1 kernels start random; PAULA's **local competitive plasticity** (`reward_hebb`, κ=0
unsupervised, **plus lateral inhibition as topology** = the competition kWTA that Hebbian RF
emergence needs) reshapes them **from the input stream in real time.** This is the biologically
canonical result — competitive/Hebbian learning on natural-image statistics develops
**Gabor-like receptive fields** (Olshausen & Field; Bell & Sejnowski) — done *online inside the
substrate*, not by offline optimization.

**Measurements unique to LRN (these are the ALife claims, and must be shown, not asserted):**
1. **RFs become Gabor-like with exposure** — reverse-correlation / weight-visualization of L1
   RFs over stream time; quantified by Gabor-fit R² rising from chance.
2. **Separability rises as experience accumulates** — the §3 gate re-run at checkpoints along the
   stream: `sep(output)` increases monotonically-ish with #samples seen. This is the literal
   "learn dynamically, accumulate experience, adapt usefully" requirement.
3. **Emergent ≈ analytic ceiling?** — does LRN approach Arm HW's separability/accuracy? The gap
   is what online emergence leaves on the table vs a hand-designed circuit.

**Known headwind (stated honestly up front):** our basin experiments showed current plasticity
does **not** sharpen *class* basins unsupervised. But that was about *class* structure; Arm LRN
asks a different and more favorable question — whether local competition develops *generic
oriented-edge features* from image statistics, which is a much lower and well-precedented bar.
The two are not in contradiction; if LRN also fails, that is itself a sharp, publishable result
about this substrate's plasticity.

---

## 6. Milestones (each gated; do not advance on a failed gate)

- **M0 — instrument.** Build the §3 separability probe + anti-passthrough controls. Measure the
  **current** 62% net's per-layer separability delta (expected ≈0 or negative — quantify the
  passthrough problem before fixing it). *Gate: probe + controls trustworthy on a known input.*
- **M1 — hardwired V1, minimal.** L0 DoG + L1 Gabor (small bank) + lateral inhibition + L2 pool,
  repeated RFs. *Gate: L1/L2 `sep(out) > sep(in)`, beats random-projection null.* If a hardwired
  Gabor stage doesn't add separability, the whole premise is wrong — find out here, cheaply.
- **M2 — hardwired to 70%.** Widen the bank (orientations/scales), add color-opponent channels
  and complex pooling, tune inhibition. *Gate: **R1 closed-form readout ≥ 70%** on RGB CIFAR-10,
  with the substrate passing §3.* This is the headline target for the fixed arm.
- **M3 — online emergence.** Arm LRN: RFs emerge from the stream; show M-unique measurements
  §5.2. *Gate: separability rises with experience AND emergent RFs are Gabor-like.*
- **M4 — PAULA-only classifier.** Replace R1 with R2. *Gate: full PAULA-only pipeline within a
  stated margin of R1 accuracy.*

Milestone order optimizes for **killing the idea fast if it's wrong**: M0/M1 are cheap and
falsify the core premise before any expensive sweep.

---

## 7. Compute discipline
Neuron count scales with (bank size × grid). **Bench per-sample cost at the target size FIRST**
(build the config, time one sample end-to-end, project dataset-build time) — no hand-extrapolated
runtimes. Storage is constrained: activity recordings are the heavy artifact; cut ticks/samples
before believing an over-budget estimate. Every stage writes small JSON/npz so metrics and RF
visualizations are recomputable offline.

## 8. What success and failure each look like
- **Success:** R1 closed-form readout ≥ 70% on RGB CIFAR-10, substrate passes §3 at L1/L2/L3
  (adds separability, beats random projection), features analytic or online-emergent (never
  data-fit), and — for the LRN arm — separability demonstrably rises as the substrate accumulates
  experience. That is a PAULA-only, bio-plausible, non-passthrough 70%.
- **Informative failure:** hardwired Gabor adds separability and clears 70% but online emergence
  (LRN) cannot develop the features → sharp statement that *this substrate needs designed
  structure, plasticity can't grow it* (which directly scopes the structure-learning milestone).
- **Premise failure:** even a hardwired Gabor+pooling front-end into PAULA can't beat the
  random-projection null at L3 → PAULA's dynamic expansion is a lossy stage for static vision,
  and the honest conclusion is that the substrate should be *bypassed* for images, not decorated.
  M0/M1 surface this before any money is spent.
