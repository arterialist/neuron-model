# Scientific test design — the CIFAR10 fixed-feature ceiling and what feature learning buys

**Status: designed and built, NOT run.** Nothing in this directory has been executed beyond a
syntax check. All numbers quoted below as "measured" come from the earlier quick passes; all
numbers quoted as "published" are literature values used only to calibrate the instrument.

## Why this exists

Our conv PAULA substrate gets **~62% top-1** on CIFAR10 (real `first_choice_accuracy` from
`evals/`, after correcting an earlier extraction error that misread top-2/top-3/stability fields
as top-1). Three quick fixed-feature baselines were then run to find the ceiling:

| quick pass | measured |
|---|---|
| `ceiling_cifar.py` — random conv (96/192/256) + linear | 60.4% |
| `ceiling2.py` — wide random conv (256/512/512) + PCA whitening + regularized linear | 64.9% |
| `ceiling_km.py` — k-means dictionary K=512, ZCA + triangle + quadrant pool | 63.1% |

Every one landed at 60–65%, which invited the conclusion "the substrate is already at the
fixed-feature ceiling." **That conclusion is not yet safe**, because the k-means pass is
**10 points below its own published reference point** (Coates & Ng 2011: 71.0% at K=400, 79.6%
at K=1600). A pipeline that underperforms a published result at matched K is a **broken
instrument**, and a broken instrument that happens to agree with our substrate is the most
dangerous possible reading — it would let us conclude "we're at the ceiling" when we are
actually just reproducing our own bug.

So: **validate the instrument against a published point before reading any ceiling off it.**

### The specific defect suspected (to be confirmed, not assumed)

Re-reading `ceiling_km.py`: images came from `ToTensor()` in **[0, 1]**, but Coates' constants
assume **[0, 255]**. At [0,1] the patch variance is ~0.01, so the contrast normalization
computes `(p − mean) / sqrt(0.01 + 10)` ≈ a **constant divisor of 3.16** — i.e. per-patch
contrast normalization was silently a no-op, and ZCA (`eps=0.1`) was over-damped relative to the
data scale. Secondary deviations from the paper: `MiniBatchKMeans` (under-converged) instead of
Lloyd; and a **fixed `LogisticRegression(C=0.01)` with no C sweep at all**, where Coates
cross-validated the L2-SVM's C.

These are hypotheses about our own bug. The ablation stage measures each one's cost rather than
assuming which mattered.

---

## Shared apparatus (`common.py`) — identical across every arm and config

Coates & Ng (2011) single-layer pipeline:

```
6×6 patches → per-patch contrast normalization → ZCA whitening
            → dictionary D (K × 108) → triangle encoding → 2×2 quadrant sum-pool → linear readout
```

- **Splits.** TRAIN / VAL / TEST. Dictionary, preprocessing, and feature standardization are fit
  on **TRAIN only**. Readout regularization `C` is chosen on **VAL**. **TEST is scored exactly
  once per config**, at the end, and never enters any fitting or selection decision. Asserted in
  code (`train_val_split` raises on overlap), not left to discipline.
- **Statistics.** ≥3 seeds for the ladder (different splits + inits). Bootstrap 95% CI over
  seeds. With <2 seeds the code prints `no CI` rather than fabricating an interval.
- **Reference values.** `COATES_NG_2011_REFERENCE` (K→accuracy) is used **only** as a
  calibration target for our instrument. It is never reported as a result of ours.

---

# EXPERIMENT 1 — the true fixed-feature ceiling (`run_kmeans_ceiling.py`)

Staged deliberately so that a broken pipeline cannot produce a confident ceiling number.

### Stage `bench` (minutes)
Times the **real** encode on 200 images at each K and projects the full-run cost from that
measurement. No hand-extrapolation of runtimes; also prints projected feature RAM per K so the
storage budget is a decision, not a surprise.

### Stage `calibrate` (the gate)
Canonical config at **K=400** → compare against the published **71.0%**.

- **H1 (instrument valid):** lands within ±2 points of 71.0%.
- If it does **not**, the script says so explicitly and **the sweep is not run** — the ablation
  runs first to find the defect. This gate is the whole point of the design.

### Stage `ablate` (diagnosis, K=400 fixed)
One-factor-at-a-time from canonical. Each lever's cost is measured:

`scale1` (the suspected bug: [0,1] pixels) · `no_contrast_norm` · `no_zca` · `zca_eps ∈
{0.01, 1.0}` · `kmeans_iters=10` (proxy for the under-converged MiniBatchKMeans) ·
`randpatch_dict` · `encoder ∈ {hard, thresh}` · `stride_2` · `patches_50k`.

**Prediction under the pixel-scale hypothesis:** `scale1` costs ~5–10 points and
`no_contrast_norm` ≈ `scale1` (because at [0,1] the normalization was already effectively off).
If `scale1` costs ~nothing, the hypothesis is wrong and the deficit is elsewhere — report that
plainly and keep looking.

### Stage `sweep` (the actual reading, only after the gate passes)
K ∈ {400, 800, 1600, 3200} with the validated config.

- **H2 (ceiling is ~80%):** accuracy rises with K and saturates near 79–80% at K=1600 → the
  fixed-feature ceiling is **~80%, not ~65%**, and our substrate's 62% has ~18 points of
  headroom that random/naive features leave on the table.
- **H2-null (ceiling really is ~65%):** the curve saturates ~65% even with a validated
  pipeline → the published numbers do not reproduce here, and the substrate genuinely is at the
  ceiling. This would *confirm* the current reading — but only then would it be earned.

### The control that reframes the question — `randpatch`
The dictionary is **random raw whitened patches; no learning whatsoever**. Coates reports this
nearly matches k-means under triangle encoding. If it replicates here, then **dictionary
learning is not what buys the points — the encoding and pooling are.** That directly undercuts
"our substrate is bad because its filters are random," and moves the target to input geometry
and readout.

---

# EXPERIMENT 2 — the learned-feature ladder (`run_learned_ladder.py`)

**The question.** Our substrate is at 60–65% along with every fixed-feature method. Is that
because the filters are (a) not learned, (b) not learned *with labels*, or (c) because filter
learning barely matters and the bottleneck is architectural? These have opposite consequences
for the roadmap, and only a matched comparison separates them.

**The design.** Five arms, **identical in every respect** — patch size, contrast-norm + ZCA
preprocessing, K, triangle encoder, quadrant pooling, dictionary row-normalization, readout
protocol, splits, seeds — differing **only in the rule that produces D**:

| arm | rule | labels? | local? | competition? |
|---|---|---|---|---|
| `random` | gaussian filters | no | — | — |
| `randpatch` | random whitened data patches | no | — | — |
| `kmeans` | Lloyd k-means | no | no (global objective) | implicit (assignment) |
| **`krotov`** | **Krotov & Hopfield 2019 competing hidden units** | **no** | **yes** | **yes (kWTA + anti-Hebbian runner-up)** |
| `supervised` | backprop through the **same** encoder | yes | no | — |

Plus `e2e_cnn`: a conventional multi-layer CNN. **Unmatched by construction**, reported only to
bound how much of the residual gap is *depth* rather than filter learning. It is never compared
arm-to-arm.

### Why `krotov` is the arm that matters for the roadmap

It is a purely **local, unsupervised** Hebbian rule whose only substantive ingredient over a
random substrate is **competition**: the winning unit is potentiated, the k-th ranked unit is
depressed by −δ, the rest are untouched. That is exactly **proposed model-change #1 for PAULA
(lateral inhibition / kWTA)** — the item the A/B/D decision tree pointed at. Its measured gain
over `random` is a **quantified payoff estimate for building competition into the substrate,
obtained before touching `neuron.py`.**

*Documented deviation:* the original learns on whole raw images; here it learns on the same
ZCA-whitened patches every other arm sees, so the ladder stays matched. This is an **adaptation
of their rule, not a replication of their protocol** — their published CIFAR number is therefore
**not** a target for this arm, and the arm is scored only against the other rungs.

### Why `supervised` is a fair rung

`MatchedEncoderNet` makes the *exact* evaluation encoder differentiable (the triangle encoder's
`relu(mean_k dist − dist_k)` is differentiable), so D is learned by backprop **through the same
function the unsupervised arms are scored through**, from the same `randpatch` initialization.
The learned D is then re-normalized and pushed through the same frozen encode + standard readout
as every other arm. So `supervised` differs from `kmeans` in the *rule*, not the *model*.

### Pre-registered hypotheses and their consequences

- **H1 — competition is load-bearing:** `krotov > random`, paired CI excludes 0.
  → Local competitive plasticity builds features from structure alone. Quantified case for PAULA
  model-change #1, expected payoff = the measured Δ.
  **H1-null:** `krotov ≈ random` → adding kWTA has low expected payoff. This *saves building it*
  and is exactly as valuable as a positive.
- **H2 — learning the dictionary matters at all:** `kmeans`/`krotov` > `randpatch`.
  **H2-null (Coates' own finding replicates):** `randpatch ≈ kmeans` → dictionary *learning* is
  not what buys performance; the encoder+pooling is. A random substrate is then **not** the
  problem.
- **H3 — labels are the missing ingredient:** `supervised ≫ all unsupervised`.
  → Unsupervised feature learning caps out here; PAULA's unsupervised hypothesis is
  fundamentally limited at this task and **Test C (supervised κ) becomes the priority path**.
  If instead `supervised ≈ kmeans`, labels are **not** the gap and the ceiling is the
  **architecture** (single layer, 6×6, quadrant pool), not the rule.
- **The combination that would settle the most:** H2-null **and** `supervised ≈ kmeans` — filters
  barely matter, labels barely matter, everything lands ~65%. That would be the strongest
  available evidence that **our 62% substrate is not underperforming** and that depth/structure
  (the milestone already planned) is the only route. `e2e_cnn` bounds how much is there.

### Supervision-amount axis
`--label-frac ∈ {0.01, 0.1, 1.0}` on the `supervised` arm (labels subsample the *D-learning*
phase only; the readout always sees full TRAIN). Asks how much supervision feature learning
actually needs — relevant because PAULA's regime is unsupervised, and a supervised arm that
needs only 1% of labels means something very different from one that needs all of them.

---

## Guards (both experiments)

1. **No leakage.** D, preprocessing, and standardization fit on TRAIN only; `C` selected on VAL;
   TEST scored once per config. Split disjointness asserted in code.
2. **Matched capacity.** Identical K across ladder arms — a rule cannot win by using more filters.
3. **Matched data budget.** No augmentation in any arm, including `supervised` and `e2e_cnn`.
4. **Degeneracy guard.** `dict_health()` records dead filters (never win the triangle encoder),
   mean filter usage, and row-norm CV. **An arm that "wins" with most filters dead is flagged as
   collapse, not learning** — the same guard that caught the weight-collapse/balloon artifacts in
   the basin runs.
5. **Reference values are calibration, not results.** The published curve gates the instrument;
   it is never reported as our finding.
6. **CIs, not vibes.** Paired-by-seed contrasts; a difference whose CI includes 0 is reported as
   no difference however good the means look. `<2 seeds` prints `no CI`.
7. **Caching.** Each config writes its own JSON and is skipped if present, so a long run is
   resumable and nothing is silently recomputed with different settings.

## Compute and budget

Run `--stage bench` **first** — it measures real encode throughput per K and projects both
runtime and feature RAM, so the K=3200 decision is made on measured numbers rather than my
estimates. Levers if the budget is tight: `--train-n` (subsample TRAIN), `stride 2` (already an
ablation, so its cost is known), lower `--readout-epochs`. Storage is constrained: results are
small JSON records; **feature matrices are never written to disk**, only held in RAM per config.

## Order of execution (proposed)

1. `--stage bench` — minutes; decides the rest of the budget.
2. `--stage calibrate` — **the gate**. If it fails, stop and ablate; do not read a ceiling.
3. `--stage ablate` — diagnosis; confirms or kills the pixel-scale hypothesis.
4. `--stage sweep` — the ceiling curve, only with a validated instrument.
5. Ladder at matched K=400, 3 seeds — H1/H2/H3.
6. Ladder `--label-frac` sweep and `e2e_cnn` reference last (most expensive, least diagnostic).

## Decision tree back into the model roadmap

- **Ceiling ~80% and `krotov > random`** → the substrate has real headroom *and* local
  competition is the lever that reaches it. Strongest case yet to add kWTA to `neuron.py`
  (feature-flagged, default-off, with a quantified expected payoff).
- **Ceiling ~80% but `krotov ≈ random ≈ randpatch ≈ kmeans`** → headroom exists but no
  filter-learning rule captures it; the gap is encoder/architecture. Do not add kWTA; target
  input geometry.
- **Ceiling ~65% (H2-null)** → the substrate's 62% is genuinely at the ceiling; CIFAR10 with a
  single fixed layer is simply a ~65% task, our result is not a failure, and the structure-
  learning milestone is the only route. Stop optimizing the substrate for CIFAR.
- **`supervised ≫ everything unsupervised`** → labels are the gap; prioritize Test C
  (supervised κ) over any unsupervised-sharpening work.

## Standing constraints honored

No `neuron.py` edit (nothing here touches the model — these are external baselines that
*inform* whether a model change is worth proposing). No `error_correcting`. No fovea work.
Measure don't assert. Distrust the clean result — **including the clean result that flattered us:
"everything lands at 63%, so we're at the ceiling" is exactly the kind of tidy convergence that
the calibration gate exists to falsify.**
