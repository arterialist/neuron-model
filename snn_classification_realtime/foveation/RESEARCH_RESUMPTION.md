# CIFAR / foveation / PAULA-vision research resumption guide

**Audit status: 2026-07-30.** This document is the operational record for the
July 2026 SNN-debug research stream. It was reconstructed by reading both
Claude-session transcripts (`be7dec7c-f6b9-4e48-b433-78e0ae71291c` and
`4630c9cc-ae60-4a85-a5da-88544a269fbb`), inspecting the recovered tools and
current `neuron-model` worktree, parsing the retained JSON/NumPy result
artifacts, and running the safe command-line import checks listed below. The
conversations are provenance only; this guide, the source, configurations, and
result records are the working context that must survive them.

The companion [machine-readable index](RESEARCH_INDEX.json) is intended for
scripts or future agents. [FOVEATION_RESEARCH.md](FOVEATION_RESEARCH.md) is the
chronological laboratory notebook. This document supersedes neither its raw
observations nor the data. It resolves their status, scope, and replay paths.

**Working-tree preservation note.** This repository currently ignores generic
`*.json` files and `configs/` directories. The machine-readable index and the
large recovered network inputs are consequently preserved at their
scientifically correct local paths but are intentionally not shown by
`git status`. This audit did not alter `.gitignore`, stage, commit, or publish
anything. Before making a release, make an explicit repository-policy decision
about whether the 238 MB recovered-config bundle belongs in version control or
in a checksummed artifact store; do not quietly lose it because it is ignored.

## Read this before quoting an accuracy

There are several incompatible protocols in this repository. They answer
different questions and must never be collapsed into a single "CIFAR score."
The older full-frame classifier runs a PAULA network, trains an external SNN
readout, and may use extended thinking. The foveated mini-brain often presents
one resampled glimpse to a small frozen or locally plastic substrate and uses a
probe. The `convfeat` ceiling explicitly bypasses PAULA. The Coates/Ng
dictionary experiment is a Torch/NumPy external baseline with no PAULA neurons.

The following notation is used throughout. **Artifact verified** means the
result file is present and parseable and its associated source is present.
**Replayed in this audit** means only a safe import, CLI, parser, or hashing
check was run; expensive CIFAR experiments were not silently rerun.
**Limited** means that a result is single-seed, single-split, low-sample, or has
no current-code reproduction. Such a result is a lead, not a publishable
comparison. All reported accuracy values are top-1 unless the column says
otherwise.

## Fast orientation and layout

Run commands from the repository package root:

```bash
cd /Users/arterialist/Projects/agi-research/neuron-model
uv sync --extra viz
```

The maintained classifier pipeline lives in
`snn_classification_realtime/activity_dataset_builder/`,
`activity_preparer/`, `snn_trainer/`, and `realtime_classifier/`. Its usual
flow is image stream → PAULA network recording → HDF5 activity features →
external SNN decoder → evaluation. `evals/` contains its immutable result
summaries and `networks/` contains network JSON inputs. Do not substitute a
mini-brain result for an evaluation result from this pipeline.

The foveation program is in `snn_classification_realtime/foveation/`.
`fovea.py` provides a movable fixed-size crop plus inhibition of return;
`retina.py` provides a sharp fovea and blurred peripheral gist on a common
grid; `retinal_conv.py` applies fixed DoG/Gabor, ON/OFF, and pooling
preprocessing; and `perception.py` builds or wraps a retina-sized PAULA
network. The `minibrain/` subtree combines these pieces into controlled
experiments. All retained foveation result artifacts are under
`foveation_results/`.

`snn_classification_realtime/paula_vision/` is a separate full-frame,
PAULA-only visual-cortex program. Its design contract is in
[EXPERIMENT_DESIGN_paula_vision.md](../paula_vision/EXPERIMENT_DESIGN_paula_vision.md).
It measures per-layer closed-form separability before accepting a classifier
score, so it is the correct home for the 70%-target work. The recovered M1/M2
network configurations are now local to this program at
`paula_vision/configs/recovered_snn_debug_2026_07/`.

`cifar_ceiling/` is neither of those systems. It is an external fixed-feature
reference implementation. Its result can establish what the task affords, but
it cannot establish what PAULA has achieved.

## Glossary

| Term | Precise meaning in this repository |
| --- | --- |
| **PAULA substrate** | The `Neuron` / `NeuronNetwork` simulation through which signals are ticked. A fixed convolution, an MLP, a linear probe, and the external Coates pipeline are not PAULA. |
| **Retina / RetinalConv** | Two different preprocessing stages. `Retina` produces sharp foveal and blurred peripheral image channels. `RetinalConv` is a fixed DoG/Gabor ON/OFF convolutional feature encoder. |
| **Foveation** | Moving a fixed-resolution retinal input over an image. It is not merely cropping a 32×32 model: the smaller network is built for the retinal grid, so all of its inputs are driven. |
| **Mini-brain** | The experimental wrapper in `minibrain/core.py`: retina, optional fixed retinal convolution, a PAULA conv or reservoir substrate, an online decoder, and optionally a teacher and plasticity rule. |
| **Conv substrate** | A small PAULA convolutional topology created by `PerceptionNetwork`; it is distinct from `RetinalConv`, which is fixed preprocessing. |
| **Retinotopic reservoir** | A reservoir whose local buffer/pool wiring samples nearby 2-D retinal positions. The random reservoir is its topology-destroying comparator. |
| **Settled / mean feature** | Per-neuron summary statistics across a terminal observation window, normally S, F_avg, and O. It intentionally discards order. |
| **Dynamics feature** | Binned firing-rate profiles or tick-wise sequences used to test whether temporal order adds label information beyond the mean. |
| **Attractor descriptor** | Compact per-image vector comprising a rate centroid, temporal standard deviation, random-projection covariance/shape, and population spectrum. It is not a single visual PCA plot. |
| **Re-entry / savings** | After input A, decay, and A again, the state lies nearer the first A trajectory than a B trajectory. This demonstrates state dependence, not class identity. |
| **Separability gate** | The PAULA-vision requirement that a layer beat both its input representation and a same-width random nonlinear projection under no-gradient probes. A good downstream MLP alone does not pass it. |
| **M0 / M1 / M2** | PAULA-vision milestones: instrument the existing circuit; test a minimal analytic V1-like circuit; then test a larger retina/V1 hierarchy. They are not the `m0`/`m1` neuromodulator channels. |
| **Teacher / κ / reward_hebb** | An explicit class-contingent reward/stress signal controlling opt-in local plasticity. It is a research intervention and can leak labels if used as a decoder feature; keep it distinct from label-free feature tests. |
| **Participation** | Fraction of neurons spiking in a measurement window. It is a regime check, not evidence of class information. |
| **R1** | Closed-form external readout family (ridge, LDA, nearest-class-centroid). It is the honest PAULA-vision headline readout because it cannot learn around a bad representation by gradient descent. |

## Evidence ledger

### 1. Maintained full-frame PAULA classifier

The strongest retained direct classifier artifact found by this audit is
`evals/conv_cifar10rgb_4l_4k2s3k1s2k1s2k1s_100t_2000s_norm06_cifar10_color_1766431808_avg_S_firings_e60_lr0.0005_b64_eval_1766697578_summary.json`.
It evaluates `networks/conv_cifar10rgb_4l_4k2s3k1s2k1s2k1s.json` with an
external SNN checkpoint on 1,000 RGB examples. Its first-choice accuracy is
**64.0%**, with 71.6% second-choice and 78.1% third-choice coverage. The
record enabled extended thinking, reports base-time accuracy of 18.7%, and
attributes a large part of its final score to extra ticks. It is therefore
evidence that this particular trained full-frame PAULA-plus-decoder protocol
reached 64.0% under its own evaluation policy; it is not evidence that a frozen
foveated substrate or a closed-form R1 probe reaches 64.0%.

The common session shorthand "about 62%" refers to this family of historical
full-frame runs, but it was used beside several incompatible baselines. For
new work, cite the exact summary filename, first-choice field, number of
examples, feature types, and thinking policy instead. A retained grayscale
evaluation of a different network/protocol reaches only 20.16% on 124 examples;
it is a warning against treating architecture names or dataset names as enough
metadata to compare results.

### 2. Early foveation and local-dynamics program

The first foveation series established useful regime constraints rather than a
classifier. The reward–saliency scan initially produced a +0.87 correlation on
two images, which is explicitly a false positive. Larger scans found about
+0.11 to +0.13 in the sparse spiking regime. Later glow mapping found strong
informativeness-to-edge correlation on simple images, but the class-stratified
localization experiment showed that it is an edge/contrast saliency map, not an
object detector on clutter. Those artifacts and their exact parameters are
`exp1_scan_*.json`, `exp1b_settle_sweep_*.json`, `exp4_glow_*.json`, and
`exp10_glow_localization_*.json` under `foveation_results/`. The correct
working interpretation is: label-free glow can propose high-contrast locations;
it does not establish semantic object localization.

The depth/capacity and delay experiments are negative but bounded. Width alone
did not lift small grayscale PAULA probes beyond roughly pixel-level
separability; deeper random stacks lost the signal in their last layer even
when firing participation was restored. The delay sweep changed dendritic and
axonal delay policies without a reliable effect. Its null is not decisive,
because it used a two-layer architecture already at chance. The appropriate
follow-up is a powered single-layer delay study with a representation that
actually carries signal, not another sweep of the failed deep stack.

The efficacy/familiarity results distinguish plastic-state behaviour from
classification. Legacy multiplicative learning drives efficacy toward a common
ceiling and behaves as a familiarity flag. The opt-in error-correcting rule
prevented that saturation and retained content-dependent values, but did not
raise the activity/readout CIFAR ceiling. These conclusions are recorded in
`experiment_familiarity.py`, `experiment_readout.py`, and the
`readout_{legacy_multiplicative,error_correcting}_*.json` records. The
borderline legacy weight-kNN result should not be promoted: it is a
single-protocol lead, not a replicated gain.

### 3. Mini-brain architecture, attribution, and time-course experiments

The 25-arm architecture sweep is stored in
`foveation_results/minibrain/arch/AGG_summary.json`. It used the static mean
lens, mostly one seed per arm, so it is directional. Retinotopic/random
reservoir linear separability was about 0.208–0.226 while the conv arm was
0.142–0.170; changing the static MLP head did not recover a hidden class code.
Teacher-gated reward-Hebbian arms reached 0.194 frozen-probe separability in
this protocol versus 0.083–0.139 for some frozen or non-teacher controls. This
is a hypothesis-generating plasticity signal, not a completed learning claim:
it needs paired multi-seed validation, an input/random-projection control, and
an explicitly documented no-label-leak protocol.

The foveated encoder ceiling is real and intentionally **not PAULA**. In
`minibrain/scale/`, `exp_scale.py --mode convfeat` sends fixed retinal-conv
features directly to an MLP: RGB grid-16 was 0.4835 at 6,000 samples, 0.5024
at 15,000, and 0.565875 at 40,000; grayscale grid-16 was 0.469. The substrate
is bypassed in this mode. These values establish that the fixed front end has
class information; they must never be presented as an achieved PAULA score.

The strongest anti-attribution control is
`minibrain/controls/AGG_controls_summary.json`, with raw records
`ctrl_{conv,reservoir_random,reservoir_retino}_c10.json`. Each arm uses 600
images, a 300-example test, dwell 200, and one seed. For the conv arm, linear
accuracy was 0.317 for pixels, 0.383 for the fixed retina, 0.230 for the PAULA
substrate, and 0.393 for a dimension-matched random tanh projection. The three
architectures agree on a substrate-versus-retina loss of 0.153–0.166 and a
substrate-versus-random loss of 0.163–0.166. The frozen, single-glimpse
substrate is therefore a **lossy bottleneck in this exact protocol**. The
result does not say that all trained PAULA circuits are intrinsically lossy.

The settling sweep independently supports that limited verdict. In
`minibrain/settle/AGG_settle_summary.json`, the conv substrate mean feature
rises to about 0.20 by 50 ticks and does not improve through 800 ticks; its
dynamics feature is lower at every checkpoint. The reservoir arms hover around
0.19–0.22; a few small positive differences are not replicated significance
tests. The powered temporal confirmation is more informative: in
`minibrain/pool_powered/AGG_powered_summary.json` its pool temporal-minus-mean
deltas are conv −0.019, random −0.013, and retinotopic −0.001. Thus the earlier
small temporal advantages are treated as noise for CIFAR-from-a-static-glimpse,
not as evidence of a general failure of dynamical representation.

The 30-trajectory attractor visualisations are retained for all three
substrates in `minibrain/attractor/{conv,random,retino}/`. They show sustained
activity (late/early velocity ratios 0.982–0.988) and re-entry; for example,
retinotopic re-entry distance to A is 0.3969 versus 1.1376 to B. These facts
show state dependence and savings. They do not establish class attractors.
That question was tested directly by the 5,000-image retinotopic RGB run in
`minibrain/attractor_class/attr_reservoir_retino_c10.json`: the dominant type
is `complex/weak-cycle` (43 of 50 inspected exemplars), Fisher ratio is 0.455,
silhouette is −0.097, and full attractor-descriptor linear accuracy is 0.241,
below centroid-only 0.253. The full-minus-centroid delta is −0.012. The
attractor-as-class-identity hypothesis is consequently **refuted for that
one retinotopic RGB frozen-substrate protocol**. Other architectures and
trained-substrate variants remain untested, rather than confirmed negative.

The active-gaze artifacts likewise do not support a success claim. Across the
three completed grayscale `gaze9` seeds, learned gaze averages 0.16333 and
random gaze 0.16778; the oracle averages 0.18333. On cluttered canvases,
learned gaze averages 0.10083, random 0.09333, and static-long 0.11333. The
policy sometimes reduces gaze-to-target distance in the early sweep, but no
completed result shows a robust accuracy advantage. Treat gaze work as a
control surface and training harness, not a validated object-seeking agent.

### 4. PAULA-only visual-cortex program: M0, M1, and M2

This later program repaired an important mistake in the mini-brain narrative:
the existing full-frame trained PAULA classifier can create substantially more
linearly decodable structure at early layers. Its evidence is a **different
protocol** from mini-brain and has its own gate. `paula_vision/separability_probe.py`
resets per sample, records each PAULA layer, and evaluates nearest-class
centroid, Fisher LDA, and closed-form ridge while comparing each layer against
its input and a random nonlinear projection.

M0, `foveation_results/paula_vision/sep/M0_current_rgb.json`, tests the
historical four-layer RGB configuration on 400 examples. Ridge is 0.1979 for
pixels, 0.2646 at L0, **0.6062 at L1**, **0.6875 at L2**, then **0.1083 at
L3**. Its random controls show L1 and L2 beating their recorded nulls, while
L3 collapses. The defensible result is not "PAULA is lossy" or "the entire
network is successful"; it is that this circuit's early layers add substantial
linear separability under this probe and its final layer destroys it.

M1 implements a 1,088-neuron circuit with 49,152 external inputs. The exact
network inputs were recovered and are now repository-local in
`paula_vision/configs/recovered_snn_debug_2026_07/`; see that directory's
README for SHA-256 provenance. On the retained 400-example probe, the random
local control reaches L1 ridge 0.6750. The analytic Gabor configuration reaches
L0 ridge 0.5375 and L1 ridge 0.6438; adding recorded lateral inhibition reaches
L0 ridge 0.5813 but lowers L1 ridge to 0.4688. These are meaningful layer-gate
measurements, but they are a single split/run and not end-to-end CIFAR
accuracy. The separate 1,500/1,000 readout record for `m1_gabor` peaks at
0.323 nearest-class-centroid on L0; it uses a different probe budget, so it
must not be placed in the same numerical table as the 400-example M1 gate.

M2's retained `retina_test.json` is a 4,160-neuron, 82,081-connection
architecture. Its 400-example result (`sep_m2/M2_retina_random.json`) has
ridge 0.250 at L0 and 0.1062 at L1/L2. It is a genuine negative depth/fade
result for that configuration, not a verdict on all possible designed visual
hierarchies. The reward-Hebbian M1 configuration was recovered but no matching
completed result summary was found, so it is explicitly **unverified**.

### 5. External fixed-feature ceiling

`cifar_ceiling/` is a task reference only. The canonical K=400 Coates-style
dictionary result at `cifar_ceiling/results/kmeans/canonical_K400_s0.json` is
69.55% test accuracy, 69.66% validation accuracy, with explicit [0,255] pixel
scale, contrast normalization, ZCA, triangle encoding, and quadrant pooling.
It contains no PAULA import, network, or tick. Its presence falsifies the old
claim that fixed-feature CIFAR necessarily caps near 63%, but does not validate
or invalidate a PAULA topology. The experiment-design file is stale where it
says the program was not run; the result JSON is the stronger provenance and
this discrepancy should be corrected when the external-baseline work is
packaged.

### 6. Later basin, completion, and continuous-plasticity follow-through

This branch is **MNIST**, not CIFAR-10. It was deliberately kept in the same
ledger because it tests the mechanistic claim behind the foveation work: can a
PAULA substrate acquire class-specific, held-out attractor basins through the
existing local-plasticity rules? Do not transfer an MNIST basin result to CIFAR
performance without a matched CIFAR experiment.

The two completed basin-generalization acquisitions are
`foveation_results/basin_generalization/{legacy,rh05}.npz`, with their
configuration metadata in the corresponding `*_meta.json` files. Both stream
80 MNIST examples per class through a 144-neuron network for 3,000 ticks per
example (settle at tick 1,200), hold out 8 examples per class, checkpoint every
20 examples, and probe `frozen` and `on` plasticity modes. They are complete
single-run records (41 checkpoints; about 9.5--10.5 CPU hours per arm), not a
multi-seed confidence interval. The contemporaneous design document calls the
unsupervised mean-descriptor experiment a negative: its stored records are
valuable for reanalysis, but do not constitute a publishable universal
negative. The next proposed tests are explicitly specified in
`basin_generalization/EXPERIMENT_DESIGN_no_model_change.md`; the required
standards are held-out REF/QUERY partitions, interleaved as the primary order,
frozen-weight evaluation, paired legacy control, and at least three seeds with
a bootstrap slope interval.

`foveation_results/completion_abd/completion_summary.json` is the first
occlusion/completion probe. It compares legacy versus `rh05`, frozen versus
plastic-on, static versus temporal descriptors, and whole, cropped, random
mask, and shuffled-pixel inputs at 1,200 streamed MNIST presentations. Its
best whole-image endpoint is 0.289 (`rh05`, frozen, dynamics); 50% crop and
random-mask endpoints remain roughly 0.094--0.156, near chance (0.10). Every
reported interval is `[null, null]`: it is one run rather than a confidence
interval. The bounded conclusion is therefore **no demonstrated held-out
pattern-completion gain**, not proof that PAULA can never complete patterns.

The activity-dataset analysis adds an important qualification to the old
"time has no information" shorthand. In
`substrate_attractors/mnist_sparse025_dynamics.json` (2,000 examples, 300
ticks) mean-rate linear accuracy is 0.558 and a dynamics-only descriptor is
0.555; adding the latter to the mean reaches 0.573, a +0.015 lift. The longer
3,000-tick record (500 examples) is 0.500, 0.487, and 0.507 respectively, a
+0.007 lift. Dynamics are label-informative above chance in this MNIST
substrate, but they did not materially outperform a settled mean or make the
full descriptor better. This is the appropriate evidence to cite when choosing
whether a new basin replay should retain trajectory recordings.

The continuous-adaptation controls correctly expose the class-equals-time
confound. The 10-class run has 150 MNIST examples per class, a continuous
3,000-tick stream, and four arms (`legacy`, `rh002`, `rh01`, `rh05`) in both
blocked and interleaved order; its exact data are in
`continuous_adapt_10c/continuous_summary.json` and per-arm `.npz`/metadata.
The Fisher gains vary by arm and ordering (legacy: +0.075 blocked, +0.035
interleaved; rh002: -0.251, +0.075; rh01: +0.103, +0.082; rh05: -0.111,
-0.016), while weight totals either balloon (legacy/rh002) or collapse
(notably rh01 interleaved ends at -807.9). The three-class `continuous_adapt_6k`
run tells the same cautionary story. These are single-run process diagnostics,
not evidence of robust class-specific learning. Any resume must use
interleaving, valid weight/range checks, held-out probes, and multiple seeds
before ranking a rule.

Two further artifact families are preserved but intentionally have no numeric
headline yet. `tref_stability/{constant,streaming}/` contains complete
16,000-tick CIFAR-gray calibration traces at connectivity 0.1/0.25/0.5/1.0
(`tref`, participation, F_avg, saturation, efficacy, and state snapshots);
`savings/{compare,formation_c0.1,formation_c0.5,timescale}/` contains the
corresponding CIFAR-gray familiarization/re-entry arrays and figures. They are
useful raw diagnostic data, but they lack a retained aggregate statistical
summary and must be re-analysed from their `.npz` arrays before becoming a
scientific claim. The earlier qualitative interpretation of this family is
already bounded in section 2: plastic state and re-entry are not classifier
evidence.

#### Complete artifact map for this branch

| Family | Dataset / purpose | Runner or analyser | Evidence and current status |
| --- | --- | --- | --- |
| Basin generalization | MNIST; held-out class-basin trajectory | `run_basin_generalization.py` | `basin_generalization/{legacy,rh05}.npz`; complete single-run acquisition, no multi-seed inference |
| Completion | MNIST; occluded held-out probing | `run_completion_probe.py`, `analyze_completion_probe.py` | `completion_abd/completion_summary.json`; no demonstrated completion gain, intervals absent |
| Continuous adaptation | MNIST; class-time confound and plastic-weight dynamics | `run_continuous_adaptation.py`, `analyze_continuous_adaptation.py` | `continuous_adapt_6k/` and `continuous_adapt_10c/`; single-run diagnostics, use interleaved as primary |
| Attractor descriptors | MNIST; mean versus temporal activity features | `analyze_substrate_attractors.py` | `substrate_attractors/*_dynamics.json`; small +0.007 to +0.015 temporal addition, no robust advantage |
| t_ref stability | CIFAR-gray; long-run operating-regime calibration | `foveation/experiment_tref_stability.py` | `tref_stability/*/tref_stability_raw.npz`; preserved raw traces, aggregate analysis pending |
| Savings/familiarity | CIFAR-gray; re-entry and adaptation dynamics | `foveation/experiment_savings.py` | `savings/*/*.npz`; preserved raw traces, aggregate analysis pending |

## Reproducible entrypoints

All long commands below create a new output tag. Start with the help command
and a one-example smoke on a new output directory before allocating a full
CIFAR run. The first command is a safe current-code import check; the others
are research runs and may take hours.

```bash
cd /Users/arterialist/Projects/agi-research/neuron-model

# Inspect supported architectures and preserve aliases used by recovered launchers.
uv run python -m snn_classification_realtime.foveation.minibrain.exp_attractor_class --help

# Reproduce a fresh, frozen M1 separability measurement. Do not overwrite sep_m1/.
uv run python -m snn_classification_realtime.paula_vision.separability_probe \
  --net snn_classification_realtime/paula_vision/configs/recovered_snn_debug_2026_07/m1_gabor.json \
  --dataset cifar10_color --per-class 40 --ticks 100 --settle-frac 0.5 \
  --signal-gain 0.336 --norm 0.6 --seeds 0,1,2 --freeze --workers 1 \
  --out foveation_results/paula_vision/replay_2026_07 --tag m1_gabor_frozen_replay

# Repeat the decisive frozen-glimpse attribution ladder into a fresh directory.
uv run python -m snn_classification_realtime.foveation.minibrain.exp_controls \
  --dataset cifar10 --archs conv,reservoir_random,reservoir_retino \
  --samples 600 --test 300 --dwell 200 --warmup 500 --split 0.6666667 \
  --seed 0 --shards 9 --out foveation_results/minibrain/controls_replay_2026_07

# Run a reduced attractor identity check; the historical 5,000-image protocol is expensive.
uv run python -m snn_classification_realtime.foveation.minibrain.exp_attractor_class \
  --arch reservoir_retino --dataset cifar10 --per-class 100 --dwell 500 --observe 500 \
  --shards 9 --seed 0 --out foveation_results/minibrain/attractor_class_replay_2026_07

# Re-analyse existing MNIST activity without changing a PAULA model.
uv run python snn_classification_realtime/analyze_substrate_attractors.py \
  --h5 activity_datasets/attr_mnist_sparse025_mnist_run1/activity_dataset.h5 \
  --tag mnist_sparse025_reanalysis --out foveation_results/substrate_attractors_reanalysis

# Summarise a fresh completion run; analysis is safe, but does write its selected output directory.
uv run python snn_classification_realtime/analyze_completion_probe.py \
  --dir foveation_results/completion_abd \
  --out foveation_results/completion_abd_reanalysis
```

The static foveation experiments have direct script entrypoints too, for
example `experiment_scan.py`, `experiment_glowmap.py`,
`experiment_paula_capacity.py`, `experiment_delays.py`, and
`experiment_readout.py`. They default to writing under `foveation_results/`;
always set `--output-dir` to a dated directory when reproducing a claim. The
recovered historical launchers live in `recovered_snn_debug_tools/launchers/`.
They are useful provenance, but their output names are historical and should
not be reused for publication runs.

## Current blockers and the shortest honest next sequence

The principal scientific gap is not another decoder sweep. The legacy trained
full-frame pipeline, the frozen foveated mini-brain, and the M0/M1 layer probes
give conflicting-looking numbers because they differ in circuit, input geometry,
state/plasticity, and readout. There is no multi-seed, matched-protocol result
that connects the promising M0/M1 separability gain to a reproducible R1 CIFAR
accuracy, or that shows online PAULA learning improves it. The 70% PAULA-only
goal is therefore **not achieved**.

The correct sequence is to first replay M0 and the three M1 configs using the
current code with at least three seeds and documented train/test splits. Report
layer-input, layer-output, random-projection, and shuffled-time controls for
each. Second, establish an R1 readout for the same frozen config and split;
only then compare it with the trained external SNN. Third, run the analytic
M1/M2 architecture sweep with a measured compute budget and preserve the
input JSON plus all result metadata. Finally, evaluate online `reward_hebb`
only after the fixed circuit passes the gate, while measuring receptive-field
change, layer separability over exposure, and a no-reward/no-teacher control.
Do not claim emergent features from a result that only changes an external
decoder.

The engineering blocker is input scatter cost. The recovered M1 configs have
49,152 external inputs per tick and the M2 retina config has 3,072. Their
historical measurements predate later performance changes in `neuron/network.py`.
The configs are now preserved, but numerical reproduction is not guaranteed
until a replay records the current package and dependency versions. Large
historical experiments must not overwrite their source JSON, shard archives, or
summary files.

The earlier mini-brain frozen-substrate negatives should not be used to dismiss
the full-frame PAULA program. Conversely, early-layer M0/M1 separability should
not be used to claim a high end-to-end PAULA-only score. The first is about a
small foveated, mostly frozen bottleneck; the second shows that a different
full-frame circuit can produce a useful intermediate code before later depth
collapses it. These are complementary constraints for the next circuit design.

## What was checked in this audit

The five recovered PAULA-vision configurations parse as JSON and their hashes
match both their Claude-job originals and the earlier non-destructive recovery
copies. Their measured sizes are 1,088 neurons / 32,662 connections / 49,152
external inputs for M1 base/Gabor/reward-Hebbian, 1,088 / 48,022 / 49,152 for
M1 Gabor-inhibition, and 4,160 / 82,081 / 3,072 for the M2 retina test.
`exp_attractor_class --help`, `paula_vision.separability_probe --help`, and
`minibrain.exp_controls --help` all executed successfully in the `uv` project
environment. Representative JSON result records named in this document were
parsed directly. No expensive training, new scientific run, maintained-model
behaviour change, or deletion was performed by this audit.

The session transcripts also contain abandoned arguments, cancelled jobs, and
claims later retracted by controls. They are deliberately not treated as
results unless a corresponding repository artifact is listed above. In
particular, the early two-image saliency positive, a small apparent temporal
gain, visual phase-portrait class clustering, and the old 63% fixed-feature
ceiling were all invalidated or downgraded. Keeping those corrections beside
the positive findings is essential for a resumable research record.
