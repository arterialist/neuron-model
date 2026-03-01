# Autonomous Network Architecture Research — Methodological Guide

A comprehensive guide for running the spiking neural network (SNN) research pipeline autonomously. All steps are CLI-driven; no interactive prompts. The agent interacts via command-line arguments, reads JSON outputs, and writes Markdown for analysis and brainstorming.

**Scope:** Do not use `pipeline/`. Use standalone Python scripts only.

---

## 1. Overview

The research loop:

```
1. Create network (YAML → JSON)
2. Generate activity recording (network + images → HDF5)
3. Prepare data (HDF5 → .pt features)
4. Train SNN (features → classifier .pth)
5. Run eval (classifier + network → JSONL + summary)
6. Analyze results (JSONL → metrics JSON + Markdown)
7. Record results & brainstorm (write MD, read JSON)
8. Repeat with modifications (edit configs, rerun)
```

Each step produces artifacts consumed by the next. The agent can parameterize any step and iterate.

**Network config:** See [NETWORK_CONFIG_GUIDE.md](NETWORK_CONFIG_GUIDE.md) for full YAML structure and layer options.

---

## 2. Recommendations

| Setting | Recommendation |
|--------|----------------|
| Web server | Do not use `--start-web-server` |
| Ablations | Use default `--ablation none`; do not enable ablations unless explicitly testing |
| Tick delay | Use `--tick-ms 0` (no delay) for activity recording |
| Training device | Use `--device mps` for training (or `cuda` if available) |
| Eval device | Use `--device cpu` for evaluation |

**Model selection:** The PyTorch model with the lowest test loss must be used for evaluation. Use `--test-every N` during training to produce checkpoints, then select the checkpoint with minimum test loss from the config JSON before running eval.

---

## 3. Efficient Exploration

You can get a general sense of direction **without** generating gigabytes of datasets, testing on thousands of samples, or running hours of simulation. The model’s paradigm supports lightweight exploration.

**Principle:** Test on small data with incremental adjustments, infer the trajectory, and scale only when it looks promising.

### Lightweight exploration

| Stage | Use small data | Purpose |
|-------|----------------|---------|
| Activity recording | `--images-per-label 10`–20 | Quick HDF5 runs, sanity checks |
| Training | **Exception** — see below | Classifier accuracy directly affects all downstream results |
| Eval | `--eval-samples 100`–200 | Fast metrics, relative comparisons |

**SNN training exception:** The lightweight rule does **not** apply to training. Classifier accuracy directly affects the rest of the pipeline; an under-trained classifier corrupts all downstream results. The loss curve must reach the bottom or at least slow down before eval. Do not test on 5 epochs when the model would perform well after 25 — that yields misleading metrics. Train until loss converges or clearly plateaus.

**Incremental adjustments:** Change one variable at a time (connectivity, layer size, ticks, etc.). Compare runs to see whether the change improves or degrades metrics. This gives a **trajectory** (e.g. “more connectivity helps up to X, then hurts”).

**Diminishing returns:** Detect when a metric’s increase provides diminishing returns (e.g. accuracy gains flatten, or ticks-per-image stops improving). Note that point for each configuration. Use it to calibrate future runs (e.g. “don’t push beyond X ticks here”). Keep exploring: different configurations can have different points of diminishing returns, so a plateau in one config does not imply a plateau in another.

**When to scale:** Run heavier experiments only when:
- The trajectory is clearly positive across several small runs
- You need stable numbers for reporting or comparison
- You are validating a chosen configuration

### Heavy runs only when worth it

**Heavy runs** = at least 1000+ images per label and 1000+ eval samples. A full iteration typically takes 2–3 hours.

- **Full datasets** (1000+ images per label): Use for final validation or when a branch looks promising.
- **Training:** Always run enough epochs for loss to converge; do not cut training short for exploration.
- **Large eval sets** (1000+ samples): Use when you need reliable accuracy estimates.

Exploration can stay light for activity recording and eval because the model’s behavior tends to be consistent across scales. Training is the exception: use small *data* (fewer images per label) for exploration, but always train until loss plateaus.

---

## 4. Step-by-Step Commands

### 4.1 Create Network

**Script:** `build_network.py`

Builds a spiking neuron network (conv + dense layers) from a YAML config.

```bash
python build_network.py --config path/to/config.yaml [--output-dir DIR] [--name NAME]
```

**YAML config:** See [NETWORK_CONFIG_GUIDE.md](NETWORK_CONFIG_GUIDE.md). Do not use the `filters` parameter in layer definitions.

**Output:** `{output_dir}/{name}.json`

**CLI overrides:** `--output-dir` and `--name` override YAML values. At least one of `output_dir` (YAML or CLI) must be set.

---

### 4.2 Generate Activity Recording

**Script:** `snn_classification_realtime/build_activity_dataset.py`

Feeds images through the network and records internal dynamics (spikes, membrane potentials, refractory periods) to HDF5.

```bash
python -m snn_classification_realtime.build_activity_dataset \
  --network-path networks/my_network.json \
  --dataset-name mnist \
  [--ablation none] \
  [--ticks-per-image 50] \
  [--images-per-label 100] \
  [--output-dir activity_datasets] \
  [--dataset-base BASE] \
  [--tick-ms 0] \
  [--fresh-run-per-label] \
  [--fresh-run-per-image] \
  [--use-multiprocessing] \
  [--export-network-states] \
  [--cifar10-color-normalization-factor 0.5]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--network-path` | required | Path to network JSON |
| `--dataset-name` | mnist | `mnist`, `cifar10`, `cifar10_color`, `cifar100`, `usps`, `svhn`, `fashionmnist` |
| `--ablation` | none | `none`, `tref_frozen`, `retrograde_disabled`, `weight_update_disabled`, `thresholds_frozen`, `directional_error_disabled` |
| `--ticks-per-image` | auto | Simulation ticks per image (auto from network if omitted) |
| `--images-per-label` | 100 | Images per class |
| `--output-dir` | activity_datasets | Output directory |
| `--dry-run` | — | Print recommended ticks per image and exit (no recording) |

**Output:** `{output_dir}/{dataset_base}_{dataset_name}_{timestamp}/` containing `activity_dataset.h5` (HDF5 format).

**Incompatibility handling:** If dataset vector size exceeds network input capacity, the script prints the reason and exits with code 1. No interactive prompts.

**Dry run:** Use `--dry-run` to obtain recommended `--ticks-per-image` before a full run.

---

### 4.3 Prepare Data

**Script:** `snn_classification_realtime/prepare_activity_data.py`

Extracts features from the HDF5 activity and splits into train/test tensors.

```bash
python -m snn_classification_realtime.prepare_activity_data \
  --input-file activity_datasets/<dataset_folder> \
  [--output-dir prepared_data] \
  [--feature-types firings avg_S] \
  [--train-split 0.8] \
  [--scaler minmax] \
  [--scale-eps 1e-8] \
  [--legacy-json] \
  [--use-streaming]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--input-file` | required | Path to HDF5 dataset directory |
| `--output-dir` | prepared_data | Output directory |
| `--feature-types` | firings | `firings`, `avg_S`, `avg_t_ref` (space-separated) |
| `--train-split` | 0.8 | Fraction for training |
| `--scaler` | none | `none`, `standard`, `minmax`, `maxabs` |

**Output:** `{output_dir}/` with `train_data.pt`, `train_labels.pt`, `test_data.pt`, `test_labels.pt`, `metadata.json`.

---

### 4.4 Train SNN Classifier

**Script:** `snn_classification_realtime/train_snn_classifier.py`

Trains the readout classifier on the prepared activity features.

```bash
python -m snn_classification_realtime.train_snn_classifier \
  --dataset-dir prepared_data \
  [--output-dir models] \
  [--model-save-path snn_model.pth] \
  [--load-model-path PATH] \
  [--epochs 50] \
  [--batch-size 64] \
  [--learning-rate 1e-3] \
  [--test-every N] \
  [--device cuda|mps|cpu]
```

**Model selection:** Use the checkpoint with lowest test loss for evaluation. Run with `--test-every N` to save checkpoints; read `*_config.json` for `test_losses` and select the corresponding `.pth` file.

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset-dir` | required | Path to prepared_data |
| `--output-dir` | models | Output directory |
| `--model-save-path` | snn_model.pth | Model filename (relative to output-dir or absolute) |
| `--epochs` | 10 | Training epochs |
| `--batch-size` | 32 | Batch size |
| `--device` | auto | `cuda`, `mps`, `cpu` |

**Output:** `{output_dir}/snn_model.pth` (or `--model-save-path`), plus `*_config.json` and loss graph. Checkpoint saved on interrupt.

---

### 4.5 Run Evaluation

**Script:** `snn_classification_realtime/realtime_classification.py`

Evaluates the trained classifier on live network simulation.

```bash
python -m snn_classification_realtime.realtime_classification \
  --snn-model-path models/snn_model.pth \
  --neuron-model-path networks/my_network.json \
  --dataset-name mnist \
  --evaluation-mode \
  [--eval-samples 1000] \
  [--ticks-per-image 50] \
  [--window-size 80] \
  [--output-dir evals] \
  [--ablation none] \
  [--think-longer] \
  [--max-thinking-multiplier 2.0] \
  [--bistability-rescue] \
  [--device cuda|mps|cpu]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--snn-model-path` | required | Trained classifier |
| `--neuron-model-path` | required | Network JSON |
| `--dataset-name` | required | Same as training |
| `--evaluation-mode` | — | Enable automated evaluation |
| `--eval-samples` | 1000 | Number of test samples |
| `--ticks-per-image` | 50 | Simulation ticks per image |
| `--window-size` | 80 | Ticks used as classifier input |
| `--output-dir` | evals | Output directory for results |
| `--think-longer` | — | Enable extended thinking for uncertain samples |
| `--bistability-rescue` | — | Count top-2 correct when confidence gap is small |

**Output:**
- `{output_dir}/{model_dir_name}_eval_{timestamp}.jsonl` — per-sample results (one JSON object per line)
- `{output_dir}/{model_dir_name}_eval_{timestamp}_summary.json` — aggregated metrics

**JSONL fields (per sample):** `image_idx`, `actual_label`, `predicted_label`, `confidence`, `correct`, `bistability_rescue_correct`, `second_predicted_label`, `second_confidence`, `second_correct`, `second_correct_strict`, `third_predicted_label`, `third_confidence`, `third_correct`, `third_correct_strict`, `first_correct_tick`, `first_correct_appearance_tick`, `first_second_correct_tick`, `first_third_correct_tick`, `had_correct_appearance_but_wrong_final`, `used_extended_thinking`, `total_ticks_added`, `base_ticks_per_image`, `base_time_prediction`, `base_time_correct`.

---

### 4.6 Analyze Results

**Script:** `snn_classification_realtime/analyze_eval.py`

Computes metrics from evaluation JSONL and writes JSON and Markdown reports.

```bash
python -m snn_classification_realtime.analyze_eval \
  --jsonl evals/model_eval_123.jsonl \
  --output-dir analysis \
  [--summary evals/model_eval_123_summary.json] \
  [--num-classes 10] \
  [--class-labels airplane,automobile,...] \
  [--format both|json|markdown]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--jsonl` | required | Path to evaluation JSONL |
| `--output-dir` | required | Output directory |
| `--summary` | auto | Summary JSON (auto-detected from JSONL path if omitted) |
| `--num-classes` | 10 | Number of classes |
| `--class-labels` | 0..9 or CIFAR-10 | Comma-separated class names |
| `--format` | both | `both`, `json`, `markdown` |

**Output:**
- `{output_dir}/metrics.json` — full metrics (JSON)
- `{output_dir}/metrics.md` — human-readable report (Markdown)

**Metrics computed:**
- **Accuracy:** top-1/2/3, strict, bistability rescue, base-time vs final
- **Per-label:** accuracy, confidence, timing, thinking effort, stability
- **Confusion matrix:** counts, normalized, top misclassification pairs
- **Timing:** ticks to first/second/third correct, extended thinking fraction
- **Concept hierarchy:** attractor matrix, leakage, pairwise distances, dendrogram
- **Stability:** bistability rescue rate, appearance vs final, early convergence
- **Specialization:** generalist vs specialist (Gini, std, weakest/strongest labels)
- **Confidence calibration:** binned accuracy, per-label entropy

---

## 5. Full Workflow Example

```bash
# 1. Create network
python build_network.py --config configs/exp_001.yaml --output-dir networks

# 2. Record activity (dry-run first to get ticks)
python -m snn_classification_realtime.build_activity_dataset \
  --network-path networks/my_network.json \
  --dataset-name mnist \
  --dry-run

python -m snn_classification_realtime.build_activity_dataset \
  --network-path networks/my_network.json \
  --dataset-name mnist \
  --ticks-per-image 50 \
  --images-per-label 100 \
  --tick-ms 0 \
  --output-dir activity_datasets

# 3. Prepare data
python -m snn_classification_realtime.prepare_activity_data \
  --input-file activity_datasets/my_network_mnist_1234567890 \
  --output-dir prepared_data \
  --feature-types firings avg_S \
  --train-split 0.8 \
  --scaler minmax

# 4. Train (use mps for training)
python -m snn_classification_realtime.train_snn_classifier \
  --dataset-dir prepared_data \
  --output-dir models \
  --epochs 50 \
  --batch-size 64 \
  --device mps

# 5. Evaluate (use cpu for evals)
python -m snn_classification_realtime.realtime_classification \
  --snn-model-path models/snn_model.pth \
  --neuron-model-path networks/my_network.json \
  --dataset-name mnist \
  --evaluation-mode \
  --eval-samples 1000 \
  --output-dir evals \
  --device cpu

# 6. Analyze
python -m snn_classification_realtime.analyze_eval \
  --jsonl evals/snn_model_eval_1234567890.jsonl \
  --output-dir analysis/exp_001
```

---

## 6. Autonomous Agent Workflow

### 6.1 Branching Experiment Tree

Keep a **tree of experiments** for nonlinear evolutionary exploration. Each experiment is a node; branches arise from modifications (network config, training params, etc.). Subagents can analyze the tree and suggest next steps.

**Directory layout:**

```
experiments/
├── root/                    # Initial experiment
│   ├── config.yaml
│   ├── networks/
│   ├── activity_datasets/
│   ├── prepared_data/
│   ├── models/
│   ├── evals/
│   ├── analysis/
│   └── notes.md
├── branch_a/                # Fork: e.g. different connectivity
│   ├── config.yaml
│   ├── parent: root
│   └── ...
├── branch_b/                # Fork: e.g. more epochs
│   ├── config.yaml
│   ├── parent: root
│   └── ...
└── branch_a_1/              # Fork of branch_a
    ├── config.yaml
    ├── parent: branch_a
    └── ...
```

**Conventions:**
- Each experiment dir is self-contained or references its parent.
- Store `parent: <path>` in a manifest (e.g. `manifest.json`) to reconstruct the tree.
- Subagents read `analysis/metrics.json` and `notes.md` across branches to compare and propose new forks.

### 6.2 Iteration Loop

1. **Initialize:** Create or load a network config (YAML). Start at root or fork from an existing node.
2. **Run pipeline:** Execute steps 1–6 in order within the experiment dir. Capture stdout/stderr for errors.
3. **Read results:** Read `metrics.json` and `metrics.md` from the analysis step.
4. **Brainstorm:** Write `notes.md` with observations, hypotheses, and candidate modifications.
5. **Branch or modify:** Create a new experiment dir (fork) or modify in place. Subagents analyze the tree and suggest next steps.
6. **Repeat:** Run steps 2–5 for the chosen branch.

### 6.3 File Path Conventions

- **Networks:** `networks/{name}.json`
- **Activity:** `activity_datasets/{base}_{dataset}_{timestamp}/`
- **Prepared data:** `prepared_data/` (or per-experiment subdir)
- **Models:** `models/` or `models/{experiment}/`
- **Evals:** `evals/{model_dir}_eval_{timestamp}.jsonl` — use `--output-dir` to organize by experiment
- **Analysis:** `analysis/{experiment}/metrics.json` and `metrics.md`

### 6.4 Error Handling

- **Scripts exit with code 1** on fatal errors (missing file, incompatibility, empty input). No interactive prompts.
- **Activity builder:** On network/dataset incompatibility, prints error details and exits. Agent should parse the message and adjust network or dataset.
- **All scripts:** Validate inputs before running; fail fast with clear messages.

### 6.5 Agent Capabilities

- **Read:** JSON (eval summaries, metrics, configs), JSONL (per-sample results)
- **Write:** Markdown (analysis, brainstorming, experiment notes)
- **Execute:** CLI commands with full argument control
- **Do not use:** `pipeline/` directory or any interactive prompts

---

## 7. Ablation Studies

Use `--ablation` in build_activity_dataset and realtime_classification to test ALERM components:

| Ablation | Effect |
|----------|--------|
| `none` | Full model (Hebbian + homeostasis + predictive error). Baseline ~84.3% |
| `tref_frozen` | Disables homeostatic regulation. ~80.4% |
| `weight_update_disabled` | Hebbian weight updates off; homeostasis only |
| `retrograde_disabled` | Retrograde signaling disabled |
| `thresholds_frozen` | Adaptive thresholds frozen |
| `directional_error_disabled` | Directional error component disabled |

---

## 8. Script Reference

| Step | Script | Entry |
|------|--------|-------|
| 1 | `build_network.py` | `python build_network.py` |
| 2 | `build_activity_dataset.py` | `python -m snn_classification_realtime.build_activity_dataset` |
| 3 | `prepare_activity_data.py` | `python -m snn_classification_realtime.prepare_activity_data` |
| 4 | `train_snn_classifier.py` | `python -m snn_classification_realtime.train_snn_classifier` |
| 5 | `realtime_classification.py` | `python -m snn_classification_realtime.realtime_classification` |
| 6 | `analyze_eval.py` | `python -m snn_classification_realtime.analyze_eval` |

