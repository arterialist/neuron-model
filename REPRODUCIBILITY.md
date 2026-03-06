# PAULA Reproducibility Guide

This guide walks through reproducing the PAULA/ALERM validation pipeline: from creating a neuron network to obtaining real-time evaluation results.

---

## Overview

The pipeline separates **biological network simulation** (inference, local learning) from **readout evaluation**:

1. **Network creation** – Build a spiking neuron network (conv + dense layers)
2. **Activity recording** – Feed images through the network, record phase-space dynamics
3. **Data preparation** – Extract features (firings, avg_S, avg_t_ref) for classifier training
4. **Classifier training** – Train an SNN readout on the recorded activity
5. **Evaluation** – Run real-time classification and measure accuracy
6. **Analysis** (optional) – Generate metrics JSON and Markdown from evaluation results
7. **Visualizations** (optional) – Explore dynamics, clustering, and emergent structure

All scripts are run from the project root. Use `python -m snn_classification_realtime.<module>` for package scripts (recommended) or `python <script>.py` for root-level entrypoints.

---

## 1. Environment Setup

```bash
git clone https://github.com/arterialist/neuron-model.git
cd neuron-model

# Install with uv
uv sync
```

---

## 2. Create a Network

### Option A: Standalone builder (JSON output only)

Use `build_network.py` when you only need a JSON network file. Fast, no Neuron objects.

Create a YAML config (e.g. `my_config.yaml`):

```yaml
dataset: mnist
output_dir: networks
name: my_network

layers:
  - type: conv
    kernel_size: 4
    stride: 2
    filters: 4
    connectivity: 0.8
  - type: conv
    kernel_size: 4
    stride: 2
    filters: 6
    connectivity: 0.8
  - type: dense
    size: 144
    connectivity: 0.25
```

Build:

```bash
python build_network.py --config my_config.yaml --output-dir networks --name my_network
# Output: networks/my_network.json
```

If `output_dir` and `name` are set in the YAML, `--output-dir` and `--name` can be omitted.

### Option B: Interactive training (build + explore)

Use `interactive_training.py` to build a network interactively, visualize it, and optionally save:

```bash
python interactive_training.py
```

Follow prompts to select dataset (MNIST, CIFAR10, etc.), build the network, and save to a JSON file.

This script hasn't been used in a while so it may not be perfectly functional.

### Option C: Pipeline (build from YAML)

The pipeline can build networks from embedded config. See Section 7.

---

## 3. Build Activity Dataset

Feed test images through the network and record internal dynamics (spike rates, membrane potentials, adaptive thresholds).

```bash
python -m snn_classification_realtime.build_activity_dataset \
  --network-path networks/my_network.json \
  --dataset-name mnist \
  --output-dir activity_datasets \
  --dataset-base my_network \
  --output-suffix run1 \
  --ticks-per-image 50 \
  --images-per-label 100 \
  --tick-ms 0 \
  --ablation none
```

- `--dataset-base`: Base name for output directory (default: network filename stem)
- `--output-suffix`: Suffix for deterministic naming (e.g. `run1`). Omit for timestamp-based names.
- Use `--dry-run` to print recommended ticks per image without recording.

Output: `activity_datasets/my_network_mnist_run1/activity_dataset.h5`

---

## 4. Prepare Data for Training

Extract features from the raw activity and split into train/test:

```bash
python -m snn_classification_realtime.prepare_activity_data \
  --input-file activity_datasets/my_network_mnist_run1 \
  --output-dir prepared_data \
  --feature-types firings avg_S \
  --train-split 0.8 \
  --scaler minmax
```

- `--input-file`: Path to the HDF5 dataset directory (or JSON file with `--legacy-json`)
- `--feature-types`: `firings`, `avg_S`, `avg_t_ref` (one or more)
- `--scaler`: `none`, `minmax`, `standard`, `maxabs`

Output: `prepared_data/my_network_mnist_run1_firings_avg_S/` with `.pt` files for the SNN trainer.

---

## 5. Train the SNN Classifier

Train the readout classifier on the prepared activity:

```bash
python -m snn_classification_realtime.train_snn_classifier \
  --dataset-dir prepared_data/my_network_mnist_run1_firings_avg_S \
  --output-dir models \
  --epochs 50 \
  --batch-size 64
```

The trainer creates a subdirectory under `models/` named `{dataset_basename}_e{epochs}_lr{lr}_b{batch_size}/`. The final model is saved as `model.pth` in that subdirectory.

Output: `models/my_network_mnist_run1_firings_avg_S_e50_lr0.001_b64/model.pth`

---

## 6. Run Real-Time Evaluation

Evaluate the trained classifier on live network simulation:

```bash
python -m snn_classification_realtime.realtime_classification \
  --snn-model-path models/<model_subdir>/model.pth \
  --neuron-model-path networks/my_network.json \
  --dataset-name mnist \
  --evaluation-mode \
  --eval-samples 1000 \
  --output-dir evals \
  --eval-output-suffix run1 \
  --ticks-per-image 50 \
  --window-size 80 \
  --think-longer \
  --bistability-rescue
```

| Argument               | Description                                                                  |
| ---------------------- | ---------------------------------------------------------------------------- |
| `--snn-model-path`     | Trained classifier (file or directory; if dir, auto-selects best checkpoint) |
| `--neuron-model-path`  | Original network JSON                                                        |
| `--dataset-name`       | `mnist`, `fashionmnist`, `cifar10`, `cifar10_color`, `cifar100`              |
| `--evaluation-mode`    | Automated testing (vs interactive mode)                                      |
| `--eval-samples`       | Number of test samples                                                       |
| `--output-dir`         | Output directory for JSONL and summary (default: `evals`)                    |
| `--eval-output-suffix` | Suffix for deterministic output filenames                                    |
| `--ticks-per-image`    | Simulation ticks per image                                                   |
| `--window-size`        | Ticks used as classifier input                                               |
| `--think-longer`       | Extend simulation time if predictions incorrect                              |
| `--bistability-rescue` | Consider correct if in top-2 with small confidence gap                       |

Output: `evals/<model_dir>_eval_<suffix>.jsonl` and `*_summary.json` with accuracy and per-sample results.

---

## 6.5. Analyze Evaluation Results (Optional)

Generate metrics JSON and Markdown from evaluation output:

```bash
python -m snn_classification_realtime.analyze_eval \
  --jsonl evals/<model_dir>_eval_run1.jsonl \
  --output-dir analysis/my_network_run1 \
  --num-classes 10 \
  --class-labels "0,1,2,3,4,5,6,7,8,9"
```

For CIFAR-10: `--class-labels "airplane,automobile,bird,cat,deer,dog,frog,horse,ship,truck"`

Output: `analysis/my_network_run1/metrics.json` and `metrics.md`

---

## 7. Visualizations (Optional)

The repository includes several visualization tools to explore network dynamics, activity patterns, and emergent structure. All require an activity dataset (Section 3) and optionally evaluation results (Section 6).

### 7.1 Activity Dataset Plots

Firing rates, membrane potentials, and time series per layer:

```bash
python visualize_activity_dataset.py activity_datasets/my_network_mnist_run1 \
  --plot firing_rate_per_layer \
  --out-dir viz/activity_dataset
```

Plot types: `firing_rate_per_layer`, `avg_S_per_layer_per_label`, `firings_time_series`, `avg_S_time_series`, `total_fired_cumulative`, `network_state_progression` (append `_3d` for 3D variants).

### 7.2 Network Activity (Plotly)

Heatmaps, spike rasters, attractor landscapes, phase portraits:

```bash
python visualize_network_activity.py \
  --input-file activity_datasets/my_network_mnist_run1 \
  --output-dir viz/network_activity \
  --num-classes 10 \
  --plots all
```

Use `--plots s_heatmap_by_class spike_raster layerwise_s_average` for a subset. Add `--skip-static-images` if Kaleido is not installed.

### 7.3 3D Brain Visualization

UMAP-projected neuron clustering in 3D (neurons that activate together cluster visually):

```bash
python -m snn_classification_realtime.viz.activity_3d.run \
  --network networks/my_network.json \
  --output-dir viz/activity_3d \
  --dataset mnist \
  --ticks 50 \
  --samples-per-class 5 \
  --clustering firings
```

### 7.4 Neuron Clustering

Cluster neurons by activity similarity (K-means, DBSCAN, or correlation-based):

```bash
python -m snn_classification_realtime.viz.cluster_neurons.run \
  --input-file activity_datasets/my_network_mnist_run1 \
  --output-dir viz/cluster_neurons \
  --clustering-mode fixed \
  --num-clusters 10 \
  --num-classes 10
```

### 7.5 Activity Clustering

Cluster activity patterns by feature vectors:

```bash
python -m snn_classification_realtime.viz.cluster_activity.run \
  --input-file activity_datasets/my_network_mnist_run1 \
  --output-dir viz/cluster_activity \
  --feature-types "firings,avg_S" \
  --clustering-mode fixed \
  --num-clusters 10 \
  --num-classes 10
```

### 7.6 Concept Hierarchy

Dendrograms from evaluation results (requires eval summary + JSONL):

```bash
python plot_concept_hierarchy.py \
  --json-file evals/<model_dir>_eval_run1_summary.json \
  --results-file evals/<model_dir>_eval_run1.jsonl \
  --output-dir viz/concept_hierarchy
```

### 7.7 Synaptic Analysis

Connectivity and weight analysis (requires `--export-network-states` when building the activity dataset):

```bash
python -m snn_classification_realtime.viz.synaptic_analysis.run \
  --network-states-dir network_state/my_network_mnist_run1 \
  --output-dir viz/synaptic_analysis \
  --method louvain \
  --n-clusters 5
```

### 7.8 Web Visualization (Live)

Enable real-time network visualization during evaluation by adding `--enable-web-server` to the realtime_classification command (Section 6). Opens http://127.0.0.1:5555 for live neuron firing and signal propagation.

Alternatively, run `python interactive_training.py` or `python cli/neuron_cli.py` and use the `web_viz` command for interactive exploration.

---

## 8. Alternative: Full Pipeline (Automated)

The pipeline runs all steps in sequence. Create a YAML config:

```yaml
job_name: reproducibility_run
network:
  source: build
  build_config:
    dataset: mnist
    layers:
      - type: conv
        kernel_size: 4
        stride: 2
        filters: 4
        connectivity: 0.8
      - type: conv
        kernel_size: 4
        stride: 2
        filters: 6
        connectivity: 0.8
      - type: dense
        size: 144
        connectivity: 0.25

activity_recording:
  dataset: mnist
  ticks_per_image: 50
  images_per_label: 100
  binary_format: true
  fresh_run_per_label: true

data_preparation:
  feature_types: [firings, avg_S]
  train_split: 0.8
  scaling_method: minmax

training:
  epochs: 50
  batch_size: 64

evaluation:
  samples: 1000
  window_size: 80
  think_longer: true
```

Run via Python:

```python
from pipeline.config import load_config
from pipeline.orchestrator import run_pipeline

config = load_config("pipeline_config.yaml")
job = run_pipeline(config, output_dir=Path("./experiments"))
print(f"Status: {job.status}")
```

Or submit via the API (see `pipeline/AGENT_GUIDE.md`).

---

## 9. Ablation Studies

To test ALERM components in isolation, use the `--ablation` flag (realtime) or select ablation when building the activity dataset:

| Ablation                     | Effect                                                                 |
| ---------------------------- | ---------------------------------------------------------------------- |
| `none`                       | Full model (Hebbian + homeostasis + predictive error). Baseline ~84.3% |
| `tref_frozen`                | Disables homeostatic regulation. ~80.4%                                |
| `weight_update_disabled`     | Hebbian weight updates off; homeostasis only                           |
| `retrograde_disabled`        | Retrograde signaling disabled                                          |
| `thresholds_frozen`          | Adaptive thresholds frozen                                             |
| `directional_error_disabled` | Directional error component disabled                                   |

---

## 10. File Layout Summary

```
neuron-model/
├── build_network.py              # Standalone network builder (YAML → JSON)
├── build_activity_dataset.py     # Shim to snn_classification_realtime.build_activity_dataset
├── interactive_training.py       # Interactive build + visualization
├── networks/                    # Network JSON files (create this dir)
├── activity_datasets/           # Raw activity (HDF5/JSON)
├── prepared_data/               # Extracted features for training
├── models/                      # Trained SNN classifiers
├── evals/                       # Evaluation results (JSONL + summary)
├── analysis/                    # Metrics from analyze_eval
├── viz/                         # Visualization outputs (optional)
├── snn_classification_realtime/
│   ├── build_activity_dataset.py
│   ├── prepare_activity_data.py
│   ├── train_snn_classifier.py
│   ├── realtime_classification.py
│   ├── analyze_eval.py
│   └── viz/                     # Visualization modules
├── visualize_activity_dataset.py
├── visualize_network_activity.py
├── visualize_activity_3d.py
├── plot_concept_hierarchy.py
├── scripts/                     # Example run scripts (e.g. 01_cifar10colored.sh)
└── pipeline/                    # Automated pipeline
```
