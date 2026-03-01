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

---

## 1. Environment Setup

```bash
git clone https://github.com/arterialist/neuron-model.git
cd neuron-model

# Install with uv (recommended) or pip
uv pip install -e .
# or: pip install -e .
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
python build_network.py --config my_config.yaml
# Output: networks/my_network.json
```

### Option B: Interactive training (build + explore)

Use `interactive_training.py` to build a network interactively, visualize it, and optionally save:

```bash
python interactive_training.py
```

Follow prompts to select dataset (MNIST, CIFAR10, etc.), build the network, and save to a JSON file.

### Option C: Pipeline (build from YAML)

The pipeline can build networks from embedded config. See Section 7.

---

## 3. Build Activity Dataset

Feed test images through the network and record internal dynamics (spike rates, membrane potentials, adaptive thresholds).

```bash
python build_activity_dataset.py \
  --network-path networks/my_network.json \
  --dataset-name mnist \
  --ablation none \
  --ticks-per-image 50 \
  --images-per-label 100 \
  --output-dir activity_datasets
```

Use `--dry-run` to print recommended ticks per image without recording.

Output: `activity_datasets/<name>_<dataset>_<timestamp>/activity_dataset.h5`

---

## 4. Prepare Data for Training

Extract features from the raw activity and split into train/test:

```bash
python snn_classification_realtime/prepare_activity_data.py \
  --input-file activity_datasets/<your_dataset_folder> \
  --output-dir prepared_data \
  --feature-types firings avg_S \
  --train-split 0.8 \
  --scaler minmax
```

- `--input-file`: Path to the HDF5 directory (or JSON with `--legacy-json`)
- `--feature-types`: `firings`, `avg_S`, `avg_t_ref` (one or more)
- `--scaler`: `none`, `minmax`, `standard`, `maxabs`

Output: `prepared_data/` with `.pt` files for the SNN trainer.

---

## 5. Train the SNN Classifier

Train the readout classifier on the prepared activity:

```bash
python snn_classification_realtime/train_snn_classifier.py \
  --dataset-dir prepared_data \
  --output-dir models \
  --epochs 50 \
  --batch-size 64
```

Output: `models/snn_model.pth` (or path specified by `--model-save-path`).

---

## 6. Run Real-Time Evaluation

Evaluate the trained classifier on live network simulation:

```bash
python snn_classification_realtime/realtime_classification.py \
  --snn-model-path models/snn_model.pth \
  --neuron-model-path networks/my_network.json \
  --dataset-name mnist \
  --evaluation-mode \
  --eval-samples 1000 \
  --ticks-per-image 50 \
  --window-size 80
```

| Argument | Description |
|----------|-------------|
| `--snn-model-path` | Trained classifier |
| `--neuron-model-path` | Original network JSON |
| `--dataset-name` | `mnist`, `fashionmnist`, `cifar10`, `cifar10_color`, `cifar100` |
| `--evaluation-mode` | Automated testing (vs interactive mode) |
| `--eval-samples` | Number of test samples |
| `--ticks-per-image` | Simulation ticks per image |
| `--window-size` | Ticks used as classifier input |

Output: `evals/<model_dir>_eval_<timestamp>.jsonl` and `*_summary.json` with accuracy and per-sample results.

---

## 7. Alternative: Full Pipeline (Automated)

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

## 8. Ablation Studies

To test ALERM components in isolation, use the `--ablation` flag (realtime) or select ablation when building the activity dataset:

| Ablation | Effect |
|----------|--------|
| `none` | Full model (Hebbian + homeostasis + predictive error). Baseline ~84.3% |
| `tref_frozen` | Disables homeostatic regulation. ~80.4% |
| `weight_update_disabled` | Hebbian weight updates off; homeostasis only |
| `retrograde_disabled` | Retrograde signaling disabled |
| `thresholds_frozen` | Adaptive thresholds frozen |
| `directional_error_disabled` | Directional error component disabled |

---

## 9. File Layout Summary

```
neuron-model/
├── build_network.py              # Standalone network builder (YAML → JSON)
├── interactive_training.py       # Interactive build + visualization
├── networks/                    # Network JSON files (create this dir)
├── activity_datasets/            # Raw activity (HDF5/JSON)
├── prepared_data/               # Extracted features for training
├── models/                      # Trained SNN classifiers
├── evals/                       # Evaluation results (JSONL + summary)
├── snn_classification_realtime/
│   ├── build_activity_dataset.py
│   ├── prepare_activity_data.py
│   ├── train_snn_classifier.py
│   └── realtime_classification.py
└── pipeline/                    # Automated pipeline
```

---

## 10. Reproducing Paper Results

For the reported **84.3% MNIST accuracy**:

1. Use a **25% sparse** dense layer (e.g. `connectivity: 0.25`)
2. Use **adaptive t_ref** (ablation `none`)
3. **100 images per label** (1000 test samples)
4. **50+ ticks** per image for propagation
5. Feature types: `firings` and `avg_S`
6. Train readout for **50 epochs** with minmax scaling

Dense (100%) connectivity leads to chaotic dynamics; sparsity is required for stable attractors.
