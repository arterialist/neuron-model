# PAULA & ALERM Reproducibility Guide

Welcome to the PAULA Reproducibility Guide.

The ALERM framework makes specific predictions evaluated via the PAULA empirical model. This document details the exact methodological pipeline used to reproduce the baseline performances (e.g., 84.3% accuracy) and the individual ablation studies observed in the paper.

## Empirical Methodology Overview

As described in the PAULA validation documents, the experimental workflow separates inference/learning (handled by the biological network) from the final evaluation mapping. The pipeline is:

1. **Network Initialization**: Building specific architectural constraints (e.g., 6 layers, 144 neurons, 25% sparse vs 100% dense connectivity).
2. **Activity Dataset Collection**: Feeding test samples (typically 1000 samples; 100 images per MNIST digit) through the network. During this phase, specific biological mechanisms (like homeostasis or Hebbian updates) are either active or **ablated**. The internal phase-space dynamics (spike rates, adaptive threshold states) are serialized.
3. **Readout Training**: A downstream Spiking Neural Network (SNN) classifier is trained strictly on the recorded activity states produced by the network to map the attractors to a measurable accuracy percentage.

---

## 1. Environment Setup

Ensure you have cloned the repository and installed the dependencies using `uv` (recommended for fast package resolution).

```bash
uv pip install -e .
```

---

## 2. Generating Ablated Activity Datasets

To measure the effects of ALERM's theoretical components (such as _Homeostatic Metaplasticity_ or _Hebbian Association_), we swap out the core rules during the simulated execution.

Run the dataset builder:

```bash
uv run python build_activity_dataset.py
```

The script will prompt you interactively. To match the paper's methodology:

- **Network File**: Select your baseline architecture (e.g., `networks/sparse_baseline.json`). You can construct custom architectures using the CLI/interaction scripts.
- **Ablation Parameter**: Select the isolated mechanism you wish to test:
  - `none` (Baseline: Hebbian + Homeostasis + Predictive Error. Matches 84.3% accuracy).
  - `tref_frozen` (Disables homeostatic regulation. Replicates catastrophic variance and ~80.4% performance drop).
  - `weight_update_disabled` (Validates network behavior with only homeostasis, disabling multiplicative weight association).
- **Number of images per label**: `100` (Yields the 1000 test samples specified in the paper's methodology).
- **Binary Format**: Select `Yes`/`True` to serialize straight to an HDF5 binary tensor format for optimal readout training speed.

This will generate an `activity_datasets/` folder containing the phase-space coordinates of the simulated network.

---

## 3. Training the SNN Readout & Evaluating Accuracy

Once the local processing is complete and the activity is recorded, train the readout classifier to measure the pattern separability and capacity of the network's attractors.

```bash
uv run python train_snn_classifier.py \
    --dataset-dir activity_datasets/<generated_dataset_folder> \
    --epochs 50 \
    --batch-size 64
```

The final output logged by the readout trainer will mirror the percentage accuracies (e.g., 84.3% mapping to the `none` baseline) present in the ALERM validation framework tables.

---

## 4. Measuring Attractor Variance & Convergence

A core claim of the validation is determining whether the network enters chaotic fluctuations or settles to a stable Limit Cycle (Attractor Variance).

These metrics (like the catastrophic `250.22` variance of frozen `t_ref` systems) are automatically captured. As `build_activity_dataset.py` processes samples and evaluates homeostasis ticks, the recorded evaluation JSON records will expose the convergence times (ticks) and variance rates necessary to build out stability tables.
