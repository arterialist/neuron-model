# Experiment Pipeline - Agent Guide

This guide documents the automated experimentation pipeline for AI agent maintenance.

## Overview

The `pipeline/` package automates the neural network experimentation workflow:

1. **Network Building** - Create or load neural networks
2. **Activity Recording** - Record network responses to datasets
3. **Data Preparation** - Extract features for classifier training
4. **Classifier Training** - Train SNN classifiers on activity data
5. **Evaluation** - Evaluate classifiers on new data

## Architecture

```
pipeline/
├── config.py          # YAML configuration parsing (Pydantic)
├── orchestrator.py    # Pipeline execution engine
├── steps/             # Individual pipeline steps
│   ├── base.py        # Abstract step interface
│   ├── network_builder.py
│   ├── activity_recorder.py
│   ├── data_preparer.py
│   ├── classifier_trainer.py
│   └── evaluator.py
├── api/               # FastAPI web application
│   ├── main.py        # App entry point
│   ├── database.py    # SQLite persistence
│   └── routes/        # API endpoints
└── tests/             # pytest test suite
```

## Quick Start

### Running the API Server

```bash
cd /Users/arterialist/Projects/agi-research/neuron-model/pipeline
python -m api.main
```

Server runs at `http://localhost:8000`. Web UI available at root path.

### Submitting a Job

```bash
curl -X POST http://localhost:8000/api/jobs \
  -H "Content-Type: application/json" \
  -d '{"config_yaml": "job_name: test\nnetwork:\n  source: file\n  path: conv_2l_9k3s4k2s_mnist.json"}'
```

### Running via Python

```python
from pipeline.config import load_config
from pipeline.orchestrator import run_pipeline

config = load_config("example_config.yaml")
job = run_pipeline(config)
print(f"Job {job.job_id} status: {job.status}")
```

## Configuration Schema

See `example_config.yaml` for full example. Key sections:

| Section              | Purpose                            |
| -------------------- | ---------------------------------- |
| `network`            | Network source (`file` or `build`) |
| `activity_recording` | Dataset, ticks per image, samples  |
| `data_preparation`   | Feature types, train/test split    |
| `training`           | Epochs, batch size, learning rate  |
| `evaluation`         | Samples, window size               |
| `webhooks`           | URLs for status notifications      |

## Visualization Types

The pipeline supports 7 visualization types, each delegating to `snn_classification_realtime/viz/` reference implementations:

| Type               | Reference Implementation              | Description                           |
| ------------------ | ------------------------------------- | ------------------------------------- |
| `activity_dataset` | `snn_classification_realtime/viz/activity_dataset` | Firing rates, S values, time series   |
| `network_activity` | `snn_classification_realtime/viz/network_activity` | Heatmaps, spike rasters, correlations |
| `activity_3d`      | `snn_classification_realtime/viz/activity_3d`      | 3D brain visualization with UMAP      |
| `cluster_neurons`  | `snn_classification_realtime/viz/cluster_neurons`  | Neuron clustering (fixed/auto/synchrony) |
| `cluster_activity` | `snn_classification_realtime/viz/cluster_activity` | Activity pattern clustering           |
| `concept_hierarchy`| `snn_classification_realtime/viz/concept_hierarchy`| Dendrograms from evaluation results   |
| `synaptic_analysis`| `snn_classification_realtime/viz/synaptic_analysis`| Connectivity cluster analysis (requires export_network_states) |

### Available Plots

**activity_dataset:**

- `firing_rate_per_layer`, `firing_rate_per_layer_3d`
- `avg_S_per_layer_per_label`, `avg_S_per_layer_per_label_3d`
- `firings_time_series`, `firings_time_series_3d`
- `avg_S_time_series`, `avg_S_time_series_3d`
- `total_fired_cumulative`, `total_fired_cumulative_3d`

**network_activity:**

- `heatmap_S`, `heatmap_firing`
- `spike_raster`, `layer_activity`
- `neuron_importance`, `correlation_matrix` |

## Adding New Steps

1. Create `pipeline/steps/new_step.py`
2. Extend `PipelineStep` base class
3. Implement `name` property and `run()` method
4. Decorate with `@StepRegistry.register`
5. Add step name to `Orchestrator.PIPELINE_STEPS`
6. Add config model to `pipeline/config.py`

## API Endpoints

| Method | Endpoint                                       | Description             |
| ------ | ---------------------------------------------- | ----------------------- |
| GET    | `/api/jobs`                                    | List all jobs           |
| POST   | `/api/jobs`                                    | Create new job          |
| GET    | `/api/jobs/{id}`                               | Get job details         |
| DELETE | `/api/jobs/{id}`                               | Cancel job              |
| GET    | `/api/jobs/{id}/steps/{step}/artifacts.tar.gz` | Download step artifacts |
| GET    | `/api/artifacts/{id}/all.tar.gz`               | Download all artifacts  |

## Running Tests

```bash
# Unit tests only
cd pipeline
pytest tests/ -v

# Include integration tests (slower)
pytest tests/ -v --integration
```

## Docker Deployment

```bash
cd /Users/arterialist/Projects/agi-research/neuron-model/pipeline
docker compose build
docker compose up -d

# Check logs
docker compose logs -f
```

## Environment Variables

| Variable              | Default         | Description          |
| --------------------- | --------------- | -------------------- |
| `PIPELINE_OUTPUT_DIR` | `./experiments` | Job output directory |
| `PIPELINE_DB_PATH`    | `pipeline.db`   | SQLite database path |

## Troubleshooting

### Import Errors

Ensure parent directory is in Python path:

```python
import sys
sys.path.insert(0, "/Users/arterialist/Projects/agi-research/neuron-model")
```

### Missing Dependencies

```bash
pip install -r pipeline/requirements.txt
```

### snntorch Required

The classifier trainer and evaluator require snntorch:

```bash
pip install snntorch
```
