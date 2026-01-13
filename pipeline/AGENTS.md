# SNN Experimentation Pipeline Documentation

This directory contains the automated pipeline for Spiking Neural Network (SNN) experimentation. It replaces manual script execution with a unified, configuration-driven workflow managed by a central runner and web interface.

## Architecture

The pipeline consists of:
1.  **Configuration:** Pydantic models in `config.py` define the schema for experiments.
2.  **Steps:** Standalone scripts (refactored from original tools) for each stage of the pipeline.
3.  **Runner:** `runner.py` orchestrates the steps, passing data between them.
4.  **Server:** `server/main.py` provides a FastAPI backend and static frontend for monitoring.

### Pipeline Steps

1.  **Step 1: Build Network (`step1_build_network.py`)**
    *   **Input:** Network configuration (layers, connectivity).
    *   **Output:** `network.json` (neuron topology).
2.  **Step 2: Record Activity (`step2_record_activity.py`)**
    *   **Input:** Simulation config, `network.json`.
    *   **Output:** `activity_dataset.h5` (HDF5 recording of spikes/potentials).
    *   **Visualization:** Triggers `visualize_network_activity.py` for static plots (raster, rates).
3.  **Step 3: Prepare Data (`step3_prepare_data.py`)**
    *   **Input:** `activity_dataset.h5`.
    *   **Output:** PyTorch tensors (`train_data.pt`, `test_data.pt`, etc.) and `scaler.pt`.
4.  **Step 4: Train Model (`step4_train_model.py`)**
    *   **Input:** Prepared PyTorch datasets.
    *   **Output:** Trained model (`model.pth`), training logs/graphs.
5.  **Step 5: Evaluate Model (`step5_evaluate.py`)**
    *   **Input:** `model.pth`, `network.json`.
    *   **Output:** Evaluation metrics (`evaluation_summary.json`), confusion matrices.

## Usage

### 1. Web Interface (Recommended)

Start the orchestration server:
```bash
uvicorn pipeline.server.main:app --host 0.0.0.0 --port 8000
```
Visit `http://localhost:8000` to:
*   Upload a YAML configuration file to start a job.
*   View running jobs and their status.
*   Read live logs.
*   Download artifacts (plots, models, logs).

### 2. Docker (Recommended for Production)

Build and run the containerized pipeline:
```bash
docker-compose up --build
```
This mounts local `experiments/`, `configs/`, and `data/` directories for persistence.
Access the web interface at `http://localhost:8000`.

### 3. CLI Runner

Run a specific experiment manually:
```bash
python pipeline/runner.py --config path/to/config.yaml
```
Output will be generated in `experiments/<job_name>_<timestamp>/`.

## Configuration

Experiments are defined in YAML files. Example structure:

```yaml
name: my_experiment
network:
  layers:
    - layer_type: conv
      filters: 0
      kernel_size: 9
      stride: 3
  inhibitory_signals: false

simulation:
  ticks_per_image: 20
  images_per_label: 50
  dataset_name_base: "mnist_exp"

preparation:
  feature_types: ["avg_S", "firings"]
  train_split: 0.8

training:
  epochs: 20
  learning_rate: 0.0005

evaluation:
  eval_samples: 100
  think_longer: true

visualizations:
  network_activity:
    enabled: true
    params:
      plots: ["spike_raster", "firing_rate_hist_by_layer"]
```

## Directory Structure

*   `pipeline/`: Root of the package.
    *   `config.py`: Pydantic schemas.
    *   `runner.py`: Main orchestration logic.
    *   `step*.py`: Individual step implementations.
    *   `visualization/`: Plotting scripts.
    *   `server/`: FastAPI backend and static frontend.
        *   `main.py`: API endpoints.
        *   `static/`: HTML/JS UI.

## Maintenance Notes for AI Agents

*   **Adding a Step:**
    1.  Create `stepN_new_task.py`.
    2.  Add configuration model in `config.py`.
    3.  Import and call the step in `runner.py`.
    4.  Update `requirements.txt` if needed.
*   **Modifying Visualizations:**
    *   Visualization scripts are in `pipeline/visualization/`.
    *   They are called via `subprocess` in `runner.py` to ensure isolation.
    *   Ensure any new arguments are handled in `runner.py`'s argument construction logic.
*   **Server Updates:**
    *   The frontend is simple HTML/JS in `server/static/index.html`. No build step required.
    *   Job state is currently in-memory. For persistence, implement a SQLite database in `server/main.py`.

## Troubleshooting

*   **Job Fails Immediately:** Check `pipeline.log` in the job directory. If missing, check the server's stdout/stderr.
*   **Visualization Error:** Ensure `kaleido` is installed for static image generation. Check `viz_*.log` or `pipeline.log`.
*   **Dependency Issues:** Use `uv pip install -r requirements.txt` to sync environment.
