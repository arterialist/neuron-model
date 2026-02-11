# External Integrations

## Datasets

### MNIST
- **Location**: Automatically downloaded via torchvision
- **Format**: PyTorch Dataset object
- **Usage**: Primary benchmark for neuron model evaluation
- **File**: Handled in `pipeline/steps/data_preparer.py`

### CIFAR-10
- **Location**: Automatically downloaded via torchvision
- **Format**: PyTorch Dataset object
- **Usage**: Alternative benchmark for more complex patterns
- **File**: Handled in `pipeline/steps/data_preparer.py`

## Webhook Notifications

### Overview
The pipeline orchestrator sends HTTP POST notifications to external endpoints when jobs complete or fail.

### Configuration
- File: `pipeline/config.py` (WebhookConfig class)
- Supported event types:
  - `JOB_COMPLETED` - Job finished successfully
  - `JOB_FAILED` - Job encountered an error
  - `JOB_CANCELLED` - Job was cancelled by user
  - `STEP_STARTED` - A step began execution
  - `STEP_COMPLETED` - A step finished successfully
  - `STEP_FAILED` - A step failed

### Implementation
- File: `pipeline/orchestrator.py` (PipelineJob._send_webhook_notification method)
- Uses `requests` library for synchronous HTTP calls
- Headers:
  - `Content-Type: application/json`
  - `User-Agent: neuron-pipeline/0.1.0`

### Payload Format
```json
{
  "event_type": "JOB_COMPLETED",
  "job_id": "uuid-string",
  "job_name": "experiment-name",
  "timestamp": "2024-01-25T10:00:00Z",
  "status": "completed",
  "details": {
    "step_name": "data_preparer",
    "artifact_paths": ["/path/to/artifact1.h5"]
  }
}
```

## File System Integration

### HDF5 Storage
- **Library**: h5py 3.7.0+
- **Usage**: Neural activity recording and lazy loading
- **Key Files**:
  - `pipeline/steps/activity_recorder.py` - Recording neural activity to HDF5
  - `pipeline/utils/activity_data.py` - Lazy loading utilities
- **Structure**:
  - `activity_data` - Main dataset group
  - `neuron_ids` - Dataset of neuron identifiers
  - `layer_structure` - Dataset encoding layer membership
  - `tick` - Simulation timestep dimension

### JSON Config Files
- **Purpose**: Network configuration and model metadata
- **Examples**: `networks/*.json`, `models/*/model_config.json`
- **Library**: Standard json module + ijson for streaming large files

### YAML Config Files
- **Purpose**: Experiment pipeline configuration
- **Library**: pyyaml
- **Example**: `pipeline/example_config.yaml`

## System Integration

### Threading & Concurrency
- **Library**: Python threading module (Event, Lock)
- **Usage**: Job pause/cancel control in orchestrator
- **File**: `pipeline/orchestrator.py` (PipelineJob class)

### Subprocess Integration
- **Purpose**: Execute external visualization scripts
- **Usage**: Running visualization steps as separate processes
- **File**: `pipeline/steps/` (various visualization steps)

## No External APIs

The codebase is designed to run primarily offline with the following exceptions:
1. Webhook notifications (optional, user-configured)
2. Dataset downloads (torchvision handles caching)

All neural computation is local and self-contained.