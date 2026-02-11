# Directory Structure

```
neuron-model/
├── .planning/                    # GSD workflow artifacts
│   └── codebase/                # This directory - codebase documentation
├── activity_datasets/           # Pre-recorded neural activity datasets
├── cli/                         # Command-line interface
│   ├── neuron_cli.py           # Main CLI entry point
│   └── web_viz/                # Real-time web visualization
│       ├── server.py           # Flask server with SocketIO
│       ├── config.py           # Visualization config
│       └── data_transformer.py # Data format conversion
├── clustering_results/          # Neuron clustering analysis outputs
├── data/                        # Raw data storage
├── evals/                       # Model evaluation results
├── models/                      # Trained model checkpoints
│   └── */model_config.json     # Model metadata
├── modality_processor/          # Multi-modal data processing
│   ├── core/                   # Core abstractions
│   │   ├── base_module.py      # Base module class
│   │   ├── data_packet.py      # Data packet structure
│   │   └── policy.py           # Processing policy
│   ├── modules/                # Modality-specific modules
│   │   ├── audio_module.py
│   │   ├── image_module.py
│   │   ├── text_module.py
│   │   └── video_module.py
│   ├── processor.py            # Main processor
│   ├── config.py               # Configuration
│   └── examples/               # Usage examples
├── networks/                    # Network topology definitions
│   └── *.json                  # Network configs
├── neuron/                      # Core neuron model
│   ├── __init__.py             # Package exports
│   ├── neuron.py               # Single neuron implementation
│   ├── network.py              # Multi-neuron network
│   ├── nn_core.py              # Neural network core
│   └── network_config.py       # Network configuration
├── neuron_clustering_results/   # Clustering analysis results
├── pipeline/                    # Experiment pipeline (FastAPI)
│   ├── api/                    # REST API
│   │   ├── main.py            # FastAPI app entry point
│   │   ├── database.py        # In-memory job storage
│   │   ├── models.py          # Pydantic models
│   │   ├── routes/            # API endpoints
│   │   │   ├── jobs.py        # Job CRUD
│   │   │   ├── artifacts.py   # Artifact retrieval
│   │   │   ├── webhooks.py    # Webhook configuration
│   │   │   ├── uploads.py     # Config upload
│   │   │   └── job_page.py    # Job detail pages
│   │   └── templates/         # HTML templates
│   ├── config.py              # Pipeline configuration
│   ├── orchestrator.py        # Job execution orchestrator
│   ├── steps/                 # Pipeline steps
│   │   ├── base.py            # Step base classes
│   │   ├── network_builder.py # Build network from config
│   │   ├── data_preparer.py   # Prepare training data
│   │   ├── activity_recorder.py # Record neural activity
│   │   ├── classifier_trainer.py # Train classifier
│   │   └── evaluator.py       # Evaluate performance
│   ├── visualizations/        # Visualization steps
│   │   ├── activity_dataset.py
│   │   ├── activity_3d.py
│   │   ├── cluster_activity.py
│   │   ├── cluster_neurons.py
│   │   ├── concept_hierarchy.py
│   │   ├── network_activity.py
│   │   └── synaptic_analysis.py
│   ├── utils/                 # Utilities
│   │   └── activity_data.py   # HDF5 activity data loading
│   ├── tests/                 # Pipeline tests
│   │   ├── test_api.py
│   │   ├── test_orchestrator.py
│   │   ├── test_config.py
│   │   ├── test_cancellation.py
│   │   ├── test_pause_cancel.py
│   │   ├── test_integration.py
│   │   └── test_zombie_cancel.py
│   ├── requirements.txt       # Pipeline dependencies
│   ├── pyproject.toml         # Modern Python config
│   ├── docker-compose.yml     # Docker configuration
│   └── example_config.yaml    # Example pipeline config
├── prepared_data/              # Pre-processed experiment data
│   └── */dataset_metadata.json # Dataset metadata
├── snn_classification_realtime/ # Real-time SNN classification demos
├── tests/                      # Core tests (minimal)
├── viz/                        # Visualization outputs
├── viz_network/                # Network visualizations
├── requirements.txt            # Core dependencies
├── interactive_mnist.py        # Interactive MNIST demo
├── interactive_training.py     # Interactive training script
├── neuron_visualizer.py        # Static neuron visualization
├── visualize_activity_3d.py    # 3D activity visualization
├── visualize_activity_dataset.py # Activity dataset visualization
└── visualize_network_activity.py # Network activity visualization
```

## Key Locations

### Entry Points
- `cli/neuron_cli.py` - Main CLI for direct neuron experimentation
- `pipeline/api/main.py` - FastAPI server for experiment pipeline
- `pipeline/orchestrator.py` - Pipeline job execution

### Core Model
- `neuron/neuron.py` - Single neuron implementation (graph-based)
- `neuron/network.py` - Multi-neuron network management
- `neuron/nn_core.py` - Neural network classification core

### Pipeline Steps
- `pipeline/steps/network_builder.py` - Build networks from JSON configs
- `pipeline/steps/data_preparer.py` - Prepare datasets (MNIST/CIFAR)
- `pipeline/steps/activity_recorder.py` - Record neural activity to HDF5
- `pipeline/steps/classifier_trainer.py` - Train classifiers on recorded data
- `pipeline/steps/evaluator.py` - Evaluate model performance

### API Routes
- `pipeline/api/routes/jobs.py` - Job lifecycle (create, start, pause, cancel)
- `pipeline/api/routes/artifacts.py` - Download job artifacts
- `pipeline/api/routes/webhooks.py` - Configure webhook notifications
- `pipeline/api/routes/uploads.py` - Upload experiment configs

### Configuration
- `pipeline/config.py` - PipelineConfig, WebhookConfig classes
- `pipeline/example_config.yaml` - Example pipeline configuration
- `networks/*.json` - Network topology definitions

### Data Storage
- `models/*/model_config.json` - Trained model metadata
- `prepared_data/*/dataset_metadata.json` - Prepared dataset metadata
- `activity_datasets/` - Pre-recorded neural activity

## Naming Conventions

### File Naming
- Snake case: `neuron_model.py`, `data_preparer.py`
- Descriptive: `activity_recorder.py` not `recorder.py`
- Test files: `test_*.py` prefix

### Directory Naming
- Snake case: `web_viz`, `activity_datasets`
- Plural for collections: `visualizations`, `routes`, `modules`
- Singular for core concepts: `neuron`, `network`, `processor`

### Class Naming
- PascalCase: `Neuron`, `NetworkTopology`, `PipelineJob`
- Dataclasses: `PostsynapticInputVector`, `PresynapticReleaseEvent`

### Function Naming
- Snake case: `prepare_data()`, `send_signal()`
- Verb-first for actions: `execute_job()`, `record_activity()`
- Noun-first for getters: `job_status()`, `neuron_count()`

### Constants
- UPPER_SNAKE_CASE: `MAX_MEMBRANE_POTENTIAL`, `MIN_SYNAPTIC_WEIGHT`

## Module Organization

### Single Responsibility
Each module has a focused purpose:
- `neuron/neuron.py` - Single neuron only
- `neuron/network.py` - Multi-neuron networks
- `pipeline/orchestrator.py` - Pipeline orchestration
- `pipeline/steps/` - Individual pipeline steps

### Layered Architecture
```
cli/           (User interaction)
    ↓
pipeline/      (Experiment orchestration)
    ↓
neuron/        (Core computation)
```

### Import Patterns
- `sys.path.insert(0, ...)` used in pipeline for importing neuron module
- Relative imports within packages: `from .core import base_module`
- Absolute imports from project root in pipeline steps