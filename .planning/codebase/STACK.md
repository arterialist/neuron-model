# Technology Stack

## Core Technologies

### Primary Language
- **Python 3.11+** - Main implementation language for neuron model, CLI, and pipeline orchestrator

### Deep Learning & Neural Computation
- **PyTorch 2.0+** - Deep learning framework for spiking neural networks
- **snntorch 0.9.1+** - Spiking neural network simulation and training
- **torchvision 0.15.0+** - Computer vision utilities and datasets
- **scikit-learn 1.1.0+** - Machine learning utilities (clustering, dimensionality reduction)

### Numerical Computation
- **NumPy 2.0+** - Core numerical operations and array manipulation
- **SciPy 1.10.0+** - Scientific computing functions (optimization, integration, etc.)

### Data Analysis & Visualization
- **pandas 2.0+** - Data manipulation and analysis
- **matplotlib 3.7.0+** - Static plotting and visualization
- **seaborn 0.12.0+** - Statistical data visualization
- **plotly 5.13.0+** - Interactive web-based plotting
- **kaleido 1.1.0+** - Static image export for Plotly
- **umap-learn 0.5.0+** - Dimensionality reduction for activity visualization

### Networking & Graph
- **NetworkX 3.0+** - Graph-based neural network topology management

### Web & API Frameworks
- **FastAPI 0.100.0+** - REST API for experiment pipeline
- **Flask 3.1.1** - Web visualization server (legacy web_viz component)
- **Flask-SocketIO 5.5.1** - WebSocket support for real-time visualization
- **Flask-CORS 6.0.1** - Cross-origin resource sharing
- **uvicorn 0.23.0+** - ASGI server for FastAPI
- **websockets 15.0.1** - WebSocket client/server support
- **requests 2.28.0+** - HTTP client for webhook notifications
- **httpx 0.24.0+** - Async HTTP client

### Data Storage & Serialization
- **h5py 3.7.0+** - HDF5 file format for neural activity data
- **ijson 3.4.0+** - Streaming JSON parsing for large datasets
- **pyyaml 6.0.0+** - YAML configuration file parsing
- **python-multipart 0.0.6+** - Multipart form data support

### CLI & User Interface
- **loguru 0.7.0+** - Advanced logging with colored output
- **rich 13.7.1** - Terminal formatting and progress bars
- **prompt-toolkit 3.0.47** - Interactive CLI with command history
- **pygments** - Syntax highlighting for CLI output

### Development & Testing
- **pytest 7.0.0+** - Test framework
- **pytest-asyncio 0.21.0+** - Async test support
- **debugpy 1.8.0+** - Python debugger
- **tqdm 4.64.0+** - Progress bars for long-running operations

### Validation
- **pydantic 2.0.0+** - Data validation and settings management
- **jinja2 3.1.0+** - Template engine for API HTML responses

## Configuration

### Python Environment
- Minimum Python version: 3.11
- Package management: pip / uv (see `pipeline/pyproject.toml`)
- Virtual environments: Standard venv

### Project Configuration Files
- `requirements.txt` - Core project dependencies
- `pipeline/requirements.txt` - Pipeline-specific dependencies
- `cli/web_viz/requirements.txt` - Web visualization dependencies
- `pipeline/pyproject.toml` - Modern Python project configuration (uses hatchling)

### Environment-Specific Configuration
- `pipeline/config.py` - Runtime configuration classes (PipelineConfig, WebhookConfig)
- YAML config files in root and pipeline directories for experiment parameters

## Runtime Architecture

### Main Components
1. **Neuron Model Core** (`neuron/`) - Pure Python/NumPy neural simulation
2. **CLI** (`cli/`) - Interactive command-line interface
3. **Pipeline** (`pipeline/`) - FastAPI-based experiment orchestrator
4. **Web Visualization** (`cli/web_viz/`) - Real-time network visualization via Flask+SocketIO

### Execution Contexts
- **Direct execution**: Running neuron simulations as Python scripts
- **CLI mode**: Interactive command-line experimentation
- **Pipeline mode**: Orchestrated multi-step experiments via FastAPI
- **Docker**: Pipeline containerization (see `pipeline/docker-compose.yml`)

## Key Design Choices

### Why NumPy over Torch for Neuron Core?
The neuron model core (`neuron/neuron.py`) uses NumPy for fine-grained control over numerical operations and explicit temporal simulation, while PyTorch is used for training and batched operations.

### Dual Web Framework Approach
- **FastAPI** for the experiment pipeline (modern async, type-safe)
- **Flask** for the web visualization (legacy component with SocketIO for real-time updates)

### HDF5 for Neural Activity
Neural activity data is stored in HDF5 format (`h5py`) for efficient binary storage with lazy loading support, enabling analysis of large datasets.