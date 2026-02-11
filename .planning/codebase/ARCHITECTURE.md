# Architecture

## High-Level Overview

The codebase is organized into three main architectural layers:

```
┌─────────────────────────────────────────────────────────────┐
│                   User Interface Layer                       │
│  ┌──────────────┐         ┌──────────────────┐              │
│  │     CLI      │         │  FastAPI Pipeline │              │
│  │  (click/cli) │         │     (REST API)   │              │
│  └──────────────┘         └──────────────────┘              │
│           │                           │                      │
└───────────┼───────────────────────────┼──────────────────────┘
            │                           │
┌───────────┼───────────────────────────┼──────────────────────┐
│           │    Experiment Layer       │                      │
│  ┌────────▼─────────────┐  ┌─────────▼──────────┐          │
│  │   Pipeline Steps     │  │  Visualization     │          │
│  │  (decorator pattern) │  │  (external scripts)│          │
│  └──────────────────────┘  └────────────────────┘          │
└─────────────────────────────────────────────────────────────┘
            │
┌───────────▼───────────────────────────────────────────────────┐
│                   Core Computation Layer                       │
│  ┌──────────────────────────────────────────────────────┐    │
│  │              Neuron Model (neuron/)                   │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐    │    │
│  │  │  Neuron  │  │ Network  │  │   NNCore         │    │    │
│  │  │ (single) │  │(multi)   │  │ (classification) │    │    │
│  │  └──────────┘  └──────────┘  └──────────────────┘    │    │
│  └──────────────────────────────────────────────────────┘    │
│  ┌──────────────────────────────────────────────────────┐    │
│  │         PyTorch/SNN-Torch Training Stack              │    │
│  └──────────────────────────────────────────────────────┘    │
└───────────────────────────────────────────────────────────────┘
```

## Core Architectural Patterns

### 1. Graph-Based Neuron Model

**Location**: `neuron/neuron.py`

The core neuron is implemented as a computational graph:
- **PostsynapticPoints (P'_in)**: Receiving connections with `PostsynapticInputVector`
- **PresynapticPoints (P'_out)**: Sending connections with `PresynapticOutputVector`
- **Axon Hillock**: Decision point for spike generation
- **Retrograde Signals**: Backpropagation-like error signals

**Key Classes**:
```python
Neuron                    # Main neuron class
  - postsynaptic_points   # Dict[int, PostsynapticPoint]
  - presynaptic_points    # Dict[int, PresynapticPoint]
  - dendrite_graph        # NetworkX graph
  - axon_hillock          # AxonHillock object
  - membrane              # Membrane object
```

**Event-Driven Communication**:
- `PresynapticReleaseEvent` - Forward signal propagation
- `RetrogradeSignalEvent` - Backward error propagation
- Events passed as tuples for optimization

### 2. Pipeline Step Pattern

**Location**: `pipeline/steps/base.py`

Experiments are composed of reusable steps using a decorator pattern:

```python
@step("data_preparer", depends_on=["network_builder"])
def prepare_data(context: StepContext, config: Dict) -> StepResult:
    # Step implementation
    return StepResult(status=StepStatus.SUCCESS, artifacts=[...])
```

**Key Components**:
- `StepContext` - Provides access to previous step artifacts
- `StepResult` - Contains status, artifacts, and metadata
- `StepRegistry` - Global registry of available steps
- `PipelineOrchestrator` - Manages step execution order

**Execution Flow**:
1. Steps declare dependencies via decorator
2. Orchestrator builds dependency graph
3. Steps execute in topological order
4. Artifacts flow between steps

### 3. Network Topology Management

**Location**: `neuron/network.py`

Multi-neuron networks use a topology-based approach:
- `NetworkTopology` - Manages neuron connections
- `TravelingSignal` - Represents signals in transit
- `NeuronNetwork` - Orchestrates multi-neuron simulation

**Connection Types**:
- Synapse connections between neurons
- Terminal points for signal routing
- Auto-connect functionality for topologies

### 4. Data Flow

**Experiment Pipeline Flow**:
```
User Config
    ↓
[1] network_builder    → NetworkConfig JSON
    ↓
[2] data_preparer      → Prepared dataset (HDF5)
    ↓
[3] activity_recorder  → Neural activity (HDF5)
    ↓
[4] classifier_trainer → Trained model (PyTorch checkpoint)
    ↓
[5] evaluator          → Metrics/evaluation results
    ↓
[Visualization Steps]  → Plots, 3D visualizations, analysis
```

**Neuron Simulation Flow**:
```
External Signal
    ↓
PresynapticReleaseEvent (to terminal)
    ↓
PostsynapticPoint processing
    ↓
Dendrite graph propagation
    ↓
Membrane potential integration
    ↓
Axon Hillock decision (spike?)
    ↓
PresynapticReleaseEvent (to connected neurons)
```

## Key Abstractions

### StepContext
- Encapsulates execution context for pipeline steps
- Provides access to: artifacts, config, job_id, output_dir
- Allows steps to communicate via artifacts

### Artifact
- Represents output from a step
- Has path, type, and metadata
- Stored in job-specific output directory

### PipelineJob
- Represents a single experiment execution
- Tracks status, step results, logs
- Supports pause/resume and cancellation

### NeuronParameters
- Dataclass configuration for neuron behavior
- Controls: membrane dynamics, plasticity, firing thresholds

## Entry Points

### CLI Entry Point
**File**: `cli/neuron_cli.py`
```python
@click.group()
def cli():
    # Main command group
```

### Pipeline API Entry Point
**File**: `pipeline/api/main.py`
```python
app = FastAPI(title="Neuron Pipeline API")
```

### Pipeline Orchestrator Entry Point
**File**: `pipeline/orchestrator.py`
```python
class PipelineOrchestrator:
    def execute_job(self, job: PipelineJob) -> None:
        # Execute pipeline steps
```

## State Management

### Job State
- Stored in memory (PipelineJob instances)
- Status transitions: PENDING → RUNNING → PAUSED → COMPLETED/FAILED/CANCELLED
- Thread-safe using Event objects

### Neuron State
- Stored in Neuron instances (in-memory during simulation)
- Can be serialized to JSON/HDF5 for analysis

### Experiment State
- Stored as artifacts in output directories
- Persisted across pipeline steps

## Visualization Architecture

### Real-Time Visualization (Web_Viz)
- Flask + SocketIO for live updates
- Client-side: Cytoscape.js for graph rendering
- WebSocket pushes neuron state changes

### Batch Visualization (Pipeline)
- External Python scripts executed via subprocess
- Output: PNG files, interactive Plotly HTML
- Runs as separate pipeline steps

## Integration Points

1. **CLI → Neuron Model**: Direct import and method calls
2. **Pipeline → Neuron Model**: Import via sys.path manipulation
3. **Pipeline → PyTorch**: Standard torch integration
4. **Webhook System**: External notification via HTTP POST
5. **HDF5 Storage**: Lazy loading of large activity datasets