# Code Conventions

## Python Style

### Type Hints
Type hints are used consistently for function signatures:
```python
def execute_step(self, step_name: str, context: StepContext) -> StepResult:
    ...

def send_signal(self, neuron_id: int, terminal_id: int, signal: float) -> None:
    ...
```

### Dataclasses
Used for structured data with type safety:
```python
@dataclass
class PostsynapticInputVector:
    info: float
    plast: float
    adapt: np.ndarray

@dataclass
class PipelineJob:
    job_id: str
    config: PipelineConfig
    status: JobStatus
```

### __slots__ Usage
Optimized classes use `__slots__` to reduce memory:
```python
class PresynapticReleaseEvent:
    __slots__ = ["source_neuron_id", "source_terminal_id", "signal_vector", "timestamp"]
```

## Error Handling

### Exceptions
Custom exceptions for specific failure modes:
```python
class StepCancelledException(Exception):
    """Raised when a step is cancelled."""
    pass
```

### Logging
Uses `loguru` for colored, structured logging:
```python
from loguru import logger

logger.info("Processing neuron {}", neuron_id)
logger.warning("Potential issue detected")
logger.error("Failed to load config: {}", str(e))
```

Neuron-specific logger with context:
```python
logger = logger.bind(neuron_int=self.neuron_int, neuron_hex=self.neuron_hex)
logger.info("Signal received at terminal {}", terminal_id)
```

## Pipeline Step Conventions

### Step Decorator
All pipeline steps use the `@step` decorator:
```python
@step("data_preparer", depends_on=["network_builder"])
def prepare_data(context: StepContext, config: Dict) -> StepResult:
    ...
```

### Step Context Usage
Steps access artifacts through the context:
```python
def prepare_data(context: StepContext, config: Dict) -> StepResult:
    network_config = context.get_artifact("network_config")
    ...
```

### Step Result Pattern
All steps return `StepResult`:
```python
return StepResult(
    status=StepStatus.SUCCESS,
    artifacts=[Artifact(path=output_path, type="hdf5")],
    metadata={"samples": n_samples}
)
```

### Cancellation Handling
Steps check for cancellation:
```python
context.check_cancelled()
# Long-running work
context.check_cancelled()
```

## Neuron Model Conventions

### Event Representation
Events use tuples for performance:
```python
NeuronEvent = Union[PresynapticReleaseEvent, RetrogradeSignalEvent, Tuple[int, int, float]]
```

### Graph Operations
NetworkX for graph management:
```python
import networkx as nx
self.dendrite_graph = nx.DiGraph()
self.dendrite_graph.add_edge(presynaptic_id, postsynaptic_id)
```

### Numerical Stability
Constants for bounds checking:
```python
MAX_MEMBRANE_POTENTIAL = 20.0
MIN_MEMBRANE_POTENTIAL = -20.0
MAX_SYNAPTIC_WEIGHT = 2.0
```

## CLI Conventions

### Click Groups
```python
@click.group()
def cli():
    """Neuron Model CLI"""
    pass

@cli.command()
@click.argument("config_file")
def run(config_file: str):
    """Run an experiment"""
    ...
```

## API Conventions

### Pydantic Models
All request/response models use Pydantic:
```python
from pydantic import BaseModel

class JobCreateRequest(BaseModel):
    config: PipelineConfig
    job_name: Optional[str] = None
```

### Route Organization
Routes grouped by resource:
- `pipeline/api/routes/jobs.py` - Job endpoints
- `pipeline/api/routes/artifacts.py` - Artifact endpoints
- `pipeline/api/routes/webhooks.py` - Webhook endpoints

## Import Patterns

### Pipeline Imports
Due to path requirements, pipeline uses explicit path manipulation:
```python
sys.path.insert(0, str(Path(__file__).parent.parent))
```

### Relative Imports in Packages
```python
from .core import base_module
from .modules import image_module
```

## Documentation

### Docstrings
Google-style docstrings used for complex functions:
```python
def execute_job(self, job: PipelineJob) -> None:
    """
    Execute a pipeline job with all configured steps.

    Args:
        job: The pipeline job to execute.

    Raises:
        StepCancelledException: If the job is cancelled.
    """
```

### Comments
Used sparingly for complex logic:
```python
# OPTIMIZATION: Allow tuples for lightweight event passing
NeuronEvent = Union[..., Tuple[int, int, float]]
```

## Concurrency

### Threading
Uses `threading.Event` for pause/cancel control:
```python
self._pause_event = Event()
self._cancel_event = Event()

# In step execution:
if self._cancel_event.is_set():
    raise StepCancelledException()
```

## Configuration

### YAML Format
Pipeline configs use YAML:
```yaml
network:
  layers: [784, 128, 64, 10]
  connectivity: sparse
training:
  epochs: 100
  batch_size: 128
```

### JSON Format
Network configs use JSON:
```json
{
  "neurons": [
    {"id": 1, "num_synapses": 4, "num_terminals": 2}
  ]
}
```

## File Organization

### Grouped Imports
Standard library first, then third-party, then local:
```python
import sys
import json
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from pipeline.config import PipelineConfig
from pipeline.steps.base import StepContext
```

## Testing Conventions

### Pytest Tests
Test files named `test_*.py`:
```python
def test_job_creation():
    job = PipelineJob(job_id="test", config=..., output_dir=...)
    assert job.status == JobStatus.PENDING
```

### Async Tests
Use `pytest-asyncio`:
```python
@pytest.mark.asyncio
async def test_api_create_job():
    response = await client.post("/jobs", ...)
    assert response.status_code == 200
```