# Areas of Concern

## Technical Debt

### Minimal Core Test Coverage
**Location**: `tests/` directory

The core neuron model (`neuron/neuron.py`, `neuron/network.py`) lacks comprehensive unit tests. This is a concern because:
- The neuron model is the foundation of the entire system
- Numerical bugs could invalidate research results
- Refactoring is risky without test coverage

**Recommendation**: Add unit tests for:
- Neuron membrane potential dynamics
- Synaptic plasticity rules
- Signal propagation correctness
- Network topology consistency

### Path Manipulation for Imports
**Location**: `pipeline/orchestrator.py`, various pipeline steps

```python
sys.path.insert(0, str(Path(__file__).parent.parent))
```

This pattern is used throughout the pipeline to import the neuron module. This is fragile and:
- Breaks if project structure changes
- Causes confusion about import paths
- Makes testing more difficult

**Recommendation**: Install the project as an editable package or restructure imports.

### Dual Web Framework Approach
**Location**: `cli/web_viz/` (Flask) and `pipeline/api/` (FastAPI)

The codebase uses both Flask and FastAPI for web interfaces:
- Flask + SocketIO for real-time visualization (legacy)
- FastAPI for the experiment pipeline (modern)

This creates:
- Dependency bloat (both frameworks required)
- Confusion about which to use for new features
- Increased learning curve for contributors

**Recommendation**: Consolidate to a single framework when possible.

## Known Issues

### Cancellation Edge Cases
**Location**: `pipeline/tests/test_cancellation.py`, `test_zombie_cancel.py`, `test_pause_cancel.py`

Multiple test files exist for cancellation behavior, indicating:
- Complexity in cancellation logic
- Potential race conditions in pause/cancel interaction
- "Zombie" jobs that don't properly clean up

**Current Status**: Tests exist but underlying issues may remain.

### Large JSON Files in Root
**Location**: Root directory contains many large `.json` files (e.g., `mnist_3L.json`, `conv_2l_*.json`)

These files:
- Clutter the root directory
- Are committed to git (increasing repo size)
- Mix experiment artifacts with source code

**Recommendation**: Move to dedicated `experiments/` or `data/` directories, add to `.gitignore` with LFS for large files.

### Inconsistent Configuration Formats
Both YAML and JSON are used for configuration:
- YAML for pipeline configs (`pipeline/example_config.yaml`)
- JSON for network configs (`networks/*.json`)

This creates:
- Two configuration systems to learn
- Potential confusion for users
- No unified config validation

**Recommendation**: Standardize on one format or create a unified config loader.

## Performance Concerns

### No Async in Neuron Simulation
**Location**: `neuron/neuron.py`, `neuron/network.py`

The neuron model uses synchronous simulation which may:
- Limit scalability for large networks
- Block the event loop in pipeline steps
- Cause responsiveness issues in CLI

**Consideration**: This may be intentional for precise temporal control, but async could be explored for specific use cases.

### HDF5 Lazy Loading Not Always Used
**Location**: `pipeline/steps/activity_recorder.py`, visualization steps

Some visualization steps may load entire datasets into memory rather than using lazy loading:
- Could cause memory issues with large experiments
- Inefficient for large-scale analysis

**Recommendation**: Audit all data loading code for lazy loading opportunities.

## Security Considerations

### Webhook URL Validation
**Location**: `pipeline/config.py`, `orchestrator.py`

Webhook notifications use user-provided URLs:
- No validation of URL schemes
- Potential for SSRF (Server-Side Request Forgery)
- Could be used to probe internal networks

**Recommendation**: Validate webhook URLs, whitelist allowed domains.

### File Upload Validation
**Location**: `pipeline/api/routes/uploads.py`

Config file uploads should:
- Validate file content before processing
- Limit file sizes
- Scan for malicious content (YAML injection, etc.)

**Current Status**: Validation may be insufficient.

### No Authentication/Authorization
**Location**: `pipeline/api/`

The FastAPI has no built-in authentication:
- Anyone can create jobs
- Anyone can cancel jobs
- No access control

**Recommendation**: Add authentication for multi-user deployments.

## Maintainability

### Generated/Boilerplate Code
**Location**: `pipeline/.venv/` dependencies

The repository includes virtual environments (`.venv/`, `pipeline/.venv/`):
- Should be in `.gitignore`
- Increases repo size significantly
- Causes environment-specific conflicts

**Current Status**: These directories are gitignored, but may exist locally.

### Inconsistent Error Handling
Some functions raise exceptions, others return error values:
- Makes error handling inconsistent
- Unclear when to use which pattern

**Recommendation**: Establish and document error handling conventions.

## Documentation Gaps

### Missing Docstrings
**Location**: Various files, especially in `neuron/`

Many functions lack docstrings:
- Makes code harder to understand
- Increases learning curve for new contributors

**Example**: `neuron/neuron.py` has minimal docstrings for complex methods.

### Architecture Documentation
**Location**: No dedicated architecture documentation

High-level design decisions are not documented:
- Why the specific neuron model was chosen
- Trade-offs in the pipeline design
- Future roadmap and technical vision

**Recommendation**: Add `ARCHITECTURE.md` or similar documentation.

## Fragile Areas

### Subprocess Visualization Execution
**Location**: `pipeline/visualizations/` (various files)

Visualization steps execute external Python scripts via subprocess:
- Process management complexity
- Harder to debug failures
- Platform-dependent behavior

**Current Status**: Works but adds complexity.

### Hardcoded Constants
**Location**: `neuron/neuron.py`

```python
MAX_MEMBRANE_POTENTIAL = 20.0
MIN_MEMBRANE_POTENTIAL = -20.0
```

These constants are hardcoded:
- May need adjustment for different use cases
- Not easily configurable
- No explanation of biological relevance

**Recommendation**: Move to configurable parameters with documentation.

## Future Considerations

### Scalability
The current architecture may not scale to:
- Very large networks (10k+ neurons)
- Distributed execution
- Real-time processing constraints

### GPU Acceleration
PyTorch is used for training but the neuron simulation is CPU-bound:
- Could benefit from GPU acceleration
- Requires significant refactoring
- May sacrifice temporal precision

### Data Management
As experiments grow:
- HDF5 file management becomes complex
- Metadata organization needs improvement
- Automated cleanup of old results