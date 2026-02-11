# Testing

## Test Framework

### Primary Framework
- **pytest 7.0.0+** - Main test framework
- **pytest-asyncio 0.21.0+** - Async test support for API testing

### Test Discovery
Tests are discovered automatically by pytest via the `test_*.py` naming convention.

## Test Locations

### Pipeline Tests
**Directory**: `pipeline/tests/`

Test files cover:
- `test_api.py` - API endpoint testing
- `test_orchestrator.py` - Orchestrator behavior
- `test_config.py` - Configuration validation
- `test_cancellation.py` - Job cancellation logic
- `test_pause_cancel.py` - Pause and cancel interaction
- `test_integration.py` - End-to-end pipeline execution
- `test_zombie_cancel.py` - Edge case: cancelling zombie jobs
- `test_cancellation_repro.py` - Cancellation bug reproduction

### Core Tests
**Directory**: `tests/` (minimal, contains mainly placeholder tests)

### Integration Tests
Located in `pipeline/tests/test_integration.py` - runs full pipeline workflows.

## Test Patterns

### Pydantic Model Testing
```python
def test_job_create_request_validation():
    config = PipelineConfig(...)
    request = JobCreateRequest(config=config)
    assert request.job_name is None or isinstance(request.job_name, str)
```

### Orchestrator Testing
```python
def test_job_execution():
    orchestrator = PipelineOrchestrator()
    job = PipelineJob(...)
    orchestrator.execute_job(job)
    assert job.status == JobStatus.COMPLETED
```

### Cancellation Testing
```python
def test_job_cancellation():
    orchestrator = PipelineOrchestrator()
    job = PipelineJob(...)
    orchestrator.cancel_job(job.job_id)
    assert job.status == JobStatus.CANCELLED
```

### Async API Testing
```python
@pytest.mark.asyncio
async def test_create_job():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/jobs", json=...)
        assert response.status_code == 201
```

## Test Configuration

### pytest.ini
Located in `pipeline/tests/conftest.py`:
- Provides test fixtures
- Configures async test client
- Sets up test data

### Key Fixtures
```python
@pytest.fixture
def sample_config():
    return PipelineConfig(...)

@pytest.fixture
async def async_client():
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client
```

## Running Tests

### Run All Tests
```bash
cd pipeline
pytest
```

### Run Specific Test File
```bash
pytest tests/test_orchestrator.py
```

### Run With Verbose Output
```bash
pytest -v
```

### Run Specific Test
```bash
pytest tests/test_api.py::test_create_job
```

## Coverage

### Current Coverage
- Pipeline orchestrator: Well covered
- API endpoints: Moderate coverage
- Cancellation logic: Extensive coverage (multiple test files)
- Core neuron model: Minimal test coverage

### Coverage Tracking
Not currently configured with coverage.py, but tests exist for critical paths.

## Test Categories

### Unit Tests
- Individual step behavior
- Configuration validation
- Model creation and state transitions

### Integration Tests
- Full pipeline execution
- API end-to-end workflows
- Job lifecycle management

### Edge Case Tests
- Zombie job cancellation
- Pause/cancel interaction
- Webhook failure handling

## Testing Goals

### Current Focus
1. Pipeline orchestrator reliability
2. Job lifecycle (start, pause, cancel)
3. API endpoint behavior

### Future Needs
1. Core neuron model unit tests
2. Network topology validation
3. Visualization step testing
4. Performance regression tests
5. Numerical accuracy validation

## Test Data

### Mock Data
Tests use in-memory configurations rather than external files.

### Test Outputs
Test outputs written to temporary directories, cleaned up after tests.

## Continuous Integration

### CI Status
No CI configuration found in repository. Tests are run manually.

### Recommended Setup
Consider adding `.github/workflows/test.yml` for automated testing.