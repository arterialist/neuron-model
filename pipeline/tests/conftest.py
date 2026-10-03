"""
Pytest fixtures for pipeline tests.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# The API constructs persistent components during import, before fixtures run.
# Always redirect every storage layer before test modules can import it.
_bootstrap_storage = tempfile.TemporaryDirectory(prefix="neuron-pipeline-tests-")
_bootstrap_root = Path(_bootstrap_storage.name)
os.environ["PIPELINE_DB_PATH"] = str(_bootstrap_root / "pipeline.db")
os.environ["PIPELINE_DB_URL"] = f"sqlite:///{_bootstrap_root / 'pipeline.db'}"
os.environ["PIPELINE_API_DB_PATH"] = str(_bootstrap_root / "api.db")
os.environ["PIPELINE_OUTPUT_DIR"] = str(_bootstrap_root / "experiments")


@pytest.fixture(autouse=True)
def isolated_pipeline_storage(tmp_path, monkeypatch):
    """Prevent tests from restoring jobs or writing into research workspaces."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from pipeline import database
    from pipeline.api.database import Database
    from pipeline.api.routes import webhooks
    from pipeline.orchestrator import Orchestrator

    monkeypatch.chdir(tmp_path)
    db_path = tmp_path / "pipeline.db"
    db_url = f"sqlite:///{db_path}"
    api_db_path = tmp_path / "api.db"
    output_dir = tmp_path / "experiments"
    monkeypatch.setenv("PIPELINE_DB_PATH", str(db_path))
    monkeypatch.setenv("PIPELINE_DB_URL", db_url)
    monkeypatch.setenv("PIPELINE_API_DB_PATH", str(api_db_path))
    monkeypatch.setenv("PIPELINE_OUTPUT_DIR", str(output_dir))

    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    sessions = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    monkeypatch.setattr(database, "DATABASE_URL", db_url)
    monkeypatch.setattr(database, "DB_PATH", str(db_path), raising=False)
    monkeypatch.setattr(database, "engine", engine)
    monkeypatch.setattr(database, "SessionLocal", sessions)
    database.Base.metadata.create_all(bind=engine)

    api_database = Database(str(api_db_path))
    monkeypatch.setattr(webhooks, "_db", api_database)
    api_main = sys.modules.get("pipeline.api.main")
    if api_main is not None:
        monkeypatch.setattr(api_main, "BASE_OUTPUT_DIR", output_dir)
        monkeypatch.setattr(api_main, "DB_PATH", str(api_db_path))
        monkeypatch.setattr(api_main, "database", api_database)

    # Join every test-created executor before disposing its database or cwd.
    orchestrators = []
    original_init = Orchestrator.__init__

    def tracked_init(orchestrator, *args, **kwargs):
        original_init(orchestrator, *args, **kwargs)
        orchestrators.append(orchestrator)

    monkeypatch.setattr(Orchestrator, "__init__", tracked_init)
    try:
        yield
    finally:
        for orchestrator in orchestrators:
            for job in orchestrator.jobs.values():
                job._cancel_event.set()
                job._pause_event.set()
            orchestrator.executor.shutdown(wait=True, cancel_futures=True)
        engine.dispose()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def sample_config_yaml():
    """Return a minimal sample configuration YAML."""
    return """
job_name: "test_job"
network:
  source: "build"
  build_config:
    dataset: "mnist"
    layers:
      - type: "dense"
        size: 10
        connectivity: 0.5
activity_recording:
  dataset: "mnist"
  ticks_per_image: 2
  images_per_label: 1
data_preparation:
  feature_types:
    - "avg_S"
  train_split: 0.8
training:
  epochs: 1
  batch_size: 8
evaluation:
  samples: 2
  window_size: 2
"""


@pytest.fixture
def sample_config_dict():
    """Return a minimal sample configuration dict."""
    return {
        "job_name": "test_job",
        "network": {
            "source": "build",
            "build_config": {
                "dataset": "mnist",
                "layers": [{"type": "dense", "size": 10, "connectivity": 0.5}],
            },
        },
        "activity_recording": {
            "dataset": "mnist",
            "ticks_per_image": 2,
            "images_per_label": 1,
        },
        "data_preparation": {
            "feature_types": ["avg_S"],
            "train_split": 0.8,
        },
        "training": {
            "epochs": 1,
            "batch_size": 8,
        },
        "evaluation": {
            "samples": 2,
            "window_size": 2,
        },
    }


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "integration: mark test as an integration test")


def pytest_collection_modifyitems(config, items):
    """Skip integration tests unless --integration flag is passed."""
    if config.getoption("--integration", default=False):
        return

    skip_integration = pytest.mark.skip(reason="need --integration option to run")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="run integration tests",
    )
