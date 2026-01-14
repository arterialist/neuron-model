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
