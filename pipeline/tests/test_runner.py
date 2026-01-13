import pytest
from unittest.mock import patch, MagicMock
from pipeline.runner import run_pipeline
from pipeline.config import JobConfig
import yaml
import os

@patch("pipeline.runner.build_network")
@patch("pipeline.runner.run_recording")
@patch("pipeline.runner.process_data")
@patch("pipeline.runner.train_model")
@patch("pipeline.runner.evaluate_model")
@patch("pipeline.runner.run_step_visualization")
def test_run_pipeline_orchestration(
    mock_viz, mock_eval, mock_train, mock_process, mock_record, mock_build, tmp_path
):
    # Create dummy config
    config_path = tmp_path / "config.yaml"
    config_data = {
        "name": "test_job",
        "network": {
            "layers": [{"layer_type": "conv", "filters": 1, "kernel_size": 3, "stride": 1}],
            "inhibitory_signals": False
        },
        "simulation": {},
        "preparation": {},
        "training": {},
        "evaluation": {},
        "visualizations": {"network_activity": {"enabled": True}}
    }
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)

    # Run pipeline
    run_pipeline(str(config_path))

    # Verify calls
    mock_build.assert_called_once()
    mock_record.assert_called_once()
    mock_process.assert_called_once()
    mock_train.assert_called_once()
    mock_eval.assert_called_once()

    # Check that visualization was called
    assert mock_viz.call_count >= 1
