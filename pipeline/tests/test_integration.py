"""
Integration tests for the full pipeline.

These tests require the full dependencies and actually run the pipeline.
Mark with @pytest.mark.integration and run with: pytest --integration
"""

import pytest
from pathlib import Path

from pipeline.config import load_config_from_string
from pipeline.orchestrator import Orchestrator, JobStatus


# Minimal config for fast testing
MINIMAL_CONFIG = """
job_name: "integration_test"
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
  batch_size: 4
evaluation:
  samples: 2
  window_size: 2
"""


@pytest.mark.integration
class TestFullPipeline:
    """Integration tests for the full pipeline execution."""

    def test_full_pipeline_run(self, temp_dir):
        """Test a complete pipeline run with all steps."""
        config = load_config_from_string(MINIMAL_CONFIG)
        orchestrator = Orchestrator(temp_dir)

        job = orchestrator.create_job(config)
        result = orchestrator.run_job_sync(job.job_id)

        # Check job completed
        assert result.status == JobStatus.COMPLETED, (
            f"Job failed: {result.error_message}"
        )

        # Check all steps have results
        expected_steps = [
            "network",
            "activity_recording",
            "data_preparation",
            "training",
            "evaluation",
        ]
        for step in expected_steps:
            assert step in result.step_results, f"Missing step result: {step}"

        # Check artifacts exist
        assert (job.output_dir / "network" / "network.json").exists()
        training_dir = job.output_dir / "training"
        model_files = list(training_dir.rglob("*.pth"))
        assert len(model_files) > 0, "No model file (model.pth) found in training"
        assert (job.output_dir / "evaluation" / "evaluation_results.json").exists()

    def test_skip_steps(self, temp_dir):
        """Test skipping specific steps."""
        config_yaml = MINIMAL_CONFIG + "\nskip_steps:\n  - evaluation"
        config = load_config_from_string(config_yaml)
        orchestrator = Orchestrator(temp_dir)

        job = orchestrator.create_job(config)
        result = orchestrator.run_job_sync(job.job_id)

        assert result.status == JobStatus.COMPLETED
        assert result.step_results["evaluation"].status.value == "skipped"

    def test_network_file_source(self, temp_dir):
        """Test loading network from file instead of building."""
        # First create a network
        build_config = load_config_from_string(MINIMAL_CONFIG)
        orchestrator = Orchestrator(temp_dir)

        build_job = orchestrator.create_job(build_config)
        orchestrator.run_job_sync(build_job.job_id)

        # Now test loading it
        network_path = build_job.output_dir / "network" / "network.json"

        file_config_yaml = f"""
job_name: "file_source_test"
network:
  source: "file"
  path: "{network_path}"
skip_steps:
  - activity_recording
  - data_preparation
  - training
  - evaluation
"""
        file_config = load_config_from_string(file_config_yaml)

        file_job = orchestrator.create_job(file_config)
        result = orchestrator.run_job_sync(file_job.job_id)

        assert result.status == JobStatus.COMPLETED
        assert "network" in result.step_results
        assert result.step_results["network"].status.value == "completed"


@pytest.mark.integration
class TestArtifacts:
    """Integration tests for artifact handling."""

    def test_artifacts_created(self, temp_dir):
        """Test that expected artifacts are created."""
        config = load_config_from_string(MINIMAL_CONFIG)
        orchestrator = Orchestrator(temp_dir)

        job = orchestrator.create_job(config)
        result = orchestrator.run_job_sync(job.job_id)

        # Check network artifacts
        net_result = result.step_results.get("network")
        assert net_result is not None
        assert len(net_result.artifacts) > 0
        assert all(a.exists() for a in net_result.artifacts)

        # Check training artifacts (snn_trainer saves model.pth)
        train_result = result.step_results.get("training")
        assert train_result is not None
        artifact_names = [a.name for a in train_result.artifacts]
        assert any(n.endswith(".pth") for n in artifact_names)
        assert "training_history.json" in artifact_names or "model_config.json" in artifact_names
