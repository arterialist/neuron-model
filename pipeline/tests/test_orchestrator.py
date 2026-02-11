"""
Tests for the orchestrator with mocked steps.
"""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from pipeline.config import PipelineConfig, load_config_from_string
from pipeline.orchestrator import Orchestrator, PipelineJob, JobStatus
from pipeline.steps.base import StepResult, StepStatus, Artifact


class MockStep:
    """Mock step for testing."""

    def __init__(self, name: str, should_fail: bool = False):
        self._name = name
        self.should_fail = should_fail

    @property
    def name(self) -> str:
        return self._name

    def run(self, context):
        if self.should_fail:
            return StepResult(
                status=StepStatus.FAILED,
                error_message="Mock failure",
                start_time=datetime.now(),
                end_time=datetime.now(),
            )

        # Create a mock artifact
        artifact_path = context.output_dir / self.name / "output.json"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text('{"mock": true}')

        return StepResult(
            status=StepStatus.COMPLETED,
            artifacts=[
                Artifact(
                    name="output.json",
                    path=artifact_path,
                    artifact_type="data",
                    size_bytes=len('{"mock": true}'),
                )
            ],
            metrics={"mock_metric": 1.0},
            start_time=datetime.now(),
            end_time=datetime.now(),
        )


class TestOrchestrator:
    """Tests for the Orchestrator class."""

    def test_create_job(self, temp_dir, sample_config_yaml):
        """Test job creation."""
        config = load_config_from_string(sample_config_yaml)
        orchestrator = Orchestrator(temp_dir)

        job = orchestrator.create_job(config)

        assert job.job_id is not None
        assert job.config.job_name == "test_job"
        assert job.status == JobStatus.PENDING
        assert job.output_dir.exists()

    def test_list_jobs(self, temp_dir, sample_config_yaml):
        """Test job listing."""
        config = load_config_from_string(sample_config_yaml)
        orchestrator = Orchestrator(temp_dir)

        # Create multiple jobs
        job1 = orchestrator.create_job(config)
        job2 = orchestrator.create_job(config)

        jobs = orchestrator.list_jobs()

        assert len(jobs) == 2
        assert job1 in jobs
        assert job2 in jobs

    def test_get_job(self, temp_dir, sample_config_yaml):
        """Test getting a job by ID."""
        config = load_config_from_string(sample_config_yaml)
        orchestrator = Orchestrator(temp_dir)

        created_job = orchestrator.create_job(config)
        retrieved_job = orchestrator.get_job(created_job.job_id)

        assert retrieved_job is created_job

    def test_get_nonexistent_job(self, temp_dir):
        """Test getting a nonexistent job returns None."""
        orchestrator = Orchestrator(temp_dir)

        job = orchestrator.get_job("nonexistent")

        assert job is None

    @patch("pipeline.orchestrator.StepRegistry.get")
    def test_run_job_with_mocked_steps(self, mock_get, temp_dir, sample_config_yaml):
        """Test running a job with mocked steps."""

        # Mock the step registry to return our mock steps
        def get_mock_step(name):
            class MockStepClass:
                def __new__(cls):
                    return MockStep(name)

            return MockStepClass

        mock_get.side_effect = get_mock_step

        config = load_config_from_string(sample_config_yaml)
        orchestrator = Orchestrator(temp_dir)

        job = orchestrator.create_job(config)
        result = orchestrator.run_job_sync(job.job_id)

        assert result.status == JobStatus.COMPLETED
        assert result.completed_at is not None

    @patch("pipeline.orchestrator.StepRegistry.get")
    def test_run_job_step_failure(self, mock_get, temp_dir, sample_config_yaml):
        """Test that job fails when a step fails."""

        # Mock the step registry to return a failing step
        def get_mock_step(name):
            if name == "activity_recording":

                class FailingStepClass:
                    def __new__(cls):
                        return MockStep(name, should_fail=True)

                return FailingStepClass
            else:

                class MockStepClass:
                    def __new__(cls):
                        return MockStep(name)

                return MockStepClass

        mock_get.side_effect = get_mock_step

        config = load_config_from_string(sample_config_yaml)
        orchestrator = Orchestrator(temp_dir)

        job = orchestrator.create_job(config)
        result = orchestrator.run_job_sync(job.job_id)

        assert result.status == JobStatus.FAILED
        assert result.error_message is not None

    def test_cancel_pending_job(self, temp_dir, sample_config_yaml):
        """Test that pending jobs cannot be cancelled."""
        config = load_config_from_string(sample_config_yaml)
        orchestrator = Orchestrator(temp_dir)

        job = orchestrator.create_job(config)
        result = orchestrator.cancel_job(job.job_id)

        # Pending jobs cannot be cancelled (only running ones)
        assert result is False


class TestJobState:
    """Tests for job state management."""

    def test_job_to_dict(self, temp_dir, sample_config_yaml):
        """Test converting job to dictionary."""
        config = load_config_from_string(sample_config_yaml)
        job = PipelineJob("test123", config, temp_dir / "output")

        data = job.to_dict()

        assert data["job_id"] == "test123"
        assert data["job_name"] == "test_job"
        assert data["status"] == JobStatus.PENDING
        assert "created_at" in data

    def test_job_state_persistence(self, temp_dir, sample_config_yaml):
        """Test that job state is saved to output directory."""
        config = load_config_from_string(sample_config_yaml)
        orchestrator = Orchestrator(temp_dir)

        job = orchestrator.create_job(config)

        # Config should be saved
        config_file = job.output_dir / "config.json"
        assert config_file.exists()
