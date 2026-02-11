import unittest
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
from threading import Event
from datetime import datetime

from pipeline.steps.base import (
    StepContext,
    StepCancelledException,
    StepStatus,
    PipelineStep,
    StepResult,
    StepRegistry,
)
from pipeline.orchestrator import PipelineJob, Orchestrator, JobStatus
from pipeline.config import PipelineConfig


# Mock step that raises StepCancelledException
@StepRegistry.register
class MockCancellableStep(PipelineStep):
    @property
    def name(self) -> str:
        return "mock_cancellable_step"

    def run(self, context: StepContext) -> StepResult:
        # Simulate some work
        time.sleep(0.1)
        # Check for cancellation which should raise StepCancelledException
        # We simulate the user cancelling right here
        if context.check_control_signals:
            # We assume the test sets the cancel event before this runs or we force it here
            pass

        # raise exception directly to simulate what happens when check_control_signals is called
        raise StepCancelledException("Step cancelled by user")


class TestCancellationRepro(unittest.TestCase):
    def test_job_cancellation_status(self):
        """Test that a job cancelled during execution ends up with CANCELLED status, not FAILED."""

        # 1. Setup Orchestrator
        with (
            patch("pipeline.database.init_db"),
            patch("pipeline.orchestrator.Orchestrator._restore_jobs"),
        ):
            orchestrator = Orchestrator(Path("."))

        # 2. Configure a job that uses our mock step (we'll inject it via mocking PIPELINE_STEPS)
        config = PipelineConfig(
            job_name="test_cancellation_repro",
            network={"source": "build", "build_config": {}},  # Minimally valid config
        )

        # 3. Create the job
        with patch.object(orchestrator, "create_job") as mock_create_job:
            job = PipelineJob("job_repro", config, Path("."))
            orchestrator.jobs[job.job_id] = job
            orchestrator.PIPELINE_STEPS = [
                "mock_cancellable_step"
            ]  # Override steps to use our mock

            # 4. Run the job synchronously (to keep test simple)
            # We expect the mock step to raise StepCancelledException.
            # CURRENT BEHAVIOR: Orchestrator catches it, logs generic error, sets status to FAILED.
            # DESIRED BEHAVIOR: Orchestrator catches it, sets status to CANCELLED.

            # Use patch for StepRegistry.get to return our mock class
            with patch(
                "pipeline.orchestrator.StepRegistry.get",
                return_value=MockCancellableStep,
            ):
                # Also we need to make sure job._cancel_event is set so check_control_signals works effectively
                # but our MockCancellableStep raises it unconditionally for repro purposes.
                # However, logic in orchestrator checks `if job._cancel_event.is_set():` inside the loop too.
                # Let's set the cancel event so orchestrator knows it was intentional.
                # job._cancel_event.set()

                orchestrator._run_job(job.job_id)

            # 5. Assertions
            print(f"Final Job Status: {job.status}")
            print(f"Final Job Error: {job.error_message}")
            if job.error_message:
                print(f"Logs: {job.logs}")

            # This assertion fails in the current buggy version (it is FAILED instead of CANCELLED)
            self.assertEqual(
                job.status,
                JobStatus.CANCELLED,
                f"Job status should be CANCELLED but was {job.status}",
            )


if __name__ == "__main__":
    unittest.main()
