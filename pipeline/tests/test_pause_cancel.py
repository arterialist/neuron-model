import unittest
import time
import threading
from pathlib import Path
from unittest.mock import patch

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


# Mock step that pauses for a while to allow testing pause/cancel
@StepRegistry.register
class MockPausableStep(PipelineStep):
    @property
    def name(self) -> str:
        return "mock_pausable_step"

    def run(self, context: StepContext) -> StepResult:
        # Simulate iterations that check control signals
        for i in range(10):
            time.sleep(0.1)
            if context.check_control_signals:
                context.check_control_signals()

        from datetime import datetime

        return StepResult(
            status=StepStatus.COMPLETED,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )


class TestPausedJobCancellation(unittest.TestCase):
    def test_cancel_paused_job(self):
        """Test that cancelling a paused job correctly transitions to CANCELLED status."""

        # 1. Setup Orchestrator
        with (
            patch("pipeline.database.init_db"),
            patch("pipeline.orchestrator.Orchestrator._restore_jobs"),
        ):
            orchestrator = Orchestrator(Path("."))

        # 2. Configure a job
        config = PipelineConfig(
            job_name="test_pause_cancel",
            network={"source": "build", "build_config": {}},
        )

        # 3. Create the job
        with patch.object(orchestrator, "create_job"):
            job = PipelineJob("job_pause_cancel", config, Path("."))
            orchestrator.jobs[job.job_id] = job
            orchestrator.PIPELINE_STEPS = ["mock_pausable_step"]

            # 4. Run the job in a thread
            with patch(
                "pipeline.orchestrator.StepRegistry.get",
                return_value=MockPausableStep,
            ):
                job_thread = threading.Thread(
                    target=orchestrator._run_job, args=(job.job_id,)
                )
                job_thread.start()

                # Wait a bit for the job to start
                time.sleep(0.2)

                # 5. Pause the job
                orchestrator.pause_job(job.job_id)
                time.sleep(0.2)  # Allow pause to take effect

                # Verify job is paused
                self.assertEqual(
                    job.status,
                    JobStatus.PAUSED,
                    f"Job should be PAUSED but was {job.status}",
                )

                # 6. Cancel the paused job
                success = orchestrator.cancel_job(job.job_id)
                self.assertTrue(success, "Cancel should succeed")

                # Wait for the job thread to finish
                job_thread.join(timeout=2.0)

                # 7. Assert final status is CANCELLED
                print(f"Final Job Status: {job.status}")
                self.assertEqual(
                    job.status,
                    JobStatus.CANCELLED,
                    f"Job status should be CANCELLED but was {job.status}",
                )


if __name__ == "__main__":
    unittest.main()
