import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline.orchestrator import PipelineJob, Orchestrator, JobStatus
from pipeline.config import PipelineConfig


class TestZombieJobCancellation(unittest.TestCase):
    def test_cancel_zombie_job(self):
        """Test that cancelling a zombie job (paused with no active thread) immediately transitions to CANCELLED."""

        # 1. Setup Orchestrator
        with (
            patch("pipeline.database.init_db"),
            patch("pipeline.orchestrator.Orchestrator._restore_jobs"),
        ):
            orchestrator = Orchestrator(Path("."))

        # 2. Configure a job
        config = PipelineConfig(
            job_name="test_zombie_cancel",
            network={"source": "build", "build_config": {}},
        )

        # 3. Create a "zombie" job - paused but with no active execution thread
        with patch.object(orchestrator, "create_job"):
            job = PipelineJob("job_zombie", config, Path("."))
            job.status = JobStatus.PAUSED  # Simulating restored paused state
            orchestrator.jobs[job.job_id] = job
            # Note: We do NOT add it to _running_futures (simulating no active thread)

            # 4. Call cancel_job
            success = orchestrator.cancel_job(job.job_id)
            self.assertTrue(success, "Cancel should succeed")

            # 5. Assert final status is CANCELLED (not CANCELLING)
            print(f"Final Job Status: {job.status}")
            self.assertEqual(
                job.status,
                JobStatus.CANCELLED,
                f"Zombie job status should be CANCELLED but was {job.status}",
            )


if __name__ == "__main__":
    unittest.main()
