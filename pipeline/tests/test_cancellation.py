import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from threading import Event
from datetime import datetime

from pipeline.steps.base import StepContext, StepCancelledException, StepStatus
from pipeline.orchestrator import PipelineJob, Orchestrator, JobStatus
from pipeline.config import PipelineConfig


class TestCancellation(unittest.TestCase):
    def test_context_cancellation_check(self):
        """Test that context.check_control_signals raises exception when cancelled."""
        job = MagicMock(spec=PipelineJob)
        job._cancel_event = MagicMock()
        job._cancel_event.is_set.return_value = True
        job._pause_event = MagicMock()
        job._pause_event.is_set.return_value = True  # Not paused

        # Define the check callback as in orchestrator
        def check_control_signals():
            if job._cancel_event.is_set():
                raise StepCancelledException("Step cancelled by user")

        context = StepContext(
            job_id="test",
            job_name="test",
            output_dir=Path("."),
            config={},
            check_control_signals=check_control_signals,
        )

        with self.assertRaises(StepCancelledException):
            context.check_control_signals()

    def test_loop_integration_mock(self):
        """Simulate a step loop integrated with cancellation."""
        job = MagicMock(spec=PipelineJob)
        job._cancel_event = MagicMock()
        # Cancel after 2 iterations
        side_effects = [False, False, True]
        job._cancel_event.is_set.side_effect = side_effects
        job._pause_event = MagicMock()
        job._pause_event.is_set.return_value = True

        def check_control_signals():
            if job._cancel_event.is_set():
                raise StepCancelledException("Cancelled")

        context = StepContext(
            job_id="test",
            job_name="test",
            output_dir=Path("."),
            config={},
            check_control_signals=check_control_signals,
        )

        # simulate loop
        processed = 0
        try:
            for i in range(5):
                context.check_control_signals()
                processed += 1
        except StepCancelledException:
            pass

        # Should have processed 2 items (0, 1) then cancelled on 3rd check (index 2)
        # Actually side_effect is called:
        # i=0 check -> False (processed=1)
        # i=1 check -> False (processed=2)
        # i=2 check -> True -> Raise (processed=2)
        self.assertEqual(processed, 2)

    def test_orchestrator_job_cancellation(self):
        """Test orchestrator cancellation logic updates job status."""
        # Mock init_db to avoid filesystem access
        with (
            patch("pipeline.database.init_db"),
            patch("pipeline.orchestrator.Orchestrator._restore_jobs"),
        ):
            orchestrator = Orchestrator(Path("."))

        config = PipelineConfig(
            job_name="test_job",
            network={
                "source": "build",
                "build_config": {"layers": [{"type": "dense", "size": 100}]},
            },
            activity_recording={"dataset": "mnist"},
        )

        # Mock create_job to avoid FS ops
        with patch.object(orchestrator, "create_job") as mock_create:
            job = PipelineJob("job_123", config, Path("."))
            orchestrator.jobs[job.job_id] = job

            # Start normally
            job.status = StepStatus.RUNNING

            # Cancel
            success = orchestrator.cancel_job("job_123")

            self.assertTrue(success)
            self.assertTrue(job._cancel_event.is_set())
            self.assertEqual(job.status, "cancelling")  # JobStatus.CANCELLING

    def test_paused_job_status_reporting(self):
        """Test that _get_steps_dict reports PAUSED status for current step."""
        # Mock init_db to avoid filesystem access
        with (
            patch("pipeline.database.init_db"),
            patch("pipeline.orchestrator.Orchestrator._restore_jobs"),
        ):
            orchestrator = Orchestrator(Path("."))

        config = PipelineConfig(
            job_name="test_job",
            network={
                "source": "build",
                "build_config": {"layers": [{"type": "dense", "size": 100}]},
            },
            activity_recording={"dataset": "mnist"},
        )

        with patch.object(orchestrator, "create_job") as mock_create:
            job = PipelineJob("job_paused", config, Path("."))
            orchestrator.jobs[job.job_id] = job

            # Simulate paused state
            job.status = StepStatus.PAUSED
            job.current_step = "training"
            job.started_at = datetime.now()

            # Orchestrator uses job.to_dict which uses _get_steps_dict
            steps_dict = job._get_steps_dict()

            self.assertIn("training", steps_dict)
            self.assertEqual(steps_dict["training"]["status"], "paused")
            self.assertIsNotNone(steps_dict["training"].get("start_time"))

    def test_zombie_job_resumption(self):
        """Test that resuming a zombie job (restored as PAUSED) restarts execution."""
        # Mock init_db
        with (
            patch("pipeline.database.init_db"),
            patch("pipeline.orchestrator.Orchestrator._restore_jobs"),
        ):
            orchestrator = Orchestrator(Path("."))

        config = PipelineConfig(
            job_name="test_job",
            network={
                "source": "build",
                "build_config": {"layers": [{"type": "dense", "size": 100}]},
            },
            activity_recording={"dataset": "mnist"},
        )

        # Manually create a "restored" zombie job
        job = PipelineJob("job_zombie", config, Path("."))
        job.status = StepStatus.PAUSED  # Restored as PAUSED
        orchestrator.jobs[job.job_id] = job

        # Mock run_job_async
        with patch.object(orchestrator, "run_job_async") as mock_run:
            orchestrator.resume_job("job_zombie")

            # Should have called run_job_async because no active future exists
            mock_run.assert_called_once_with("job_zombie")

            # Verify event is set
            self.assertTrue(job._pause_event.is_set())

            # Verify event is set
            self.assertTrue(job._pause_event.is_set())

    def test_resumed_job_logs(self):
        """Test that logs from a resumed step are preserved and updated."""
        # Mock init_db
        with (
            patch("pipeline.database.init_db"),
            patch("pipeline.orchestrator.Orchestrator._restore_jobs"),
        ):
            orchestrator = Orchestrator(Path("."))

        config = PipelineConfig(
            job_name="test_job",
            network={
                "source": "build",
                "build_config": {"layers": [{"type": "dense", "size": 100}]},
            },
            activity_recording={"dataset": "mnist"},
        )

        job = PipelineJob("job_log_resume", config, Path("."))
        orchestrator.jobs[job.job_id] = job

        # Simulate a partially completed step in step_results
        from pipeline.steps.base import StepResult, StepStatus

        # Create a partial result with some logs
        partial_result = StepResult(
            status=StepStatus.PAUSED,
            start_time=datetime.now(),
            end_time=None,
            logs=["Log line 1", "Log line 2"],
        )
        job.step_results["activity_recording"] = partial_result
        job.status = JobStatus.PAUSED

        # We need to mock StepRegistry.get to return a mock step that appends a log
        mock_step_class = MagicMock()
        mock_step_instance = MagicMock()

        def mock_step_run(context):
            # This should run during _run_job
            # Verify logs are passed to handler
            context.logger.info("Log line 3 (new)")
            return StepResult(
                StepStatus.COMPLETED, datetime.now(), datetime.now(), logs=[]
            )

        mock_step_instance.run.side_effect = mock_step_run
        mock_step_class.return_value = mock_step_instance

        with patch(
            "pipeline.orchestrator.StepRegistry.get", return_value=mock_step_class
        ):
            # Run job sync to avoid thread issues in test
            job_out = orchestrator.run_job_sync("job_log_resume")

            # Verify that activity_recording logs contain ALL logs
            # The final result logs in step_results are separate from live_logs during run,
            # but let's check what happened to live_logs during the run.
            # In our mock run, we logged "New".
            # The orchestrator should have initialized live_logs with ["Log line 1", "Log line 2"]
            # And the MemoryHandler should have appended "Log line 3 (new)"

            self.assertIn("Log line 1", job_out.live_logs["activity_recording"])
            self.assertIn("Log line 3 (new)", job_out.live_logs["activity_recording"])


if __name__ == "__main__":
    unittest.main()
