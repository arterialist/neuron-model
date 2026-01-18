"""
Pipeline orchestrator for executing experiment pipelines.

Manages step execution, artifact tracking, and webhook notifications.
"""

import json
import logging
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, Future
from datetime import datetime
from pathlib import Path
from threading import Event
from typing import Any, Callable, Dict, List, Optional

import requests

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import PipelineConfig, WebhookConfig, WebhookEventType
from pipeline.steps.base import (
    StepContext,
    StepResult,
    StepStatus,
    Artifact,
    StepRegistry,
    StepCancelledException,
)

# Steps are registered via decorators in their respective modules
# We just need to ensure they are imported at runtime if not already
from pipeline.steps import (
    network_builder,
    activity_recorder,
    data_preparer,
    classifier_trainer,
    evaluator,
)
from pipeline import visualizations


class JobStatus:
    """Status of a pipeline job."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    CANCELLING = "cancelling"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PipelineJob:
    """Represents a pipeline job execution."""

    def __init__(
        self,
        job_id: str,
        config: PipelineConfig,
        output_dir: Path,
    ):
        self.job_id = job_id
        self.config = config
        self.output_dir = output_dir
        self.status = JobStatus.PENDING
        self.created_at = datetime.now()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.error_message: Optional[str] = None
        self.step_results: Dict[str, StepResult] = {}
        self.current_step: Optional[str] = None
        self.logs: List[str] = []
        # Threading events for pause/cancel control
        self._pause_event = Event()
        self._cancel_event = Event()
        self._pause_event.set()  # Not paused by default

        # Real-time logs buffer per step
        self.live_logs: Dict[str, List[str]] = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary for serialization."""
        return {
            "job_id": self.job_id,
            "job_name": self.config.job_name,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
            "error_message": self.error_message,
            "current_step": self.current_step,
            "steps": self._get_steps_dict(),
            "output_dir": str(self.output_dir),
        }

    def _get_steps_dict(self) -> Dict[str, Any]:
        """Combine completed results with live running logs."""
        steps_dict = {}
        for name, result in self.step_results.items():
            steps_dict[name] = result.to_dict()

        # If current step is running, add its live logs
        if self.current_step and self.current_step not in steps_dict:
            # Report actual job status if it's relevant to the step
            status = JobStatus.RUNNING
            if self.status in [JobStatus.PAUSED, JobStatus.CANCELLING]:
                status = self.status

            steps_dict[self.current_step] = {
                "status": status,
                "start_time": self.started_at.isoformat() if self.started_at else None,
                "logs": self.live_logs.get(self.current_step, []),
                "artifacts": [],
                "metrics": {},
                "duration_seconds": (
                    datetime.now() - (self.started_at or datetime.now())
                ).total_seconds(),
            }

        return steps_dict


class Orchestrator:
    """Pipeline orchestrator for executing experiments."""

    # Define the standard pipeline step order
    PIPELINE_STEPS = [
        "network",
        "activity_recording",
        "data_preparation",
        "training",
        "evaluation",
        "visualizations",
        "visualizations",
    ]

    class MemoryLogHandler(logging.Handler):
        """Log handler that buffers logs to a list."""

        def __init__(self, buffer: List[str]):
            super().__init__()
            self.buffer = buffer
            self.formatter = logging.Formatter("%(message)s")

        def emit(self, record):
            try:
                msg = self.format(record)
                self.buffer.append(msg)
            except Exception:
                self.handleError(record)

    def __init__(
        self,
        base_output_dir: Path,
        logger: Optional[logging.Logger] = None,
        on_job_update: Optional[Callable[[PipelineJob], None]] = None,
    ):
        """Initialize the orchestrator.

        Args:
            base_output_dir: Base directory for job outputs
            logger: Optional logger instance
            on_job_update: Callback for job status updates
        """
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logger or logging.getLogger(__name__)
        self.on_job_update = on_job_update

        self.jobs: Dict[str, PipelineJob] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._running_futures: Dict[str, Future] = {}

        # Initialize persistence
        from pipeline.database import init_db

        init_db()
        self._restore_jobs()

    def _restore_jobs(self) -> None:
        """Restore jobs from the database."""
        from pipeline.database import SessionLocal, JobModel
        from pipeline.config import PipelineConfig

        db = SessionLocal()
        try:
            stored_jobs = db.query(JobModel).all()
            for job_model in stored_jobs:
                try:
                    # Restore config
                    config = PipelineConfig(**json.loads(job_model.config_json))
                    output_dir = Path(job_model.output_dir)

                    # Reconstruct job
                    job = PipelineJob(job_model.job_id, config, output_dir)
                    job.status = job_model.status
                    job.created_at = job_model.created_at
                    job.started_at = job_model.started_at
                    job.completed_at = job_model.completed_at

                    # Restore full state if available
                    if job_model.state_json:
                        state = json.loads(job_model.state_json)
                        job.current_step = state.get("current_step")
                        job.error_message = state.get("error_message")
                        job.logs = state.get("logs", [])
                        job.live_logs = state.get(
                            "live_logs", {}
                        )  # Though usually empty for restored jobs

                        # Restore step results
                        for step_name, step_data in state.get("steps", {}).items():
                            # Skip if it's the current running step placeholder
                            if (
                                step_name == job.current_step
                                and job.status == JobStatus.RUNNING
                            ):
                                continue

                            # Restore StepResult
                            from pipeline.steps.base import (
                                StepResult,
                                StepStatus,
                                Artifact,
                            )

                            artifacts = []
                            for art_data in step_data.get("artifacts", []):
                                artifacts.append(
                                    Artifact(
                                        name=art_data["name"],
                                        path=Path(art_data["path"]),
                                        artifact_type=art_data["artifact_type"],
                                        size_bytes=art_data["size_bytes"],
                                        metadata=art_data.get("metadata", {}),
                                    )
                                )

                            # Handle potential legacy data where logs might be a string due to a bug
                            step_logs = step_data.get("logs", [])
                            if isinstance(step_logs, str):
                                step_logs = [step_logs]

                            result = StepResult(
                                status=StepStatus(step_data["status"]),
                                start_time=datetime.fromisoformat(
                                    step_data["start_time"]
                                )
                                if step_data.get("start_time")
                                else None,
                                end_time=datetime.fromisoformat(step_data["end_time"])
                                if step_data.get("end_time")
                                else None,
                                metrics=step_data.get("metrics", {}),
                                logs=step_logs,
                                error_message=step_data.get("error_message"),
                                artifacts=artifacts,
                            )
                            job.step_results[step_name] = result

                    job.step_results[step_name] = result

                    # Handle zombie jobs (interrupted by restart)
                    # Handle zombie jobs (interrupted by restart)
                    if job.status in [
                        JobStatus.RUNNING,
                        JobStatus.PAUSED,
                        JobStatus.CANCELLING,
                    ]:
                        self.logger.info(
                            f"Job {job.job_id} was in {job.status} state but process restarted. Setting to PAUSED to allow resumption."
                        )
                        job.status = JobStatus.PAUSED
                        job.logs.append(
                            f"Job execution interrupted by system restart. Paused at {datetime.now().isoformat()}."
                        )

                        # Update DB immediately
                        job_model.status = job.status
                        job_model.state_json = json.dumps(job.to_dict(), default=str)
                        db.add(job_model)

                    self.jobs[job.job_id] = job
                    self.logger.info(f"Restored job {job.job_id} from database")

                except Exception as e:
                    self.logger.error(f"Failed to restore job {job_model.job_id}: {e}")

            db.commit()  # Commit any status updates

        except Exception as e:
            self.logger.error(f"Failed to query jobs from database: {e}")
            db.rollback()
        finally:
            db.close()

    def _save_job_state(self, job: PipelineJob) -> None:
        """Save job state to database."""
        # Also save to disk for backup/consistency with original design
        try:
            state_path = job.output_dir / "job_state.json"
            job_dict = job.to_dict()
            with open(state_path, "w") as f:
                json.dump(job_dict, f, indent=2, default=str)
        except Exception as e:
            self.logger.warning(f"Failed to save job state to disk: {e}")

        # Save to DB
        from pipeline.database import SessionLocal, JobModel

        db = SessionLocal()
        try:
            job_model = db.query(JobModel).filter(JobModel.job_id == job.job_id).first()

            if not job_model:
                job_model = JobModel(
                    job_id=job.job_id,
                    job_name=job.config.job_name,
                    config_json=json.dumps(job.config.model_dump(), default=str),
                    output_dir=str(job.output_dir),
                )
                db.add(job_model)

            job_model.status = job.status
            job_model.started_at = job.started_at
            job_model.completed_at = job.completed_at
            job_model.state_json = json.dumps(job.to_dict(), default=str)

            db.commit()
        except Exception as e:
            self.logger.error(f"Failed to save job to database: {e}")
            db.rollback()
        finally:
            db.close()

    def create_job(self, config: PipelineConfig) -> PipelineJob:
        """Create a new pipeline job.

        Args:
            config: Pipeline configuration

        Returns:
            Created PipelineJob instance
        """
        job_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        output_dir = self.base_output_dir / f"{config.job_name}_{timestamp}_{job_id}"
        output_dir.mkdir(parents=True, exist_ok=True)

        job = PipelineJob(job_id, config, output_dir)
        self.jobs[job_id] = job

        # Save config to output dir
        with open(output_dir / "config.json", "w") as f:
            json.dump(config.model_dump(), f, indent=2, default=str)

        self.logger.info(f"Created job {job_id}: {config.job_name}")

        # Persist initial state
        self._save_job_state(job)

        return job

    def run_job_async(self, job_id: str) -> Future:
        """Run a job asynchronously.

        Args:
            job_id: ID of the job to run

        Returns:
            Future for the job execution
        """
        if job_id not in self.jobs:
            raise ValueError(f"Job not found: {job_id}")

        future = self.executor.submit(self._run_job, job_id)
        self._running_futures[job_id] = future

        return future

    def run_job_sync(self, job_id: str) -> PipelineJob:
        """Run a job synchronously.

        Args:
            job_id: ID of the job to run

        Returns:
            Completed PipelineJob
        """
        return self._run_job(job_id)

    def _run_job(self, job_id: str) -> PipelineJob:
        """Internal job execution method."""
        job = self.jobs[job_id]

        try:
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now()
            self._notify_job_update(job)
            self._save_job_state(job)
            self._send_webhook(job, WebhookEventType.JOB_STARTED)

            self.logger.info(f"Starting job {job_id}: {job.config.job_name}")
            job.logs.append(f"Job started at {job.started_at.isoformat()}")

            # Track artifacts from each step
            all_artifacts: Dict[str, List[Artifact]] = {}

            # Populate previous artifacts if this is a resumed job (omitted for brevity in this patch but important concept)
            # For now assuming clean run or simplistic resume where partial results might be loaded (TODO)

            # Execute each step in order
            for step_name in self.PIPELINE_STEPS:
                existing_logs = []

                # Check for existing results first (skip if already done in a restored job)
                if step_name in job.step_results:
                    result = job.step_results[step_name]
                    if (
                        result.status == StepStatus.COMPLETED
                        or result.status == StepStatus.SKIPPED
                    ):
                        all_artifacts[step_name] = result.artifacts
                        self.logger.info(
                            f"Skipping already completed step: {step_name}"
                        )
                        continue
                    else:
                        # Incomplete step found in results (resuming).
                        # We must remove it from step_results so the UI switches to monitoring live_logs.
                        # We also preserve the logs.
                        self.logger.info(f"Resuming incomplete step: {step_name}")
                        existing_logs = result.logs
                        del job.step_results[step_name]

                # --- Check for cancellation ---
                if job._cancel_event.is_set():
                    job.status = JobStatus.CANCELLED
                    job.completed_at = datetime.now()
                    job.logs.append(f"Job cancelled before step: {step_name}")
                    self._notify_job_update(job)
                    self._save_job_state(job)
                    self.logger.info(f"Job {job_id} cancelled.")
                    return job

                # --- Check for pause (wait if paused) ---
                if not job._pause_event.is_set():
                    job.status = JobStatus.PAUSED
                    job.logs.append(f"Job paused before step: {step_name}")
                    self._notify_job_update(job)
                    self._save_job_state(job)
                    self.logger.info(f"Job {job_id} paused. Waiting for resume...")
                    job._pause_event.wait()  # Block until resumed
                    job.status = JobStatus.RUNNING
                    job.logs.append(f"Job resumed, continuing to step: {step_name}")
                    self._notify_job_update(job)
                    self._save_job_state(job)
                    self.logger.info(f"Job {job_id} resumed.")

                # Check if step should be skipped
                if step_name in job.config.skip_steps:
                    self.logger.info(f"Skipping step: {step_name}")
                    job.logs.append(f"Skipped step: {step_name}")
                    job.step_results[step_name] = StepResult(
                        status=StepStatus.SKIPPED,
                        start_time=datetime.now(),
                        end_time=datetime.now(),
                    )
                    self._save_job_state(job)
                    continue

                # Get step class
                step_class = StepRegistry.get(step_name)
                if step_class is None:
                    self.logger.warning(f"Step not registered: {step_name}")
                    continue

                # Execute step
                job.current_step = step_name
                self._notify_job_update(job)
                self._save_job_state(job)
                self._send_webhook(
                    job, WebhookEventType.STEP_STARTED, {"step": step_name}
                )

                self.logger.info(f"Running step: {step_name}")
                job.logs.append(f"Running step: {step_name}")

                step = step_class()
                step_config = self._get_step_config(job.config, step_name)

                # Initialize live logs for this step
                job.live_logs[step_name] = existing_logs

                # Setup capturing logger
                step_logger = logging.getLogger(f"step.{step_name}")
                step_logger.setLevel(logging.INFO)
                handler = self.MemoryLogHandler(job.live_logs[step_name])
                step_logger.addHandler(handler)

                # Define control check callback
                def check_control_signals():
                    # Check cancellation
                    if job._cancel_event.is_set():
                        raise StepCancelledException("Step cancelled by user")

                    # Check pause
                    if not job._pause_event.is_set():
                        self.logger.info(
                            f"Step {step_name} paused mid-execution. Waiting for resume..."
                        )
                        job.status = JobStatus.PAUSED
                        self._notify_job_update(job)
                        self._save_job_state(job)

                        job._pause_event.wait()

                        # Re-check cancellation after waking from pause
                        if job._cancel_event.is_set():
                            raise StepCancelledException("Step cancelled by user")

                        self.logger.info(f"Step {step_name} resumed.")
                        job.status = JobStatus.RUNNING
                        self._notify_job_update(job)
                        self._save_job_state(job)

                context = StepContext(
                    job_id=job_id,
                    job_name=job.config.job_name,
                    output_dir=job.output_dir,
                    config=step_config,
                    previous_artifacts=all_artifacts.copy(),
                    logger=step_logger,
                    check_control_signals=check_control_signals,
                )

                try:
                    # Update DB occasionally with logs?
                    # For now, we only save state after step completes to avoid perf hit.
                    result = step.run(context)
                except StepCancelledException as e:
                    self.logger.info(f"Step {step_name} cancelled: {e}")
                    job.logs.append(f"Step {step_name} cancelled.")

                    result = StepResult(
                        status=StepStatus.CANCELLED,
                        error_message=str(e),
                        start_time=datetime.now(),  # Approximate
                        end_time=datetime.now(),
                    )
                finally:
                    # Clean up handler
                    step_logger.removeHandler(handler)

                job.step_results[step_name] = result

                if result.status == StepStatus.CANCELLED:
                    job.status = JobStatus.CANCELLED
                    job.completed_at = datetime.now()
                    job.logs.append(f"Job cancelled during step: {step_name}")
                    self._notify_job_update(job)
                    self._save_job_state(job)
                    self.logger.info(f"Job {job_id} cancelled during step {step_name}.")
                    return job
                elif (
                    result.status == StepStatus.COMPLETED
                    or result.status == StepStatus.SKIPPED
                ):
                    all_artifacts[step_name] = result.artifacts
                    self._send_webhook(
                        job,
                        WebhookEventType.STEP_COMPLETED,
                        {
                            "step": step_name,
                            "metrics": result.metrics,
                        },
                    )
                    self.logger.info(f"Step {step_name} completed successfully")
                    job.logs.append(f"Step {step_name} completed")
                else:
                    self._send_webhook(
                        job,
                        WebhookEventType.STEP_FAILED,
                        {
                            "step": step_name,
                            "error": result.error_message,
                        },
                    )
                    raise RuntimeError(
                        f"Step {step_name} failed: {result.error_message}"
                    )

                self._notify_job_update(job)
                self._save_job_state(job)

            # Job completed successfully
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.now()
            job.current_step = None
            job.logs.append(f"Job completed at {job.completed_at.isoformat()}")

            self._notify_job_update(job)
            self._send_webhook(job, WebhookEventType.JOB_COMPLETED)

            # Save job state
            self._save_job_state(job)

            self.logger.info(f"Job {job_id} completed successfully")

        except Exception as e:
            import traceback

            error_trace = traceback.format_exc()

            job.status = JobStatus.FAILED
            job.completed_at = datetime.now()
            job.error_message = str(e)
            job.logs.append(f"ERROR: {e}")
            job.logs.append(error_trace)

            self._notify_job_update(job)
            self._send_webhook(job, WebhookEventType.JOB_FAILED, {"error": str(e)})

            self._save_job_state(job)

            self.logger.error(f"Job {job_id} failed: {e}")

        finally:
            if job_id in self._running_futures:
                del self._running_futures[job_id]

        return job

    def _get_step_config(self, config: PipelineConfig, step_name: str) -> Any:
        """Get the configuration for a specific step."""
        config_map = {
            "network": config.network,
            "activity_recording": config.activity_recording,
            "data_preparation": config.data_preparation,
            "training": config.training,
            "evaluation": config.evaluation,
            "visualizations": config.visualizations,
        }
        return config_map.get(step_name)

    def _notify_job_update(self, job: PipelineJob) -> None:
        """Notify callback of job update."""
        if self.on_job_update:
            try:
                self.on_job_update(job)
            except Exception as e:
                self.logger.warning(f"Job update callback failed: {e}")

    def _send_webhook(
        self,
        job: PipelineJob,
        event: WebhookEventType,
        extra_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Send webhook notification."""
        for webhook_config in job.config.webhooks:
            if event not in webhook_config.events:
                continue

            payload = {
                "event": event.value,
                "job_id": job.job_id,
                "job_name": job.config.job_name,
                "status": job.status,
                "timestamp": datetime.now().isoformat(),
            }

            if extra_data:
                payload.update(extra_data)

            try:
                response = requests.post(
                    webhook_config.url,
                    json=payload,
                    headers=webhook_config.headers,
                    timeout=webhook_config.timeout_seconds,
                )
                self.logger.debug(
                    f"Webhook sent to {webhook_config.url}: {response.status_code}"
                )
            except Exception as e:
                self.logger.warning(f"Webhook failed for {webhook_config.url}: {e}")

    # _save_job_state merged into the main flow above as it's now dual-write (DB + File)
    # Keeping the file method signature compatible if needed, but the main logic is updated.

    def get_job(self, job_id: str) -> Optional[PipelineJob]:
        """Get a job by ID."""
        # If in memory, return. If not (unlikely with _restore_jobs), try fetching?
        # For now, rely on _restore_jobs loading everything into memory.
        return self.jobs.get(job_id)

    def list_jobs(self) -> List[PipelineJob]:
        """List all jobs."""
        return list(self.jobs.values())

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running or paused job.

        Sets the cancel event to signal job termination.
        Returns:
            True if cancellation was signaled successfully.
        """
        if job_id not in self.jobs:
            return False

        job = self.jobs[job_id]

        if job.status in [JobStatus.RUNNING, JobStatus.PAUSED]:
            job._cancel_event.set()
            # If paused, also resume so the loop can exit
            job._pause_event.set()

            # Check if job has an active execution thread
            has_active_thread = (
                job_id in self._running_futures
                and not self._running_futures[job_id].done()
            )

            if has_active_thread:
                # Execution thread will handle the cancellation
                job.status = JobStatus.CANCELLING
                job.logs.append(
                    f"Job cancellation requested at {datetime.now().isoformat()}"
                )
            else:
                # No active thread (zombie job) - immediately cancel
                job.status = JobStatus.CANCELLED
                job.completed_at = datetime.now()
                job.logs.append(
                    f"Job cancelled (no active execution) at {datetime.now().isoformat()}"
                )
                self.logger.info(f"Zombie job {job_id} cancelled immediately.")

            self._notify_job_update(job)
            self._save_job_state(job)
            return True

        return False

    def pause_job(self, job_id: str) -> bool:
        """Pause a running job after the current step.

        Returns:
            True if pause was signaled successfully.
        """
        if job_id not in self.jobs:
            return False

        job = self.jobs[job_id]

        if job.status == JobStatus.RUNNING:
            job._pause_event.clear()  # Signal pause
            job.logs.append(f"Pause requested at {datetime.now().isoformat()}")
            self._notify_job_update(job)
            self._save_job_state(job)
            return True

        return False

    def resume_job(self, job_id: str) -> bool:
        """Resume a paused job.

        Returns:
            True if resume was signaled successfully.
        """
        if job_id not in self.jobs:
            return False

        job = self.jobs[job_id]

        if job.status == JobStatus.PAUSED:
            job._pause_event.set()  # Signal resume
            job.logs.append(f"Resume requested at {datetime.now().isoformat()}")

            # Check if there is an active future running this job
            if (
                job_id not in self._running_futures
                or self._running_futures[job_id].done()
            ):
                self.logger.info(f"Restarting execution loop for resumed job {job_id}")
                self.run_job_async(job_id)

            # Status will be set to RUNNING in the job loop
            return True

        return False

    def delete_job(self, job_id: str) -> bool:
        """Delete a job and its output directory.

        Only works for completed, failed, or cancelled jobs.
        Returns:
            True if deletion was successful.
        """
        import shutil
        from pipeline.database import SessionLocal, JobModel

        if job_id not in self.jobs:
            return False

        job = self.jobs[job_id]

        if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            # Remove output directory
            if job.output_dir.exists():
                shutil.rmtree(job.output_dir, ignore_errors=True)
                self.logger.info(f"Deleted job output directory: {job.output_dir}")

            # Remove from DB
            db = SessionLocal()
            try:
                db.query(JobModel).filter(JobModel.job_id == job_id).delete()
                db.commit()
            except Exception as e:
                self.logger.error(f"Failed to delete job {job_id} from database: {e}")
            finally:
                db.close()

            # Remove from jobs dict
            del self.jobs[job_id]
            self.logger.info(f"Job {job_id} deleted.")
            return True

        return False

    def shutdown(self) -> None:
        """Shutdown the orchestrator."""
        self.executor.shutdown(wait=False)


def run_pipeline(
    config: PipelineConfig,
    output_dir: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
) -> PipelineJob:
    """Convenience function to run a pipeline synchronously.

    Args:
        config: Pipeline configuration
        output_dir: Output directory (defaults to ./experiments)
        logger: Optional logger

    Returns:
        Completed PipelineJob
    """
    if output_dir is None:
        output_dir = Path("./experiments")

    orchestrator = Orchestrator(output_dir, logger)
    job = orchestrator.create_job(config)
    return orchestrator.run_job_sync(job.job_id)
