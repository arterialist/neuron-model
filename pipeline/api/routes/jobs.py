"""
Job management API routes.
"""

import io
import tarfile
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse

from pipeline.api.models import (
    JobCreate,
    JobSummary,
    JobDetail,
    StepInfo,
    ArtifactInfo,
    ApiResponse,
)
from pipeline.config import load_config_from_string
from pipeline.orchestrator import Orchestrator, PipelineJob


router = APIRouter(prefix="/api/jobs", tags=["jobs"])

# Global orchestrator instance (will be set from main.py)
_orchestrator: Optional[Orchestrator] = None


def set_orchestrator(orchestrator: Orchestrator) -> None:
    """Set the global orchestrator instance."""
    global _orchestrator
    _orchestrator = orchestrator


def get_orchestrator() -> Orchestrator:
    """Get the global orchestrator instance."""
    if _orchestrator is None:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")
    return _orchestrator


def job_to_summary(job: PipelineJob) -> JobSummary:
    """Convert a PipelineJob to a JobSummary."""
    return JobSummary(
        job_id=job.job_id,
        job_name=job.config.job_name,
        status=job.status,
        created_at=job.created_at.isoformat(),
        started_at=job.started_at.isoformat() if job.started_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
        current_step=job.current_step,
        error_message=job.error_message,
    )


def job_to_detail(job: PipelineJob) -> JobDetail:
    """Convert a PipelineJob to a JobDetail."""
    steps = {}
    total_storage = 0
    total_duration = 0.0

    for step_name, result in job.step_results.items():
        step_storage = sum(a.size_bytes for a in result.artifacts)
        total_storage += step_storage

        artifacts = [
            ArtifactInfo(
                name=a.name,
                path=str(a.path),
                artifact_type=a.artifact_type,
                size_bytes=a.size_bytes,
                exists=a.exists(),
                metadata=a.metadata,
            )
            for a in result.artifacts
        ]

        if result.duration_seconds:
            total_duration += result.duration_seconds

        steps[step_name] = StepInfo(
            step_name=step_name,
            status=result.status.value,
            artifacts=artifacts,
            metrics=result.metrics,
            logs=result.logs[-100] if len(result.logs) > 100 else result.logs,
            error_message=result.error_message,
            start_time=result.start_time.isoformat() if result.start_time else None,
            end_time=result.end_time.isoformat() if result.end_time else None,
            duration_seconds=result.duration_seconds,
            storage_bytes=step_storage,
        )

    # Check for running step that isn't in results yet
    if job.current_step and job.current_step not in steps and job.status == "running":
        # Estimate duration as time since job start ( imperfect but useful)
        duration = 0.0
        if job.started_at:
            duration = (datetime.now() - job.started_at).total_seconds()
            # Note: We don't add ongoing step duration to total_duration to avoid double counting
            # effectively until it's done, or we could add it. Let's add it for "live" feel.
            total_duration += duration

        steps[job.current_step] = StepInfo(
            step_name=job.current_step,
            status="running",
            artifacts=[],
            metrics={},
            logs=job.live_logs.get(job.current_step, []),
            duration_seconds=duration,
            storage_bytes=0,  # Live steps usually haven't finalized artifacts yet
        )

    return JobDetail(
        job_id=job.job_id,
        job_name=job.config.job_name,
        status=job.status,
        output_dir=str(job.output_dir),
        created_at=job.created_at.isoformat(),
        started_at=job.started_at.isoformat() if job.started_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
        current_step=job.current_step,
        error_message=job.error_message,
        steps=steps,
        total_duration_seconds=total_duration,
        total_storage_bytes=total_storage,
        config=job.config.model_dump(),
    )


@router.get("", response_model=list[JobSummary])
async def list_jobs(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    status: Optional[str] = Query(default=None),
) -> list[JobSummary]:
    """List all jobs with optional filtering."""
    orchestrator = get_orchestrator()
    jobs = orchestrator.list_jobs()

    # Filter by status if provided
    if status:
        jobs = [j for j in jobs if j.status == status]

    # Sort by creation time (newest first)
    jobs.sort(key=lambda j: j.created_at, reverse=True)

    # Apply pagination
    jobs = jobs[offset : offset + limit]

    return [job_to_summary(j) for j in jobs]


@router.post("", response_model=JobSummary)
async def create_job(job_create: JobCreate) -> JobSummary:
    """Create and start a new pipeline job."""
    orchestrator = get_orchestrator()

    try:
        config = load_config_from_string(job_create.config_yaml)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid config: {e}")

    job = orchestrator.create_job(config)

    # Start job asynchronously
    orchestrator.run_job_async(job.job_id)

    return job_to_summary(job)


@router.get("/{job_id}", response_model=JobDetail)
async def get_job(job_id: str) -> JobDetail:
    """Get detailed information about a job."""
    orchestrator = get_orchestrator()
    job = orchestrator.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return job_to_detail(job)


@router.post("/{job_id}/pause", response_model=ApiResponse)
async def pause_job(job_id: str) -> ApiResponse:
    """Pause a running job after the current step completes."""
    orchestrator = get_orchestrator()

    if orchestrator.pause_job(job_id):
        return ApiResponse(success=True, message="Pause requested")
    else:
        raise HTTPException(status_code=400, detail="Cannot pause job")


@router.post("/{job_id}/resume", response_model=ApiResponse)
async def resume_job(job_id: str) -> ApiResponse:
    """Resume a paused job."""
    orchestrator = get_orchestrator()

    if orchestrator.resume_job(job_id):
        return ApiResponse(success=True, message="Resume requested")
    else:
        raise HTTPException(status_code=400, detail="Cannot resume job")


@router.post("/{job_id}/cancel", response_model=ApiResponse)
async def cancel_job(job_id: str) -> ApiResponse:
    """Cancel a running or paused job."""
    orchestrator = get_orchestrator()

    if orchestrator.cancel_job(job_id):
        return ApiResponse(success=True, message="Cancellation requested")
    else:
        raise HTTPException(status_code=400, detail="Cannot cancel job")


@router.delete("/{job_id}", response_model=ApiResponse)
async def delete_job(job_id: str) -> ApiResponse:
    """Delete a completed, failed, or cancelled job and its files."""
    orchestrator = get_orchestrator()

    if orchestrator.delete_job(job_id):
        return ApiResponse(success=True, message="Job deleted")
    else:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete job (must be completed, failed, or cancelled)",
        )


@router.get("/{job_id}/logs")
async def get_job_logs(job_id: str, limit: int = Query(default=100, ge=1, le=1000)):
    """Get logs for a job."""
    orchestrator = get_orchestrator()
    job = orchestrator.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return {"logs": job.logs[-limit:]}


@router.get("/{job_id}/steps/{step_name}/artifacts")
async def list_step_artifacts(job_id: str, step_name: str):
    """List artifacts for a specific step."""
    orchestrator = get_orchestrator()
    job = orchestrator.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if step_name not in job.step_results:
        raise HTTPException(status_code=404, detail="Step not found")

    result = job.step_results[step_name]
    artifacts = [
        ArtifactInfo(
            name=a.name,
            path=str(a.path),
            artifact_type=a.artifact_type,
            size_bytes=a.size_bytes,
            exists=a.exists(),
            metadata=a.metadata,
        )
        for a in result.artifacts
    ]

    return {"artifacts": artifacts}


@router.get("/{job_id}/steps/{step_name}/artifacts.tar.gz")
async def download_step_artifacts(job_id: str, step_name: str):
    """Download all artifacts for a step as a tar.gz archive."""
    orchestrator = get_orchestrator()
    job = orchestrator.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if step_name not in job.step_results:
        raise HTTPException(status_code=404, detail="Step not found")

    result = job.step_results[step_name]

    # Create tar.gz in memory
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for artifact in result.artifacts:
            if artifact.exists():
                tar.add(artifact.path, arcname=artifact.name)

    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="application/gzip",
        headers={
            "Content-Disposition": f"attachment; filename={job_id}_{step_name}_artifacts.tar.gz"
        },
    )


@router.get("/{job_id}/artifacts/{artifact_path:path}")
async def download_artifact(job_id: str, artifact_path: str):
    """Download a specific artifact file."""
    orchestrator = get_orchestrator()
    job = orchestrator.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Find artifact
    for result in job.step_results.values():
        for artifact in result.artifacts:
            if artifact.name == artifact_path or str(artifact.path).endswith(
                artifact_path
            ):
                if artifact.exists():
                    return FileResponse(
                        artifact.path,
                        filename=artifact.name,
                        media_type="application/octet-stream",
                    )
                else:
                    raise HTTPException(
                        status_code=404, detail="Artifact file not found"
                    )

    raise HTTPException(status_code=404, detail="Artifact not found")
