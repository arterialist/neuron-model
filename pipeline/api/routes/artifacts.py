"""
Artifact download routes.
"""

import io
import os
import tarfile
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse

from pipeline.api.routes.jobs import get_orchestrator
from pipeline.api.models import ArtifactInfo


router = APIRouter(prefix="/api/artifacts", tags=["artifacts"])


@router.get("/{job_id}")
async def list_all_job_artifacts(job_id: str):
    """List all artifacts for a job across all steps."""
    orchestrator = get_orchestrator()
    job = orchestrator.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    all_artifacts = []
    for step_name, result in job.step_results.items():
        for artifact in result.artifacts:
            all_artifacts.append(
                {
                    "step": step_name,
                    "name": artifact.name,
                    "path": str(artifact.path),
                    "artifact_type": artifact.artifact_type,
                    "size_bytes": artifact.size_bytes,
                    "exists": artifact.exists(),
                    "metadata": artifact.metadata,
                }
            )

    return {"artifacts": all_artifacts}


@router.get("/{job_id}/download")
async def download_selected_artifacts(
    job_id: str,
    files: List[str] = Query(
        default=[], description="List of artifact names to download"
    ),
):
    """Download selected artifacts as a tar.gz archive.

    If no files are specified, downloads all artifacts.
    """
    orchestrator = get_orchestrator()
    job = orchestrator.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Collect artifacts to include
    artifacts_to_include = []
    for step_name, result in job.step_results.items():
        for artifact in result.artifacts:
            if not files or artifact.name in files:
                if artifact.exists():
                    artifacts_to_include.append((step_name, artifact))

    if not artifacts_to_include:
        raise HTTPException(status_code=404, detail="No artifacts found to download")

    # Create tar.gz in memory
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for step_name, artifact in artifacts_to_include:
            arcname = f"{step_name}/{artifact.name}"
            tar.add(artifact.path, arcname=arcname)

    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="application/gzip",
        headers={
            "Content-Disposition": f"attachment; filename={job_id}_artifacts.tar.gz"
        },
    )


@router.get("/{job_id}/all.tar.gz")
async def download_all_artifacts(job_id: str):
    """Download all artifacts for a job as a single tar.gz archive."""
    orchestrator = get_orchestrator()
    job = orchestrator.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Create tar.gz in memory
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for step_name, result in job.step_results.items():
            for artifact in result.artifacts:
                if artifact.exists():
                    arcname = f"{step_name}/{artifact.name}"
                    tar.add(artifact.path, arcname=arcname)

    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="application/gzip",
        headers={
            "Content-Disposition": f"attachment; filename={job_id}_all_artifacts.tar.gz"
        },
    )
