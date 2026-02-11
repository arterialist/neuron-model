"""
Pydantic models for API requests and responses.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class JobCreate(BaseModel):
    """Request model for creating a new job."""

    config_yaml: str = Field(..., description="YAML configuration string")


class JobSummary(BaseModel):
    """Summary model for job listing."""

    job_id: str
    job_name: str
    status: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    current_step: Optional[str] = None
    error_message: Optional[str] = None


class ArtifactInfo(BaseModel):
    """Information about an artifact."""

    name: str
    path: str
    artifact_type: str
    size_bytes: int
    exists: bool
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StepInfo(BaseModel):
    """Information about a pipeline step."""

    step_name: str
    status: str
    artifacts: List[ArtifactInfo] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    logs: List[str] = Field(default_factory=list)
    error_message: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_seconds: Optional[float] = None
    storage_bytes: int = 0


class JobDetail(BaseModel):
    """Detailed job information."""

    job_id: str
    job_name: str
    status: str
    output_dir: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    current_step: Optional[str] = None
    error_message: Optional[str] = None
    total_duration_seconds: Optional[float] = None
    total_storage_bytes: int = 0
    steps: Dict[str, StepInfo] = Field(default_factory=dict)
    config: Dict[str, Any] = Field(default_factory=dict)


class WebhookCreate(BaseModel):
    """Request model for creating a webhook."""

    url: str
    events: List[str] = Field(
        default_factory=lambda: ["job_started", "job_completed", "job_failed"]
    )
    headers: Dict[str, str] = Field(default_factory=dict)


class WebhookInfo(BaseModel):
    """Information about a webhook."""

    id: int
    url: str
    events: List[str]
    headers: Dict[str, str] = Field(default_factory=dict)
    active: bool = True
    created_at: str


class LogEntry(BaseModel):
    """Log entry model."""

    timestamp: str
    level: str
    message: str


class ApiResponse(BaseModel):
    """Generic API response wrapper."""

    success: bool
    message: Optional[str] = None
    data: Optional[Any] = None
