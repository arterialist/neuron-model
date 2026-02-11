"""
Base classes for pipeline steps.

Defines the abstract interface that all pipeline steps must implement.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable


class StepStatus(str, Enum):
    """Status of a pipeline step."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class StepCancelledException(Exception):
    """Exception raised when a step is cancelled."""

    pass


@dataclass
class Artifact:
    """Represents an output artifact from a pipeline step."""

    name: str
    path: Path
    artifact_type: str  # e.g., "model", "dataset", "visualization", "log"
    size_bytes: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def exists(self) -> bool:
        """Check if the artifact file exists."""
        return self.path.exists()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "path": str(self.path),
            "artifact_type": self.artifact_type,
            "size_bytes": self.size_bytes,
            "metadata": self.metadata,
            "exists": self.exists(),
        }


@dataclass
class StepContext:
    """Context passed to each pipeline step during execution."""

    job_id: str
    job_name: str
    output_dir: Path
    config: Any  # Step-specific config section
    previous_artifacts: dict[str, list[Artifact]] = field(default_factory=dict)
    logger: logging.Logger | None = None
    check_control_signals: Callable[[], None] = lambda: None

    def get_artifact(self, step_name: str, artifact_name: str) -> Artifact | None:
        """Get a specific artifact from a previous step."""
        for artifact in self.previous_artifacts.get(step_name, []):
            if artifact.name == artifact_name:
                return artifact
        return None

    def get_artifact_path(self, step_name: str, artifact_name: str) -> Path | None:
        """Get the path to a specific artifact from a previous step."""
        artifact = self.get_artifact(step_name, artifact_name)
        return artifact.path if artifact else None


@dataclass
class StepResult:
    """Result returned by a pipeline step after execution."""

    status: StepStatus
    artifacts: list[Artifact] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    logs: list[str] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float | None:
        """Calculate the duration of the step execution."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "status": self.status.value,
            "artifacts": [a.to_dict() for a in self.artifacts],
            "metrics": self.metrics,
            "error_message": self.error_message,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "logs": self.logs[-100:]
            if len(self.logs) > 100
            else self.logs,  # Limit logs
        }


class PipelineStep(ABC):
    """Abstract base class for all pipeline steps.

    Each step must implement:
    - name: A unique identifier for the step
    - run: The main execution logic
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name identifier for this step."""
        pass

    @property
    def display_name(self) -> str:
        """Human-readable name for display purposes."""
        return self.name.replace("_", " ").title()

    @abstractmethod
    def run(self, context: StepContext) -> StepResult:
        """Execute the step.

        Args:
            context: Step execution context with config and previous artifacts

        Returns:
            StepResult with status, artifacts, and metrics
        """
        pass

    def validate_config(self, config: Any) -> list[str]:
        """Validate the step-specific configuration.

        Args:
            config: The configuration section for this step

        Returns:
            List of validation error messages (empty if valid)
        """
        return []

    def cleanup(self, context: StepContext) -> None:
        """Clean up any temporary resources after step execution.

        Override this method if your step creates temporary files or
        resources that should be cleaned up after execution.
        """
        pass


class StepRegistry:
    """Registry for pipeline steps.

    Allows dynamic registration and lookup of step implementations.
    """

    _steps: dict[str, type[PipelineStep]] = {}

    @classmethod
    def register(cls, step_class: type[PipelineStep]) -> type[PipelineStep]:
        """Register a step class.

        Can be used as a decorator:
            @StepRegistry.register
            class MyStep(PipelineStep):
                ...
        """
        # Create instance to get name
        instance = step_class.__new__(step_class)
        step_name = instance.name
        cls._steps[step_name] = step_class
        return step_class

    @classmethod
    def get(cls, name: str) -> type[PipelineStep] | None:
        """Get a registered step class by name."""
        return cls._steps.get(name)

    @classmethod
    def all_steps(cls) -> dict[str, type[PipelineStep]]:
        """Get all registered steps."""
        return cls._steps.copy()

    @classmethod
    def step_names(cls) -> list[str]:
        """Get names of all registered steps."""
        return list(cls._steps.keys())
