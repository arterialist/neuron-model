"""Shared utilities for visualization steps."""

import os
import re
from pathlib import Path


def subprocess_output_to_log_lines(stdout: str = "", stderr: str = "") -> list[str]:
    """Convert subprocess stdout/stderr to sanitized log lines for step logs."""
    combined = ((stdout or "") + "\n" + (stderr or "")).strip()
    if not combined:
        return []
    ansi_escape = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\r")
    # Strip control chars that break JSON (U+0000..U+001F except \n \r \t)
    control_chars = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
    lines = []
    for line in combined.split("\n"):
        line = ansi_escape.sub("", line)
        line = control_chars.sub(" ", line)
        line = line.replace("\n", " ").replace("\r", " ").strip()
        if not line:
            continue
        # Skip tqdm progress bar noise
        if re.search(r"\d+%\|[^\|]*\|", line) or "it/s]" in line:
            continue
        lines.append(line)
    return lines


def get_project_root() -> Path:
    """Resolve project root. In Docker: /app/project. Locally: parent of pipeline."""
    # pipeline/visualizations/utils.py -> parent.parent.parent = project root or /app
    base = Path(__file__).resolve().parent.parent.parent
    if os.environ.get("PIPELINE_PROJECT_ROOT"):
        return Path(os.environ["PIPELINE_PROJECT_ROOT"])
    if (base / "project").exists() and (base / "pipeline").exists():
        return base / "project"
    return base
