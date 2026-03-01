"""Load evaluation JSONL and optional summary JSON."""

from __future__ import annotations

import json
import os
from typing import Any


def load_jsonl(path: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def load_summary(path: str) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def resolve_summary_path(jsonl_path: str) -> str | None:
    base = os.path.dirname(os.path.abspath(jsonl_path))
    stem = os.path.splitext(os.path.basename(jsonl_path))[0]
    if stem.endswith("_eval"):
        candidate = os.path.join(base, stem + "_summary.json")
    else:
        candidate = os.path.join(base, stem.replace(".jsonl", "") + "_summary.json")
    if os.path.isfile(candidate):
        return candidate
    return None
