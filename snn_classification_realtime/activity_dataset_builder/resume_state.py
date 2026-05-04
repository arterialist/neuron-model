"""Deterministic sample ordering and JSON manifest for resumable activity builds."""

from __future__ import annotations

import hashlib
import json
import os
import random
from typing import Any

MANIFEST_NAME = "activity_build_state.json"
MANIFEST_VERSION = 1


def deterministic_shuffle_seed(
    network_path: str,
    dataset_name: str,
    images_per_label: int,
    dataset_base: str,
    ticks_per_image: int,
    ablation: str,
    cifar10_color_normalization_factor: float,
) -> int:
    """Stable 32-bit seed from build inputs (same CLI → same per-label image draws)."""
    parts = "|".join(
        str(p)
        for p in (
            os.path.abspath(network_path),
            dataset_name,
            images_per_label,
            dataset_base,
            ticks_per_image,
            ablation,
            cifar10_color_normalization_factor,
        )
    )
    digest = hashlib.sha256(parts.encode()).digest()
    return int.from_bytes(digest[:4], "big", signed=False) % (2**31)


def build_per_label_plan(
    num_classes: int,
    label_to_indices: dict[int, list[int]],
    images_per_label: int,
    shuffle_seed: int,
) -> list[tuple[int, list[int]]]:
    """Return (label_idx, chosen_image_indices) for every label in processing order."""
    rng = random.Random(shuffle_seed)
    plan: list[tuple[int, list[int]]] = []
    for label_idx in range(num_classes):
        indices = label_to_indices.get(label_idx, [])
        if not indices:
            plan.append((label_idx, []))
            continue
        k = min(images_per_label, len(indices))
        chosen = rng.sample(indices, k)
        plan.append((label_idx, chosen))
    return plan


def manifest_path(dataset_dir: str) -> str:
    return os.path.join(dataset_dir, MANIFEST_NAME)


def load_manifest(dataset_dir: str) -> dict[str, Any] | None:
    path = manifest_path(dataset_dir)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_manifest(dataset_dir: str, data: dict[str, Any]) -> None:
    path = manifest_path(dataset_dir)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def cumulative_samples_before_label(
    plan: list[tuple[int, list[int]]], stop_before_label_idx: int
) -> int:
    total = 0
    for label_idx, chosen in plan:
        if label_idx >= stop_before_label_idx:
            break
        total += len(chosen)
    return total


def infer_resume_from_row_count(
    num_rows: int, plan: list[tuple[int, list[int]]]
) -> tuple[int, int]:
    """Return (committed_global_samples, next_label_idx) when manifest is missing.

    Assumes rows 0..committed-1 are contiguous completed prefixes per deterministic plan.
    If the file has extra rows beyond a full plan, committed is total plan size.
    """
    g = 0
    for label_idx, chosen in plan:
        ln = len(chosen)
        if ln == 0:
            continue
        if g + ln > num_rows:
            return g, label_idx
        g += ln
    if num_rows > g:
        return g, len(plan)
    return g, len(plan)
