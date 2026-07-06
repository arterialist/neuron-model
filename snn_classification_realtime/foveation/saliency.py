"""Image saliency proxies and fixation/reward correlation metrics.

CIFAR-10 has no object masks, so "did the fovea find the object?" is scored
against image-derived saliency (edge/contrast energy) and against a center
prior (CIFAR objects are center-biased). A reward that tracks saliency but not
merely center is the signal we want; a reward that tracks flatness (negative
correlation with saliency) is the dark-room failure.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr


def edge_energy_map(image: torch.Tensor) -> np.ndarray:
    """Per-pixel gradient magnitude of a (C,H,W) image (channel-averaged)."""
    arr = image.detach().cpu().numpy().astype(np.float32)
    if arr.ndim == 3:
        arr = arr.mean(axis=0)
    gy, gx = np.gradient(arr)
    return np.sqrt(gy * gy + gx * gx)


def patch_saliency(
    image: torch.Tensor, positions: list[tuple[int, int]], size: int
) -> np.ndarray:
    """Mean edge energy within the k x k patch at each grid position."""
    emap = edge_energy_map(image)
    out = np.empty(len(positions), dtype=np.float32)
    for i, (y, x) in enumerate(positions):
        out[i] = float(emap[y : y + size, x : x + size].mean())
    return out


def center_prior(
    positions: list[tuple[int, int]], image_h: int, image_w: int, size: int
) -> np.ndarray:
    """Negative distance of each patch center from the image center (higher =
    more central), for disentangling saliency from a pure center bias."""
    cy, cx = image_h / 2.0, image_w / 2.0
    out = np.empty(len(positions), dtype=np.float32)
    for i, (y, x) in enumerate(positions):
        py, px = y + size / 2.0, x + size / 2.0
        out[i] = -float(np.hypot(py - cy, px - cx))
    return out


def safe_corr(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(pearson, spearman); 0 if a series is constant."""
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0.0, 0.0
    p = float(pearsonr(a, b)[0])
    s = float(spearmanr(a, b)[0])
    return p, s


def partial_corr_saliency_given_center(
    reward: np.ndarray, saliency: np.ndarray, center: np.ndarray
) -> float:
    """Correlation of reward with saliency after regressing out the center
    prior from both (does reward track saliency beyond mere centrality?)."""
    if np.std(reward) < 1e-9 or np.std(saliency) < 1e-9 or np.std(center) < 1e-9:
        return 0.0

    def resid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
        A = np.vstack([x, np.ones_like(x)]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        return y - A @ coef

    r_res = resid(reward, center)
    s_res = resid(saliency, center)
    if np.std(r_res) < 1e-9 or np.std(s_res) < 1e-9:
        return 0.0
    return float(pearsonr(r_res, s_res)[0])
