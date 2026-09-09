"""Retinal convolution front-end: the ganglion/LGN stage of the eye.

Biological retina is not a raw pixel array -- bipolar/ganglion cells apply
center-surround (difference-of-Gaussians) and oriented receptive fields, then
ON/OFF cells half-wave RECTIFY. Convolution alone is linear (cannot raise a linear
readout's separability); the rectification is the nonlinearity that exposes class
structure a simple decoder can read. Measured lift on a single foveated glimpse:
grayscale 0.22 -> 0.34, color 0.29 -> 0.38 linear separability.

Output feeds the PAULA substrate in place of raw retina pixels. Fixed (not learned)
kernels: this is sensory preprocessing, the substrate + teacher do the learning.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _dog(sigma1: float, sigma2: float, k: int = 5) -> np.ndarray:
    ax = np.arange(k) - k // 2
    xx, yy = np.meshgrid(ax, ax)
    g1 = np.exp(-(xx**2 + yy**2) / (2 * sigma1**2)); g1 /= g1.sum()
    g2 = np.exp(-(xx**2 + yy**2) / (2 * sigma2**2)); g2 /= g2.sum()
    return g1 - g2


def _edge_kernels() -> list[np.ndarray]:
    sob = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)
    ks = [sob, sob.T, np.rot90(sob), np.rot90(sob).T]      # 0/90/180/270 oriented
    out = []
    for k in ks:
        kk = np.zeros((5, 5)); kk[1:4, 1:4] = k; out.append(kk)
    return out


def _gabor(theta: float, sigma: float = 1.6, lam: float = 3.0, k: int = 7) -> np.ndarray:
    """Oriented Gabor (cosine * Gaussian) -- a smoother, tuned edge detector than the
    5x5 Sobel, at an arbitrary orientation theta."""
    ax = np.arange(k) - k // 2
    xx, yy = np.meshgrid(ax, ax)
    xr = xx * np.cos(theta) + yy * np.sin(theta)
    yr = -xx * np.sin(theta) + yy * np.cos(theta)
    g = np.exp(-(xr**2 + yr**2) / (2 * sigma**2)) * np.cos(2 * np.pi * xr / lam)
    return g - g.mean()


def _build_kernels(bank: str = "default") -> torch.Tensor:
    if bank == "rich":
        # 4 center-surround scales + 8 oriented Gabors -- more multiscale/orientation
        # coverage, the locality-preserving front-end the working conv pipeline relies on
        ks = [_dog(0.5, 1.0), _dog(0.8, 1.6), _dog(1.2, 2.4), _dog(1.8, 3.6)]
        ks += [_gabor(t) for t in np.linspace(0, np.pi, 8, endpoint=False)]
    else:
        ks = [_dog(0.6, 1.2), _dog(1.0, 2.0)] + _edge_kernels()  # 2 center-surround + 4 edge
    kmax = max(k.shape[0] for k in ks)
    ks = [_pad(k, kmax) for k in ks]
    return torch.tensor(np.stack(ks), dtype=torch.float32).unsqueeze(1)   # (F,1,K,K)


def _pad(k: np.ndarray, size: int) -> np.ndarray:
    """Center-pad a square kernel to size x size (so a bank can mix kernel sizes)."""
    if k.shape[0] == size:
        return k
    out = np.zeros((size, size)); off = (size - k.shape[0]) // 2
    out[off:off + k.shape[0], off:off + k.shape[1]] = k
    return out


class RetinalConv:
    """Fixed conv front-end. feature_dim() lets the substrate size its input layer;
    encode() returns a [-1,1] tensor so the dense input mapper (which does (v+1)/2)
    recovers the [0,1] rectified feature back."""

    def __init__(self, pool: int = 2, bank: str = "default"):
        self.kernels = _build_kernels(bank)
        self.pad = self.kernels.shape[-1] // 2
        self.n_filters = self.kernels.shape[0]
        self.pool = pool
        self.scale = 1.0                    # set by fit(); robust contrast normaliser

    def _raw(self, retina_out: torch.Tensor) -> torch.Tensor:
        """retina_out: (C, H, W) -> (2*C*F, H/pool, W/pool) ON/OFF rectified maps."""
        if retina_out.ndim == 2:
            retina_out = retina_out.unsqueeze(0)
        x = retina_out.unsqueeze(1)                       # (C,1,H,W)
        y = F.conv2d(x, self.kernels, padding=self.pad)   # (C,F,H,W)
        on = F.relu(y); off = F.relu(-y)                  # ON / OFF rectification
        if self.pool > 1:
            on = F.avg_pool2d(on, self.pool); off = F.avg_pool2d(off, self.pool)
        C, Fn, h, w = on.shape
        return torch.cat([on, off], dim=1).reshape(-1, h, w)   # (2*C*F, h, w)

    def fit(self, retina_outs: list[torch.Tensor]) -> "RetinalConv":
        """Set a fixed contrast scale from a sample (99th pct) so features ~[0,1]."""
        vals = torch.cat([self._raw(r).flatten() for r in retina_outs])
        self.scale = float(torch.quantile(vals[vals > 0], 0.99)) if (vals > 0).any() else 1.0
        self.scale = max(self.scale, 1e-6)
        return self

    def feature_dim(self, channels: int, grid: int) -> int:
        gp = grid // self.pool
        return 2 * channels * self.n_filters * gp * gp

    def encode(self, retina_out: torch.Tensor) -> torch.Tensor:
        """Return a [-1,1] tensor (shape (2*C*F, h, w)) for the input mapper."""
        f = self._raw(retina_out) / self.scale
        f = torch.clamp(f, 0.0, 1.0)
        return 2.0 * f - 1.0
