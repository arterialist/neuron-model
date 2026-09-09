"""Analytic, data-free visual kernels for the hardwired (HW) feature arm.

Every kernel here is generated from closed-form math with NO reference to any dataset — so the
resulting features are label-free AND data-free by construction, satisfying the hard constraint.
These are the priors the current random reservoir lacks: oriented edge detectors (V1 simple
cells) and color-opponent center-surround (retinal ganglion / LGN).

A "bank" is an array of shape (F, in_ch, k, k): F output filters, each a k×k kernel over in_ch
input channels. wire_designed.py maps bank[f, c, ky, kx] onto the PAULA synapse whose id decodes
as (c·k + ky)·k + kx, and repeats the same kernel at every spatial position (translation
invariance — the thing `filters` alone does not provide).

Kernels are zero-mean (so a flat input gives ~zero drive; the detector responds to structure,
not brightness) except the pure luminance/DoG channels where the sign IS the signal.
"""
from __future__ import annotations

import numpy as np


def _grid(k: int):
    """Centered coordinate grid for a k×k kernel."""
    c = (k - 1) / 2.0
    y, x = np.mgrid[0:k, 0:k]
    return (x - c).astype(np.float64), (y - c).astype(np.float64)


def gabor(k: int, theta: float, wavelength: float, phase: float, sigma: float,
          aspect: float = 0.6) -> np.ndarray:
    """One real Gabor kernel (k×k), zero-mean, unit L2 norm.
    theta: orientation (rad); wavelength: sinusoid period (px); phase: 0=even/bar, pi/2=odd/edge;
    sigma: envelope width (px); aspect: envelope elongation along the bar."""
    x, y = _grid(k)
    xr = x * np.cos(theta) + y * np.sin(theta)
    yr = -x * np.sin(theta) + y * np.cos(theta)
    env = np.exp(-(xr ** 2 + (aspect ** 2) * yr ** 2) / (2 * sigma ** 2))
    car = np.cos(2 * np.pi * xr / wavelength + phase)
    g = env * car
    g = g - g.mean()                              # zero-mean: no response to flat field
    n = np.linalg.norm(g)
    return g / n if n > 0 else g


def dog(k: int, sigma_c: float, sigma_s: float, polarity: float = 1.0) -> np.ndarray:
    """Difference-of-Gaussians center-surround (k×k), zero-mean, unit L2 norm.
    polarity +1 = ON-center/OFF-surround, -1 = OFF-center. sigma_s > sigma_c."""
    x, y = _grid(k)
    r2 = x ** 2 + y ** 2
    c = np.exp(-r2 / (2 * sigma_c ** 2)); c /= c.sum()
    s = np.exp(-r2 / (2 * sigma_s ** 2)); s /= s.sum()
    d = polarity * (c - s)
    d = d - d.mean()
    n = np.linalg.norm(d)
    return d / n if n > 0 else d


# ---- color-opponent projections over the 3 RGB input channels ----
# Each entry maps the single-channel spatial kernel onto RGB with an opponent weighting, giving
# retinal-ganglion-style channels. Weights sum to ~0 across channels -> chromatic opponency.
OPPONENT = {
    "lum":  np.array([1.0, 1.0, 1.0]) / 3.0,      # luminance (R+G+B)
    "rg":   np.array([1.0, -1.0, 0.0]),           # red-green
    "by":   np.array([-0.5, -0.5, 1.0]),          # blue-yellow
}


def gabor_bank(k: int, in_ch: int = 3, orientations: int = 4, scales=(0.5, 0.8),
               phases=(0.0, np.pi / 2), color: str = "lum") -> tuple[np.ndarray, list[dict]]:
    """Bank of oriented Gabors across orientations × scales × phases, projected onto in_ch.

    color: 'lum' = grayscale Gabors on all channels (luminance edges);
           'opponent' = each (ori,scale,phase) replicated once per opponent channel (lum/rg/by),
                        i.e. oriented chromatic edges. Returns (bank (F,in_ch,k,k), meta list).
    """
    chans = ["lum"] if color == "lum" else list(OPPONENT.keys())
    kernels, meta = [], []
    for ch in chans:
        proj = OPPONENT[ch] if in_ch == 3 else np.ones(in_ch) / in_ch
        for si, sc in enumerate(scales):
            sigma = sc * k / 2.0
            wavelength = max(2.0, 2.0 * sigma)
            for oi in range(orientations):
                theta = np.pi * oi / orientations
                for ph in phases:
                    g = gabor(k, theta, wavelength, ph, sigma)     # (k,k)
                    if in_ch == 3:
                        kern = np.stack([proj[c] * g for c in range(3)])  # (3,k,k)
                    else:
                        kern = np.stack([g for _ in range(in_ch)])
                    kernels.append(kern.astype(np.float32))
                    meta.append(dict(kind="gabor", color=ch, orientation=round(float(theta), 3),
                                     scale=sc, phase=round(float(ph), 3)))
    return np.stack(kernels), meta


def dog_bank(k: int, in_ch: int = 3, scales=(0.6,), color: str = "opponent"
             ) -> tuple[np.ndarray, list[dict]]:
    """Center-surround bank: ON + OFF at each scale, over luminance or opponent channels.
    This is the retinal/LGN front-end (L0)."""
    chans = ["lum"] if color == "lum" else list(OPPONENT.keys())
    kernels, meta = [], []
    for ch in chans:
        proj = OPPONENT[ch] if in_ch == 3 else np.ones(in_ch) / in_ch
        for sc in scales:
            sigma_c = sc * k / 4.0
            sigma_s = sigma_c * 2.2
            for pol in (1.0, -1.0):
                d = dog(k, sigma_c, sigma_s, pol)                  # (k,k)
                kern = (np.stack([proj[c] * d for c in range(3)]) if in_ch == 3
                        else np.stack([d for _ in range(in_ch)]))
                kernels.append(kern.astype(np.float32))
                meta.append(dict(kind="dog", color=ch, scale=sc, polarity=pol))
    return np.stack(kernels), meta


def describe(meta: list[dict]) -> str:
    return f"{len(meta)} kernels: " + ", ".join(
        sorted({f"{m['kind']}/{m.get('color','')}" for m in meta}))


if __name__ == "__main__":
    # self-check: shapes, zero-mean, unit norm
    for k in (4, 6, 7):
        gb, gm = gabor_bank(k, 3, orientations=4, scales=(0.5, 0.8), color="lum")
        db, dm = dog_bank(k, 3, color="opponent")
        assert gb.shape[1:] == (3, k, k) and db.shape[1:] == (3, k, k)
        print(f"k={k}: gabor bank {gb.shape} ({describe(gm)}); dog bank {db.shape} ({describe(dm)})")
        # each gabor channel-kernel zero-mean
        flat = gb.reshape(gb.shape[0], -1)
        print(f"       gabor |mean|max={np.abs(flat.mean(1)).max():.2e} "
              f"L2 range=[{np.linalg.norm(gb.reshape(len(gb),-1),axis=1).min():.3f},"
              f"{np.linalg.norm(gb.reshape(len(gb),-1),axis=1).max():.3f}]")
