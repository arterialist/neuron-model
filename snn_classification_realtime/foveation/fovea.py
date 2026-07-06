"""Fixed-resolution retina that resamples a patch from a larger image.

The retina is k x k. Its top-left position (fy, fx) moves over the image; the
network always receives a full k x k patch, so every input neuron is driven no
matter where the eye looks. This is the correct framing of a fovea: fixed
foveal resolution, the eye moves the world onto it (as opposed to feeding a
crop into a full-image network, which would leave 3/4 of its inputs silent).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class Fovea:
    """A movable k x k retina over a (C, H, W) image.

    Position is the top-left corner (fy, fx) in pixels, kept in
    [0, H-k] x [0, W-k]. Movement is continuous; integer crop by default,
    bilinear resample when subpixel=True.
    """

    image_h: int
    image_w: int
    size: int = 16
    subpixel: bool = False
    fy: float = 0.0
    fx: float = 0.0
    # Inhibition-of-return: a decaying penalty map over visited positions.
    ior_decay: float = 0.9
    ior_gain: float = 1.0
    _ior: np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.size > self.image_h or self.size > self.image_w:
            raise ValueError(
                f"Fovea size {self.size} exceeds image {self.image_h}x{self.image_w}"
            )
        # IOR grid indexed by integer top-left position.
        self._ior = np.zeros(
            (self.max_fy + 1, self.max_fx + 1), dtype=np.float32
        )
        self.center()

    @property
    def max_fy(self) -> int:
        return self.image_h - self.size

    @property
    def max_fx(self) -> int:
        return self.image_w - self.size

    def center(self) -> None:
        self.fy = self.max_fy / 2.0
        self.fx = self.max_fx / 2.0

    def set_position(self, fy: float, fx: float) -> None:
        self.fy = float(np.clip(fy, 0.0, self.max_fy))
        self.fx = float(np.clip(fx, 0.0, self.max_fx))

    def move(self, dvy: float, dvx: float) -> None:
        """Integrate a velocity step, clamped to bounds."""
        self.set_position(self.fy + dvy, self.fx + dvx)

    def crop(self, image: torch.Tensor) -> torch.Tensor:
        """Return the k x k patch under the retina.

        image: (C, H, W) tensor. Returns (C, k, k).
        """
        if image.ndim == 2:
            image = image.unsqueeze(0)
        if self.subpixel:
            return self._crop_bilinear(image)
        y0 = int(round(self.fy))
        x0 = int(round(self.fx))
        y0 = min(max(y0, 0), self.max_fy)
        x0 = min(max(x0, 0), self.max_fx)
        return image[:, y0 : y0 + self.size, x0 : x0 + self.size].clone()

    def _crop_bilinear(self, image: torch.Tensor) -> torch.Tensor:
        c, h, w = image.shape
        ys = torch.arange(self.size, dtype=torch.float32) + self.fy
        xs = torch.arange(self.size, dtype=torch.float32) + self.fx
        ys = ys.clamp(0, h - 1)
        xs = xs.clamp(0, w - 1)
        y0 = ys.floor().long()
        x0 = xs.floor().long()
        y1 = (y0 + 1).clamp(max=h - 1)
        x1 = (x0 + 1).clamp(max=w - 1)
        wy = (ys - y0.float()).view(1, -1, 1)
        wx = (xs - x0.float()).view(1, 1, -1)
        ia = image[:, y0][:, :, x0]
        ib = image[:, y0][:, :, x1]
        ic = image[:, y1][:, :, x0]
        idd = image[:, y1][:, :, x1]
        top = ia * (1 - wx) + ib * wx
        bot = ic * (1 - wx) + idd * wx
        return top * (1 - wy) + bot * wy

    # --- inhibition of return -------------------------------------------
    def mark_visited(self) -> None:
        yi = int(round(self.fy))
        xi = int(round(self.fx))
        self._ior *= self.ior_decay
        self._ior[yi, xi] = min(1.0, self._ior[yi, xi] + self.ior_gain)

    def ior_penalty(self, fy: float | None = None, fx: float | None = None) -> float:
        yi = int(round(self.fy if fy is None else fy))
        xi = int(round(self.fx if fx is None else fx))
        yi = min(max(yi, 0), self.max_fy)
        xi = min(max(xi, 0), self.max_fx)
        return float(self._ior[yi, xi])

    def reset_ior(self) -> None:
        self._ior[:] = 0.0

    def grid_positions(self, stride: int = 1) -> list[tuple[int, int]]:
        """All integer top-left positions on a stride grid (for scans)."""
        ys = list(range(0, self.max_fy + 1, stride))
        xs = list(range(0, self.max_fx + 1, stride))
        if ys[-1] != self.max_fy:
            ys.append(self.max_fy)
        if xs[-1] != self.max_fx:
            xs.append(self.max_fx)
        return [(y, x) for y in ys for x in xs]
