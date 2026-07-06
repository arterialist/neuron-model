"""Multi-resolution retina: a small sharp fovea + a blurred peripheral gist.

Biological foveation is not one uniform patch. It is a tiny high-acuity fovea
surrounded by low-acuity periphery that carries coarse layout ("where is stuff")
without detail. This retina renders both, concentric on a movable center, onto a
common grid so a single perception net can consume them as extra channels:

    output: (2*C, grid, grid)
      channels [0:C]   = fovea    (small window, upsampled -> sharp detail)
      channels [C:2C]  = periphery (large window, squeezed through a low-res
                                    bottleneck -> blurred gist)

The perception net is built with `channels = 2*C`, `size = grid`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class Retina:
    image_h: int
    image_w: int
    grid: int = 16            # net input resolution (per region)
    fovea_extent: int = 10    # side of the sharp central window (px)
    periph_extent: int = 0    # side of the coarse window (0 = whole image)
    periph_bottleneck: int = 6  # low-res bottleneck that blurs the periphery
    cy: float = 0.0
    cx: float = 0.0

    def __post_init__(self) -> None:
        if self.periph_extent <= 0:
            self.periph_extent = min(self.image_h, self.image_w)
        self.fovea_extent = int(min(self.fovea_extent, self.image_h, self.image_w))
        self.periph_extent = int(min(self.periph_extent, self.image_h, self.image_w))
        self.center()

    # --- position (center of gaze) --------------------------------------
    def center(self) -> None:
        self.cy = self.image_h / 2.0
        self.cx = self.image_w / 2.0

    def set_center(self, cy: float, cx: float) -> None:
        self.cy = float(np.clip(cy, 0.0, self.image_h))
        self.cx = float(np.clip(cx, 0.0, self.image_w))

    def move(self, dy: float, dx: float) -> None:
        self.set_center(self.cy + dy, self.cx + dx)

    def _window(self, image: torch.Tensor, extent: int) -> tuple[torch.Tensor, int, int]:
        """extent x extent window centered on gaze, clamped inside the image."""
        half = extent / 2.0
        y0 = int(round(self.cy - half))
        x0 = int(round(self.cx - half))
        y0 = min(max(y0, 0), self.image_h - extent)
        x0 = min(max(x0, 0), self.image_w - extent)
        return image[:, y0:y0 + extent, x0:x0 + extent], y0, x0

    def fovea_box(self) -> tuple[int, int, int]:
        _, y0, x0 = self._window(torch.zeros(1, self.image_h, self.image_w),
                                 self.fovea_extent)
        return y0, x0, self.fovea_extent

    def periph_box(self) -> tuple[int, int, int]:
        _, y0, x0 = self._window(torch.zeros(1, self.image_h, self.image_w),
                                 self.periph_extent)
        return y0, x0, self.periph_extent

    @staticmethod
    def _resize(win: torch.Tensor, size: int) -> torch.Tensor:
        return F.interpolate(win.unsqueeze(0), size=(size, size),
                             mode="bilinear", align_corners=False).squeeze(0)

    def render(self, image: torch.Tensor) -> torch.Tensor:
        """Return the (2*C, grid, grid) retina image for the current gaze."""
        if image.ndim == 2:
            image = image.unsqueeze(0)
        fov_win, _, _ = self._window(image, self.fovea_extent)
        per_win, _, _ = self._window(image, self.periph_extent)
        fovea = self._resize(fov_win, self.grid)                       # sharp
        per_small = self._resize(per_win, self.periph_bottleneck)      # low-pass
        periph = self._resize(per_small, self.grid)                    # blurred gist
        return torch.cat([fovea, periph], dim=0)

    @property
    def out_channels_factor(self) -> int:
        return 2
