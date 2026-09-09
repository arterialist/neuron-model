"""Cluttered-canvas active-vision task: the object must be FOUND.

Place one CIFAR object at a random location on a larger canvas, surrounded by
distractor crops. The sharp fovea sees only a small window, so classification
requires saccading to the object; the wide (blurred) periphery gives the gist of
WHERE things are so a learned gaze can steer the fovea onto the target. This is the
regime where "learn where to look" should beat a random walk (CIFAR-32 was too
small -- the fovea already covered it).

Deterministic per index (seeded by index) so every arm/seed sees the same canvases.
"""

from __future__ import annotations

import numpy as np
import torch

from snn_classification_realtime.activity_dataset_builder.vision_datasets import (
    load_dataset_by_name,
)


class ClutteredCanvas:
    def __init__(self, base, canvas=72, n_distractors=4, distractor=16, seed=0):
        self.base = base
        self.canvas = int(canvas)
        self.nd = int(n_distractors)
        self.dsz = int(distractor)
        self.seed = int(seed)
        img0, _ = base[0]
        self.C = img0.shape[0]
        self.osz = img0.shape[1]

    def __len__(self):
        return len(self.base)

    def _object_loc(self, i):
        """Deterministic object top-left (oy,ox). Drawn FIRST so it's reproducible
        without replaying the distractor draws -> lets an oracle gaze fixate it."""
        rng = np.random.RandomState(self.seed * 1000003 + i)
        K, osz = self.canvas, self.osz
        return int(rng.randint(0, K - osz + 1)), int(rng.randint(0, K - osz + 1))

    def object_center(self, i):
        oy, ox = self._object_loc(i)
        return oy + self.osz / 2.0, ox + self.osz / 2.0        # (cy, cx)

    def __getitem__(self, i):
        rng = np.random.RandomState(self.seed * 1000003 + i)   # per-index determinism
        img, y = self.base[i]
        C, osz, K = self.C, self.osz, self.canvas
        oy = int(rng.randint(0, K - osz + 1)); ox = int(rng.randint(0, K - osz + 1))  # FIRST
        canvas = torch.full((C, K, K), -1.0)                   # bg = -1 (post-Normalize)
        for _ in range(self.nd):                               # distractor crops
            dimg, _ = self.base[rng.randint(0, len(self.base))]
            cy, cx = rng.randint(0, osz - self.dsz + 1), rng.randint(0, osz - self.dsz + 1)
            patch = dimg[:, cy:cy + self.dsz, cx:cx + self.dsz]
            py, px = rng.randint(0, K - self.dsz + 1), rng.randint(0, K - self.dsz + 1)
            canvas[:, py:py + self.dsz, px:px + self.dsz] = patch
        canvas[:, oy:oy + osz, ox:ox + osz] = img              # object painted on top
        return canvas, int(y)


def make_cluttered_cfg(name, canvas, n_distractors, distractor, seed, train=True):
    dc = load_dataset_by_name(name, train=train)
    dc.dataset = ClutteredCanvas(dc.dataset, canvas, n_distractors, distractor, seed)
    return dc
