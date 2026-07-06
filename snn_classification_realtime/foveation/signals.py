"""Free-energy proxy and structured-stability reward.

Naive free-energy minimization fixates on the flattest patch (the dark-room
problem): a blank region is the easiest to predict, so it settles fastest.
The reward that should instead peak on objects is *structured* stability:

    reward = settledness x informativeness

- settledness: the population state stops changing between ticks (attractor
  reached). Flat and structured inputs both settle, so this alone is not
  enough.
- informativeness: the settled state is rich, not the dead rest state
  (variance of S across the population; participation of the assembly).

The product requires BOTH: a flat patch is settled-but-dead (reward ~ 0), an
object is settled-and-rich (reward high). This is a crude expected-free-energy
with the epistemic term restored. Whether it actually tracks object location
on textured CIFAR backgrounds is an empirical question (see EXP1).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from snn_classification_realtime.foveation.perception import PopulationState


def _l2(x: np.ndarray) -> float:
    return float(np.sqrt(np.sum(x * x)))


@dataclass
class FreeEnergyProbe:
    """Streaming free-energy proxy over the perception population.

    Call update(state) each tick. Reads settledness (temporal), informativeness
    (spatial), and their product (the structured-stability reward).
    """

    history: int = 6
    _S_prev: np.ndarray | None = field(default=None, repr=False)
    _settle: deque = field(default_factory=lambda: deque(maxlen=6), repr=False)
    _info: deque = field(default_factory=lambda: deque(maxlen=6), repr=False)
    _part: deque = field(default_factory=lambda: deque(maxlen=6), repr=False)

    def __post_init__(self) -> None:
        self._settle = deque(maxlen=self.history)
        self._info = deque(maxlen=self.history)
        self._part = deque(maxlen=self.history)

    def reset(self) -> None:
        self._S_prev = None
        self._settle.clear()
        self._info.clear()
        self._part.clear()

    def update(self, state: PopulationState) -> None:
        S = state.S
        # Informativeness: spread of the population state (rich vs dead).
        info = float(np.std(S))
        # Participation: fraction of the assembly spiking this tick.
        part = float(np.mean(state.O))
        # Settledness: how little the state moved since last tick.
        if self._S_prev is None:
            settle = 0.0
        else:
            change = _l2(S - self._S_prev) / (_l2(S) + 1e-6)
            settle = float(np.exp(-change))
        self._S_prev = S.copy()
        self._settle.append(settle)
        self._info.append(info)
        self._part.append(part)

    # --- scalar readouts (means over the recent window) -----------------
    @property
    def settledness(self) -> float:
        return float(np.mean(self._settle)) if self._settle else 0.0

    @property
    def informativeness(self) -> float:
        return float(np.mean(self._info)) if self._info else 0.0

    @property
    def participation(self) -> float:
        return float(np.mean(self._part)) if self._part else 0.0

    @property
    def reward(self) -> float:
        """Structured-stability reward = settledness x informativeness."""
        return self.settledness * self.informativeness

    @property
    def free_energy(self) -> float:
        """Scalar FE proxy (low on a good structured fixation). Used by the
        klinotaxis drift as the quantity to descend."""
        return -self.reward


@dataclass
class StabilityReward:
    """Map the free-energy proxy to ALERM modulators (m0 stress, m1 reward).

    m1 rises when the network is settled-and-rich (consolidate -> fixate).
    m0 rises when the network cannot settle (stress -> search/saccade).
    Both are tanh-saturated like the C. elegans transduction so the motor
    limit cycle is not destroyed at extremes.
    """

    info_scale: float = 1.0  # normalizer for informativeness (set from a scan)
    k_reward: float = 3.0
    k_stress: float = 3.0

    def modulators(self, probe: FreeEnergyProbe) -> tuple[float, float]:
        info_norm = probe.informativeness / (self.info_scale + 1e-8)
        info_norm = min(info_norm, 2.0)
        r = probe.settledness * info_norm
        m1 = float(np.tanh(self.k_reward * r))
        # Stress from failure to settle.
        m0 = float(np.tanh(self.k_stress * (1.0 - probe.settledness)))
        return m0, m1
