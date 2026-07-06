"""Mini-brain core: a continuous, time-based closed loop.

A sparse, auto-calibrated PAULA substrate views CIFAR (grayscale or color) through
a retina. An ALERM teacher with access to the TRUE label injects neuromodulators
(m0 stress / m1 reward) by volume transmission; via the NATIVE path these shift
BOTH excitability (w_r) AND the plasticity window (w_tref), driving reward-gated
LTP / stress-driven LTD in the legacy multiplicative rule -- NO error-correcting
rule. A decoder is trained online on the substrate's dynamical representation.

Goal (Stage A): does teacher feedback make the substrate representation more
class-separable over continuous time -- enough that the teacher can be removed and
the decoder still classifies? The substrate state is NOT reset between images:
the system is inherently time-based (the slow t_ref window is its working memory).

Native levers (read from neuron.py, tunable here via sensitivity vectors):
  reward m1: t_ref += w_tref[1]*M1 (+10 default -> WIDER LTP window) ; r += w_r[1]*M1
  stress m0: t_ref += w_tref[0]*M0 (-20 default -> narrower -> LTD)  ; r += w_r[0]*M0
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import torch

from snn_classification_realtime.foveation.perception import (
    PerceptionNetwork, build_fovea_network_json,
)
from snn_classification_realtime.foveation.retina import Retina
from snn_classification_realtime.foveation.experiment_tref_stability import autoscale_gain


# --------------------------------------------------------------------------- #
#  Config
# --------------------------------------------------------------------------- #
@dataclass
class MiniBrainConfig:
    dataset_name: str = "cifar10_grayscale"   # or "cifar10" (color)
    # retina
    fovea: int = 8          # sharp foveal window (source px)
    periph: int = 24        # blurred surround window (source px)
    grid: int = 12          # common retinal grid the substrate sees
    # substrate
    substrate_type: str = "reservoir"   # "reservoir" (LSM) or "conv"
    input_period: int = 1   # feed the image every k ticks (>1 = intermittent, so
                            # units calm between pulses -> silence-driven LTD)
    # reservoir params
    n_buffer: int = 48
    n_pool: int = 160
    pool_fan_in: int = 6
    n_hub: int = 8
    hub_fan_in: int = 24
    recurrent_frac: float = 0.5
    # conv params (substrate_type == "conv")
    conv: tuple = ((4, 2, 3), (3, 2, 4))
    connectivity: float = 0.2
    auto_gain: float = 1.8
    # hidden-layer firing calibration (intrinsic homeostasis): keep every layer in
    # a graded band so LTD can engage instead of railing weights to the ceiling.
    calibrate_hidden: bool = True
    target_participation: float = 0.04   # per-layer target spike fraction
    recalibrate_every: int = 100         # fixations between threshold re-tunes (0=off)
    seed: int = 0
    # neuromodulator sensitivity (multipliers on the model defaults; tunable)
    w_tref_scale: float = 1.0    # scales [-20, 10] learning-window lever
    w_r_scale: float = 1.0       # scales [-0.2, 0.05] excitability lever
    gamma: tuple | None = (0.9, 0.9)  # neuromod decay; faster than model default
                                      # (0.99,0.995) so M builds within a fixation
                                      # (tau~10 vs ~100-200 ticks). None = model default.
    # loop timing
    dwell: int = 40              # ticks per fixation
    perceive_frac: float = 0.5   # first fraction = perceive; rest = consolidate
    # teacher
    reward_gain: float = 1.0     # m1 = reward_gain * p(true_class)
    stress_gain: float = 1.0     # m0 = stress_gain * (1 - p(true_class))
    teacher_mode: str = "graded" # "graded" (uses p_true) or "binary" (correct?)
    # decoder
    decoder_lr: float = 0.05
    # bookkeeping
    output_dir: str = "foveation_results/minibrain"


# --------------------------------------------------------------------------- #
#  Online linear decoder (multiclass logistic, SGD) -- continuously trained
# --------------------------------------------------------------------------- #
class OnlineLinearDecoder:
    def __init__(self, dim, n_classes, lr=0.05):
        self.W = np.zeros((n_classes, dim), np.float64)
        self.b = np.zeros(n_classes, np.float64)
        self.lr = lr
        self.n = n_classes

    def _logits(self, x):
        return self.W @ x + self.b

    def proba(self, x):
        z = self._logits(x); z -= z.max()
        e = np.exp(z); return e / (e.sum() + 1e-12)

    def predict(self, x):
        return int(np.argmax(self._logits(x)))

    def update(self, x, y):
        p = self.proba(x)
        g = -p; g[y] += 1.0                  # (onehot - p)
        self.W += self.lr * np.outer(g, x)
        self.b += self.lr * g


# --------------------------------------------------------------------------- #
#  Teacher (VTA / basal ganglia) -- has the true label, emits m0/m1
# --------------------------------------------------------------------------- #
class Teacher:
    def __init__(self, cfg: MiniBrainConfig):
        self.cfg = cfg

    def signal(self, proba, y_true):
        """Return (m0 stress, m1 reward). Uses the true label directly."""
        p_true = float(proba[y_true])
        if self.cfg.teacher_mode == "binary":
            correct = int(np.argmax(proba)) == y_true
            return (0.0, self.cfg.reward_gain) if correct else (self.cfg.stress_gain, 0.0)
        # graded: reward tracks prob mass on the truth, stress the mass off it
        return (self.cfg.stress_gain * (1.0 - p_true), self.cfg.reward_gain * p_true)


# --------------------------------------------------------------------------- #
#  Mini-brain
# --------------------------------------------------------------------------- #
class MiniBrain:
    def __init__(self, cfg: MiniBrainConfig, ds_cfg):
        self.cfg = cfg
        self.ds_cfg = ds_cfg
        os.makedirs(cfg.output_dir, exist_ok=True)
        ds = ds_cfg.dataset
        img0, _ = ds[0]
        self.C = img0.shape[0]
        self.H = img0.shape[1]
        self.retina = Retina(self.H, self.H, grid=cfg.grid,
                             fovea_extent=cfg.fovea, periph_extent=cfg.periph)
        self.retina.center()
        # substrate sees 2C channels at grid resolution
        if cfg.substrate_type == "reservoir":
            from snn_classification_realtime.foveation.minibrain.reservoir import (
                build_reservoir_json,
            )
            net = os.path.join(cfg.output_dir, f"reservoir_{2*self.C}x{cfg.grid}.json")
            net, self.res_info = build_reservoir_json(
                net, channels=self.C, grid=cfg.grid, n_buffer=cfg.n_buffer,
                n_pool=cfg.n_pool, pool_fan_in=cfg.pool_fan_in, n_hub=cfg.n_hub,
                hub_fan_in=cfg.hub_fan_in, recurrent_frac=cfg.recurrent_frac, seed=cfg.seed)
        else:
            net = os.path.join(cfg.output_dir, f"substrate_{2*self.C}x{cfg.grid}.json")
            layers = [{"type": "conv", "kernel_size": k, "stride": s, "filters": f,
                       "connectivity": cfg.connectivity} for (k, s, f) in cfg.conv]
            build_fovea_network_json(net, channels=2 * self.C, size=cfg.grid,
                                     layers=layers, seed=cfg.seed)
        self.sub = PerceptionNetwork(net, ds_cfg)
        self._set_sensitivity()
        autoscale_gain(self.sub, ds_cfg, cfg.auto_gain)
        self.n_neurons = self.sub.num_neurons
        self._input_layer = min(self.sub.layer_indices)
        self._hidden = [L for L in self.sub.layer_indices if L != self._input_layer]
        # readout = the deepest layer (pool for reservoir; last conv layer otherwise)
        self._readout_layer = max(self.sub.layer_indices)
        self._readout_mask = self.sub.layer_of_pos == self._readout_layer
        if cfg.calibrate_hidden and self._hidden:
            self.calibrate_layers(cfg.target_participation)
        self._fixation = 0
        self.dim = 3 * int(self._readout_mask.sum())   # [S | F_avg | O] of readout
        self.decoder = OnlineLinearDecoder(self.dim, 10, lr=cfg.decoder_lr)
        self.teacher = Teacher(cfg)
        self._feat_mu = np.zeros(self.dim)            # running feature normaliser
        self._feat_n = 0

    def _set_sensitivity(self):
        for nrn in self.sub.sim.network.neurons.values():
            nrn.params.w_tref = np.asarray(nrn.params.w_tref, float) * self.cfg.w_tref_scale
            nrn.params.w_r = np.asarray(nrn.params.w_r, float) * self.cfg.w_r_scale
            nrn.params.w_b = np.asarray(nrn.params.w_b, float) * self.cfg.w_r_scale
            if self.cfg.gamma is not None:
                nrn.params.gamma = np.asarray(self.cfg.gamma, float)

    def calibrate_layers(self, target=0.04, iters=12, n_imgs=8, ticks=25, step=0.2):
        """Intrinsic homeostasis: tune each HIDDEN layer's firing threshold r_base
        so its spike fraction sits near `target` (graded band). Learning is frozen
        during the probe. Called at init and, if recalibrate_every>0, periodically
        as weights drift -- the substrate's slow self-regulation, a system-level
        stand-in for the firing homeostasis the neuron lacks intrinsically."""
        ds = self.ds_cfg.dataset
        sigs = [self.sub.patch_to_signals(self.retina.render(ds[i][0]))
                for i in range(n_imgs)]
        lay = self.sub.layer_of_pos
        self.sub.set_learning(False)
        for _ in range(iters):
            self.sub.reset()
            acc = np.zeros(self.n_neurons)
            n = 0
            for si, sig in enumerate(sigs):
                for _t in range(ticks):
                    acc += self.sub.step(sig).O; n += 1
            acc /= max(1, n)
            for L in self._hidden:
                p = float(acc[lay == L].mean())
                if p > target * 1.25:
                    self.sub.scale_layer_threshold(L, 1.0 + step)
                elif p < target * 0.75:
                    self.sub.scale_layer_threshold(L, max(0.3, 1.0 - step))
        self.sub.reset()
        self.sub.set_learning(True)

    def _rep(self, states):
        """Dynamical representation = mean readout-layer [S|F_avg|O] over a window."""
        m = self._readout_mask
        S = np.mean([s.S[m] for s in states], axis=0)
        F = np.mean([s.F_avg[m] for s in states], axis=0)
        O = np.mean([s.O[m] for s in states], axis=0)
        x = np.concatenate([S, F, O]).astype(np.float64)
        # cheap running standardisation so the linear decoder is well-conditioned
        self._feat_n += 1
        self._feat_mu += (x - self._feat_mu) / self._feat_n
        return x - self._feat_mu

    def present(self, image, label, learn=True, teach=True):
        """One fixation: perceive -> teacher reward -> consolidate. No reset."""
        sig = self.sub.patch_to_signals(self.retina.render(image))
        if not learn:
            self.sub.set_learning(False)
        D = self.cfg.dwell
        split = max(1, int(D * self.cfg.perceive_frac))
        per = max(1, self.cfg.input_period)
        states = []
        for t in range(D):
            if teach and t == split:
                x_mid = self._rep(states)
                m0, m1 = self.teacher.signal(self.decoder.proba(x_mid), label)
                self._last_nm = (m0, m1)
            if teach and t >= split:
                self.sub.broadcast_neuromod(*self._last_nm)
            drive = sig if (t % per == 0) else []   # intermittent: pulse then silence
            states.append(self.sub.step(drive))
        if not learn:
            self.sub.set_learning(True)
        x = self._rep(states)
        pred = self.decoder.predict(x)
        if teach:
            self.decoder.update(x, label)
        return x, pred


def knn_accuracy(X, y, k=5, split=0.5):
    """Fresh-probe separability of the substrate representation (independent of the
    online decoder): kNN on a train/test split of buffered (rep, label)."""
    X = np.asarray(X); y = np.asarray(y)
    n = len(y); ntr = int(n * split)
    idx = np.random.RandomState(0).permutation(n)
    tr, te = idx[:ntr], idx[ntr:]
    if len(te) == 0 or len(tr) == 0:
        return float("nan")
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    # normalise then cosine/euclidean kNN
    correct = 0
    for xq, yq in zip(Xte, yte):
        d = ((Xtr - xq) ** 2).sum(1)
        nn = ytr[np.argsort(d)[:k]]
        correct += int(np.bincount(nn).argmax() == yq)
    return correct / len(te)
