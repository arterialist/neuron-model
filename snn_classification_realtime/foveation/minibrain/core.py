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
from snn_classification_realtime.foveation.retinal_conv import RetinalConv
from snn_classification_realtime.foveation.experiment_tref_stability import autoscale_gain
from snn_classification_realtime.activity_dataset_builder.drive_calibration import (
    _synapse_weight, _saturation_ratio_limit,
)


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
    # retinal convolutions (ganglion/LGN front-end: center-surround + oriented edges
    # + ON/OFF rectification). The rectification nonlinearity lifts linear
    # separability on a single glimpse (gray 0.22->0.34, color 0.29->0.38 measured).
    retinal_conv: bool = True
    retinal_pool: int = 2
    # retinal-conv filter bank: "default" (2 DoG + 4 edge) or "rich" (4 DoG scales +
    # 8 oriented Gabor-like edges) -- the richer, locality-preserving front-end tests
    # whether more oriented/multiscale features lift single-glimpse separability
    # toward the working conv-PAULA pipeline's regime. Fixed, not learned.
    conv_bank: str = "default"   # "default" | "rich"
    # substrate
    substrate_type: str = "reservoir"   # "reservoir" (LSM) or "conv"
    # reservoir wiring: "random" (dense scrambling recurrent pool -- the baseline that
    # loses conv locality) or "retinotopic" (pool units have LOCAL receptive fields in
    # retinal space, preserving the locality that makes conv-PAULA beat a linear probe
    # on MNIST). The MNIST-vs-reservoir reconciliation: locality is the missing prior.
    wiring: str = "random"       # "random" | "retinotopic"
    input_period: int = 1   # feed the image every k ticks (>1 = intermittent, so
                            # units calm between pulses -> silence-driven LTD)
    # reservoir params
    n_in: int = 96          # input neurons; too few crushes the conv features (864-d)
                            # through an 18:1 bottleneck -> lifts conv->input loss
    n_buffer: int = 48
    n_pool: int = 160
    pool_fan_in: int = 6
    readout: str = "input"  # "input" (best single-glimpse), "pool" (deepest), or "all"
    n_hub: int = 8
    hub_fan_in: int = 24
    recurrent_frac: float = 0.5
    output_strength: float = 1.0   # axon-hillock output p; higher = stronger spike
                                   # transmission (A: self-sustaining sim; B: burst)
    tonic_drive: float = 0.15      # constant background current to every reservoir
                                   # unit (C. elegans-style Ca leak): keeps the
                                   # liquid simmering so input pulses ignite it
    reservoir_lambda: float = 10.0 # membrane time const for reservoir units (faster
                                   # than default 20 so transient spikes integrate)
    freeze_retrograde: bool = True # eta_retro=0: keep u_o.info (propagation gain)
                                   # stable; retrograde otherwise self-damps it
    freeze_reservoir: bool = True  # eta_post=0 in the pool: fixed random reservoir
                                   # (classic RC). Without it the recurrent weights
                                   # decay-spiral and the liquid dies.
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
    teacher_mode: str = "graded" # "rpe" (dopamine RPE vs per-class baseline; self-
                                 # balancing), "graded" (p_true; all-stress at chance),
                                 # or "binary" (correct?)
    baseline_tau: float = 0.02   # EMA rate of the rpe teacher's per-class baseline
    # decoder
    decoder_lr: float = 0.05
    # readout classifier: "linear" (single-layer online logistic -- the baseline) or
    # "mlp" (a trained nonlinear readout standing in for the pipeline's SNN classifier,
    # isolating whether the linear decoder -- not PAULA -- was the ceiling).
    readout_head: str = "linear"     # "linear" | "mlp"
    mlp_hidden: int = 128
    # external substrate plasticity (feature-flagged; neuron.py UNTOUCHED, native
    # eta_post stays 0). "none" = frozen reservoir (baseline). The others apply a
    # weight update OUTSIDE the model to the reservoir's u_i.info each fixation:
    #   "oja"    -- normalised Hebbian (bounded, no runaway) on pre/post spikes
    #   "bcm"    -- BCM sliding-threshold (LTP above per-unit threshold, LTD below)
    #   "rmhebb" -- reward-modulated Hebbian gated by the teacher's (m1-m0) signal
    # Tests whether reward-gated plasticity ACCUMULATES structure when it is a proper
    # rule in a locality-preserving topology (vs the legacy multiplicative saturation).
    ext_plasticity: str = "none"     # "none" | "oja" | "bcm" | "rmhebb"
    ext_plast_eta: float = 0.002
    assoc_layer: bool = False        # add an external reward-gated association layer
                                     # (numpy three-factor) between readout and decoder
    assoc_dim: int = 256
    # NATIVE substrate plasticity rule (in neuron.py, feature-flagged there; default
    # value here reproduces today's behavior). Set "reward_hebb" + freeze_reservoir
    # False to let the reservoir learn with the non-saturating native rule; nm_kappa>0
    # makes it three-factor reward-gated by the teacher.
    plasticity_mode: str = "legacy_multiplicative"
    nm_kappa: float = 0.0            # three-factor gate strength (nm_plasticity_kappa)
    rh_decay: float = 0.1            # reward_hebb bounded-fixed-point decay coeff
    eta_post_res: float | None = None  # override reservoir learning rate when unfrozen
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
    def __init__(self, cfg: MiniBrainConfig, n_classes=10):
        self.cfg = cfg
        # per-class running baseline of p(true) -- the "expected reward" a dopaminergic
        # VTA compares against. Starts at chance so early modulation is ~zero (no
        # all-stress collapse) and only diverges as some classes become well-encoded.
        self.baseline = np.full(n_classes, 1.0 / n_classes, dtype=np.float64)
        self.bl_tau = float(getattr(cfg, "baseline_tau", 0.02))

    def signal(self, proba, y_true):
        """Return (m0 stress, m1 reward). Uses the true label directly."""
        p_true = float(proba[y_true])
        if self.cfg.teacher_mode == "binary":
            correct = int(np.argmax(proba)) == y_true
            return (0.0, self.cfg.reward_gain) if correct else (self.cfg.stress_gain, 0.0)
        if self.cfg.teacher_mode == "rpe":
            # reward-prediction-error: reward when the true class beats its running
            # baseline, stress when below. Self-balancing (mean ~0) so it can't drown
            # the substrate in constant LTD the way "graded" does at chance.
            b = self.baseline[y_true]
            delta = p_true - b
            self.baseline[y_true] += self.bl_tau * delta
            m1 = self.cfg.reward_gain * max(0.0, delta)
            m0 = self.cfg.stress_gain * max(0.0, -delta)
            return (m0, m1)
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
        # retinal-conv front-end: fit its contrast scale on a sample of glimpses,
        # then the substrate's input layer is sized to the conv feature dim.
        self.retconv = None
        input_dim = None
        if cfg.retinal_conv:
            self.retconv = RetinalConv(pool=cfg.retinal_pool,
                                       bank=getattr(cfg, "conv_bank", "default"))
            sample = [self.retina.render(ds[i][0]) for i in range(24)]
            self.retconv.fit(sample)
            input_dim = self.retconv.feature_dim(2 * self.C, cfg.grid)
            # conv features feed the DENSE input mapper (channel handling already done)
            ds_cfg.is_colored_cifar10 = False
        # substrate sees 2C channels at grid resolution
        if cfg.substrate_type == "reservoir":
            from snn_classification_realtime.foveation.minibrain.reservoir import (
                build_reservoir_json,
            )
            net = os.path.join(cfg.output_dir, f"reservoir_{2*self.C}x{cfg.grid}.json")
            net, self.res_info = build_reservoir_json(
                net, channels=self.C, grid=cfg.grid, n_in=cfg.n_in,
                n_buffer=cfg.n_buffer, n_pool=cfg.n_pool, pool_fan_in=cfg.pool_fan_in,
                n_hub=cfg.n_hub, hub_fan_in=cfg.hub_fan_in,
                recurrent_frac=cfg.recurrent_frac, output_strength=cfg.output_strength,
                input_dim=input_dim, wiring=getattr(cfg, "wiring", "random"),
                seed=cfg.seed)
        else:
            net = os.path.join(cfg.output_dir, f"substrate_{2*self.C}x{cfg.grid}.json")
            layers = [{"type": "conv", "kernel_size": k, "stride": s, "filters": f,
                       "connectivity": cfg.connectivity} for (k, s, f) in cfg.conv]
            build_fovea_network_json(net, channels=2 * self.C, size=cfg.grid,
                                     layers=layers, seed=cfg.seed)
        self.sub = PerceptionNetwork(net, ds_cfg)
        self._set_sensitivity()
        self._calibrate_gain(cfg.auto_gain)   # on the ACTUAL encoded input, not raw imgs
        self.n_neurons = self.sub.num_neurons
        self._input_layer = min(self.sub.layer_indices)
        self._hidden = [L for L in self.sub.layer_indices if L != self._input_layer]
        # readout choice. Measured (frozen, conv on): the shallow layers preserve most
        # of the conv separability the deep recurrent pool loses -- reservoir mixing is
        # lossy for a single glimpse (its real value is temporal integration over gaze).
        #   "input" = best single-glimpse readout ; "pool" = deepest ; "all" = concat.
        self._readout_layer = max(self.sub.layer_indices)
        ro = getattr(cfg, "readout", "input")
        if ro == "all":
            self._readout_mask = np.ones(self.sub.num_neurons, dtype=bool)
        elif ro == "input":
            self._readout_mask = self.sub.layer_of_pos == self._input_layer
        else:
            self._readout_mask = self.sub.layer_of_pos == self._readout_layer
        # tonic background drive to reservoir units (injected every tick)
        self._tonic = []
        if cfg.substrate_type == "reservoir" and cfg.tonic_drive > 0:
            self._tonic = [(nid, sid, float(cfg.tonic_drive))
                           for (nid, sid) in self.res_info.get("tonic_targets", [])]
        if cfg.calibrate_hidden and self._hidden:
            self.calibrate_layers(cfg.target_participation)
        self._fixation = 0
        self.dim = 3 * int(self._readout_mask.sum())   # [S | F_avg | O] of readout
        # optional external reward-gated association layer (Proposal 3, readout-side)
        self.assoc = None
        dec_dim = self.dim
        if getattr(cfg, "assoc_layer", False):
            from snn_classification_realtime.foveation.minibrain.heads import AssociationLayer
            self.assoc = AssociationLayer(self.dim, out_dim=cfg.assoc_dim, seed=cfg.seed)
            dec_dim = cfg.assoc_dim
        self.decoder = OnlineLinearDecoder(dec_dim, 10, lr=cfg.decoder_lr)
        # optional external synaptic plasticity on the reservoir (Proposal 3, substrate-
        # side; native eta_post stays 0). Built after the network exists.
        self.ext_plast = None
        if getattr(cfg, "ext_plasticity", "none") != "none":
            from snn_classification_realtime.foveation.minibrain.heads import ExternalReservoirPlasticity
            self.ext_plast = ExternalReservoirPlasticity(
                self, rule=cfg.ext_plasticity, eta=cfg.ext_plast_eta, seed=cfg.seed)
        self.teacher = Teacher(cfg, n_classes=self.decoder.n)
        self._feat_mu = np.zeros(self.dim)            # running feature normaliser
        self._feat_n = 0
        self._gaze_rng = np.random.RandomState(cfg.seed + 999)   # random saccade starts

    def _encode(self, image, fovea_only=False):
        """Retina glimpse -> (optional) retinal-conv features -> substrate input tensor.
        fovea_only zeroes the periphery channels (to background) so the substrate is
        driven by the sharp fovea alone -- used in the memorize phase so peripheral
        clutter doesn't drown the foveated object in the classifier's readout."""
        r = self.retina.render(image)
        if fovea_only:
            r = r.clone(); C = r.shape[0] // 2
            r[C:] = -1.0                                  # periphery -> background
        return self.retconv.encode(r) if self.retconv is not None else r

    def _calibrate_gain(self, target_ratio, n=16):
        """Set ds_cfg.signal_gain so mean input-layer drive/threshold ~= target_ratio,
        measured on the ACTUAL encoded input the substrate will see (retina + conv),
        not raw dataset images (which estimate_input_drive uses and which mismatch)."""
        ds = self.ds_cfg.dataset
        neurons = self.sub.sim.network.neurons
        ids = self.sub.input_layer_ids
        id_to_pos = {nid: i for i, nid in enumerate(ids)}
        drive = np.zeros((n, len(ids)))
        for row in range(n):
            sigs = self.sub.patch_to_signals(self._encode(ds[row][0]))
            for nid, sid, strength in sigs:
                pos = id_to_pos.get(nid)
                if pos is not None:
                    drive[row, pos] += strength * _synapse_weight(neurons[nid], sid)
        thr = np.array([float(getattr(neurons[nid], "r",
                        getattr(neurons[nid].params, "r_base", 1.0))) for nid in ids])
        ratios = drive.mean(0) / np.maximum(thr, 1e-9)
        ratio_mean = float(ratios.mean())
        cur = float(getattr(self.ds_cfg, "signal_gain", 1.0))
        if ratio_mean > 1e-9:
            self.ds_cfg.signal_gain = cur * target_ratio / ratio_mean
        sat = _saturation_ratio_limit(neurons[ids[0]])
        print(f"  [gain] encoded input ratio {ratio_mean:.2f} (sat>{sat:.2f}) "
              f"-> gain {self.ds_cfg.signal_gain:.4f} (target {target_ratio})")

    def _set_sensitivity(self):
        native_rule = getattr(self.cfg, "plasticity_mode", "legacy_multiplicative")
        for nrn in self.sub.sim.network.neurons.values():
            nrn.params.w_tref = np.asarray(nrn.params.w_tref, float) * self.cfg.w_tref_scale
            nrn.params.w_r = np.asarray(nrn.params.w_r, float) * self.cfg.w_r_scale
            nrn.params.w_b = np.asarray(nrn.params.w_b, float) * self.cfg.w_r_scale
            if self.cfg.gamma is not None:
                nrn.params.gamma = np.asarray(self.cfg.gamma, float)
            # NATIVE gated plasticity rule (default leaves legacy behavior intact)
            if native_rule != "legacy_multiplicative":
                nrn.params.plasticity_mode = native_rule
                nrn.params.nm_plasticity_kappa = float(self.cfg.nm_kappa)
                nrn.params.rh_decay = float(self.cfg.rh_decay)
            if self.cfg.substrate_type == "reservoir":
                nrn.params.lambda_param = float(self.cfg.reservoir_lambda)
                if self.cfg.freeze_retrograde:
                    nrn.params.eta_retro = 0.0
                layer = int(nrn.metadata.get("layer", 0))
                if self.cfg.freeze_reservoir and layer >= 1:
                    nrn.params.eta_post = 0.0   # fixed reservoir; readout learns
                elif not self.cfg.freeze_reservoir and self.cfg.eta_post_res is not None \
                        and layer >= 1:
                    nrn.params.eta_post = float(self.cfg.eta_post_res)  # learn rate

    def calibrate_layers(self, target=0.04, iters=12, n_imgs=8, ticks=25, step=0.2):
        """Intrinsic homeostasis: tune EVERY layer's firing threshold r_base/b_base
        so its spike fraction sits near `target` (graded band). Learning is frozen
        during the probe. Called at init and, if recalibrate_every>0, periodically
        as weights drift -- the substrate's slow self-regulation, a system-level
        stand-in for the firing homeostasis the neuron lacks intrinsically.

        The INPUT layer is included and matters most: it has NO tonic synapse, so it
        fires only when the image is present. Calibrating it down (driven S ~0.5 vs
        default threshold ~1.1 -> never fires otherwise) is what lets the image
        actually ignite the reservoir instead of tonic doing all the work."""
        ds = self.ds_cfg.dataset
        sigs = [self.sub.patch_to_signals(self._encode(ds[i][0]))
                for i in range(n_imgs)]
        lay = self.sub.layer_of_pos
        # Bootstrap under CONTINUOUS drive: intermittent input is too sparse for a
        # cold pool to ever fire, so thresholds can't be set. Once alive, the run's
        # intermittent cadence just makes it burst-then-calm.
        per = 1
        neurons = self.sub.sim.network.neurons

        def scale_thr(L, f):
            for nrn in neurons.values():
                if int(nrn.metadata.get("layer", 0)) == L:
                    nrn.params.r_base *= f    # both thresholds gate firing: r pre-
                    nrn.params.b_base *= f    # cooldown, b post-cooldown
        calib_layers = list(self.sub.layer_indices)   # input + hidden: input must
                                                       # fire for the image to enter
        self.sub.set_learning(False)
        for _ in range(iters):
            self.sub.reset()
            acc = np.zeros(self.n_neurons); n = 0
            for sig in sigs:
                for _t in range(ticks):
                    d = (sig if _t % per == 0 else []) + self._tonic
                    acc += self.sub.step(d).O; n += 1
            acc /= max(1, n)
            for L in calib_layers:
                p = float(acc[lay == L].mean())
                if p > target * 1.25:
                    scale_thr(L, 1.0 + step)
                elif p < target * 0.75:
                    scale_thr(L, 1.0 - step)
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

    def present(self, image, label, learn=True, teach=True, train_decoder=None,
                return_states=False):
        """One fixation: perceive -> teacher reward -> consolidate. No reset.

        Three INDEPENDENT levers (decoupled so control arms are clean):
          learn         -- substrate synaptic plasticity on/off (set_learning)
          teach         -- teacher volume-transmits m0/m1 to the substrate
          train_decoder -- online decoder gets a supervised update (defaults to teach
                           for back-compat; pass True explicitly for a no-teacher arm
                           whose decoder still trains).
          return_states -- also return the per-tick PopulationState list (for trajectory
                           recording; additive, default off keeps the (x, pred) contract).
        """
        if train_decoder is None:
            train_decoder = teach
        sig = self.sub.patch_to_signals(self._encode(image))
        if not learn:
            self.sub.set_learning(False)
        D = self.cfg.dwell
        split = max(1, int(D * self.cfg.perceive_frac))
        per = max(1, self.cfg.input_period)
        states = []
        self._last_nm = (0.0, 0.0)
        for t in range(D):
            if teach and t == split:
                m0, m1 = self.teacher.signal(self.decoder.proba(self._feature(states)), label)
                self._last_nm = (m0, m1)
            if teach and t >= split:
                self.sub.broadcast_neuromod(*self._last_nm)
            drive = sig if (t % per == 0) else []   # intermittent: pulse then silence
            states.append(self.sub.step(drive + self._tonic))  # tonic re-injected each tick
        if not learn:
            self.sub.set_learning(True)
        # external substrate plasticity (numpy, gated) -- reservoir weights only
        if self.ext_plast is not None and learn:
            self.ext_plast.apply(states, self._last_nm)
        x = self._feature(states)               # readout (optionally via assoc layer)
        pred = self.decoder.predict(x)
        if self.assoc is not None and learn:    # reinforce assoc gain by (m1 - m0)
            self.assoc.reinforce(self._last_nm[1] - self._last_nm[0])
        if train_decoder:
            self.decoder.update(x, label)
        if return_states:
            return x, pred, states
        return x, pred

    def _feature(self, states):
        """Decoder-input feature = readout rep, optionally through the association
        layer. Kept separate from _rep so frozen-probe separability still reads the
        raw substrate representation."""
        x = self._rep(states)
        return self.assoc.project(x) if self.assoc is not None else x


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
