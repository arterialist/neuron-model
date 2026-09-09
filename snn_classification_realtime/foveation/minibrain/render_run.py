"""Record a mini-brain run and render it as a multi-panel video, to build visual
intuition for the substrate's dynamics.

What it shows, tick-synchronised:
  - the CIFAR image with the fovea (sharp) + periphery (gist) windows drawn where
    the eye is looking, plus the gaze trail (the fovea MOVES -- see GazeController);
  - the retina view: exactly what the eye currently feeds the substrate;
  - a spike RASTER (neurons x recent time), grouped input / buffer / pool;
  - a spatial spike/MEMBRANE map of the pool (the "liquid state" laid out in 2D);
  - dynamics traces: per-layer participation + teacher neuromod (m0 stress / m1 reward);
  - the online decoder's class posterior (true class highlighted) + running reward.

The gaze is NOT a hardcoded scan path: it is klinotaxis (C. elegans-style) -- the eye
drifts and steers by the temporal gradient of a substrate salience signal (pool
activity), so it climbs toward image regions that drive the reservoir. This is an
exploratory bottom-up controller, a stepping stone to a fully learned gaze.

Measurement / visualisation only: neuron.py is untouched.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import gridspec, animation


# --------------------------------------------------------------------------- #
#  Gaze: klinotaxis drift steered by a substrate salience signal (non-hardcoded)
# --------------------------------------------------------------------------- #
class GazeController:
    def __init__(self, retina, mode="klinotaxis", step=2.2, turn_noise=0.35, seed=0):
        self.retina = retina
        self.mode = mode
        self.step = step
        self.turn_noise = turn_noise
        self.rng = np.random.RandomState(seed)
        self.heading = self.rng.uniform(0, 2 * np.pi)
        self.prev = None

    def reset(self, recenter=True):
        if recenter:
            self.retina.center()
        self.heading = self.rng.uniform(0, 2 * np.pi)
        self.prev = None

    def update(self, salience: float):
        """One saccadic micro-step. Klinotaxis: if salience fell vs last tick, turn
        away (bias); otherwise keep heading. Always add exploratory wiggle."""
        if self.mode == "center":
            return
        if self.prev is not None and salience < self.prev:
            self.heading += np.pi + self.rng.uniform(-0.8, 0.8)   # reverse-ish
        self.heading += self.rng.uniform(-self.turn_noise, self.turn_noise)
        dy = self.step * np.sin(self.heading)
        dx = self.step * np.cos(self.heading)
        self.retina.move(dy, dx)
        self.prev = salience


# --------------------------------------------------------------------------- #
#  Recording
# --------------------------------------------------------------------------- #
@dataclass
class Recording:
    ticks: list = field(default_factory=list)          # global tick
    img_idx: list = field(default_factory=list)         # which image this frame shows
    images: list = field(default_factory=list)          # HxW (grayscale) per frame
    fovea_box: list = field(default_factory=list)       # (y0,x0,extent)
    periph_box: list = field(default_factory=list)
    gaze: list = field(default_factory=list)            # (cy,cx)
    retina_fov: list = field(default_factory=list)      # grid x grid
    retina_per: list = field(default_factory=list)
    O: list = field(default_factory=list)               # spikes, per neuron
    S: list = field(default_factory=list)               # membrane, per neuron
    m0: list = field(default_factory=list)
    m1: list = field(default_factory=list)
    proba: list = field(default_factory=list)           # decoder posterior
    label: list = field(default_factory=list)
    pred: list = field(default_factory=list)
    reward_run: list = field(default_factory=list)
    layer_of_pos: np.ndarray = None
    readout_mask: np.ndarray = None
    n_classes: int = 10
    class_names: list = None


def record_run(brain, images, labels, ticks_per_image=40, teach=True, learn=True,
               gaze_mode="klinotaxis", move_every=1, seed=0, perceive_frac=0.5):
    """Run the continuous loop (NO reset between images) recording every tick.

    Mirrors MiniBrain.present() teacher timing but re-renders the retina each tick as
    the eye moves, so the video shows real gaze motion + the substrate responding.
    """
    rec = Recording(layer_of_pos=brain.sub.layer_of_pos.copy(),
                    readout_mask=brain._readout_mask.copy(),
                    n_classes=brain.decoder.n)
    gaze = GazeController(brain.retina, mode=gaze_mode, seed=seed)
    m = brain._readout_mask
    split = max(1, int(ticks_per_image * perceive_frac))
    run_correct, run_n, running_reward = 0, 0, 0.0
    gtick = 0
    if not learn:
        brain.sub.set_learning(False)
    for k, (img, y) in enumerate(zip(images, labels)):
        y = int(y)
        gaze.reset(recenter=True)
        gray = img[0].numpy() if img.ndim == 3 else img.numpy()
        states, last_nm = [], (0.0, 0.0)
        for t in range(ticks_per_image):
            sig = brain.sub.patch_to_signals(brain._encode(img))
            if teach and t == split:
                x_mid = brain._rep(states)
                last_nm = brain.teacher.signal(brain.decoder.proba(x_mid), y)
            if teach and t >= split:
                brain.sub.broadcast_neuromod(*last_nm)
            st = brain.sub.step(sig + brain._tonic)
            states.append(st)
            # klinotaxis salience = current pool participation
            sal = float(st.O[m].mean())
            if (t % move_every) == 0:
                gaze.update(sal)
            # record
            fy0, fx0, fe = brain.retina.fovea_box()
            py0, px0, pe = brain.retina.periph_box()
            retv = brain.retina.render(img)
            half = retv.shape[0] // 2
            proba = brain.decoder.proba(brain._rep(states))
            rec.ticks.append(gtick); rec.img_idx.append(k)
            rec.images.append(gray)
            rec.fovea_box.append((fy0, fx0, fe)); rec.periph_box.append((py0, px0, pe))
            rec.gaze.append((brain.retina.cy, brain.retina.cx))
            rec.retina_fov.append(retv[:half].mean(0).numpy())
            rec.retina_per.append(retv[half:].mean(0).numpy())
            rec.O.append(st.O.copy()); rec.S.append(st.S.copy())
            rec.m0.append(last_nm[0]); rec.m1.append(last_nm[1])
            rec.proba.append(proba); rec.label.append(y)
            rec.pred.append(int(np.argmax(proba)))
            rec.reward_run.append(running_reward)
            gtick += 1
        x = brain._rep(states)
        pred = brain.decoder.predict(x)
        run_correct += int(pred == y); run_n += 1
        running_reward = 0.9 * running_reward + 0.1 * float(pred == y)
        if teach and learn:
            brain.decoder.update(x, y)
    if not learn:
        brain.sub.set_learning(True)
    return rec


# --------------------------------------------------------------------------- #
#  Rendering
# --------------------------------------------------------------------------- #
CIFAR10 = ["plane", "auto", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def _grid_shape(n):
    w = int(np.ceil(np.sqrt(n)))
    return int(np.ceil(n / w)), w


def render_video(rec: Recording, out_path, fps=12, raster_window=60, dpi=110,
                 class_names=None):
    """Write the recording to an mp4 (falls back to gif if ffmpeg missing)."""
    names = class_names or CIFAR10
    lay = rec.layer_of_pos
    layer_ids = sorted(set(int(x) for x in lay))
    lname = {layer_ids[0]: "input"}
    if len(layer_ids) >= 2: lname[layer_ids[1]] = "buffer"
    if len(layer_ids) >= 3: lname[layer_ids[-1]] = "pool"
    order = np.argsort(lay, kind="stable")          # neurons sorted by layer for raster
    lay_sorted = lay[order]
    bounds = [np.searchsorted(lay_sorted, L) for L in layer_ids] + [len(lay_sorted)]
    pool_mask = rec.readout_mask
    ph, pw = _grid_shape(int(pool_mask.sum()))
    O = np.array(rec.O); S = np.array(rec.S)
    proba = np.array(rec.proba)
    m0 = np.array(rec.m0); m1 = np.array(rec.m1)
    Nf = len(rec.ticks)
    part = {L: O[:, lay == L].mean(1) for L in layer_ids}

    fig = plt.figure(figsize=(13, 8.6), constrained_layout=True)
    gs = gridspec.GridSpec(4, 4, figure=fig, height_ratios=[1.25, 1.0, 0.9, 0.9])
    ax_img = fig.add_subplot(gs[0:2, 0:2])
    ax_fov = fig.add_subplot(gs[0, 2]); ax_per = fig.add_subplot(gs[0, 3])
    ax_map = fig.add_subplot(gs[1, 2:4])
    ax_ras = fig.add_subplot(gs[2, 0:4])
    ax_dyn = fig.add_subplot(gs[3, 0:3]); ax_dec = fig.add_subplot(gs[3, 3])

    for ax in (ax_img, ax_fov, ax_per, ax_map):
        ax.set_xticks([]); ax.set_yticks([])

    def draw(fr):
        for ax in (ax_img, ax_fov, ax_per, ax_map, ax_ras, ax_dyn, ax_dec):
            ax.clear()
        # --- image + gaze ---
        ax_img.imshow(rec.images[fr], cmap="gray", interpolation="nearest")
        fy, fx, fe = rec.fovea_box[fr]; py, px, pe = rec.periph_box[fr]
        ax_img.add_patch(Rectangle((px, py), pe, pe, fill=False, ec="#39c5ff", lw=1.6))
        ax_img.add_patch(Rectangle((fx, fy), fe, fe, fill=False, ec="#ffd21e", lw=2.2))
        k = rec.img_idx[fr]
        trail = [g for g, ki in zip(rec.gaze[:fr + 1], rec.img_idx[:fr + 1]) if ki == k]
        if len(trail) > 1:
            ty, tx = zip(*trail)
            ax_img.plot(tx, ty, "-", color="#ffd21e", lw=1.0, alpha=0.7)
        cy, cx = rec.gaze[fr]; ax_img.plot([cx], [cy], "o", color="#ffd21e", ms=5)
        y = rec.label[fr]; pr = rec.pred[fr]
        ok = "OK" if pr == y else "x"
        ax_img.set_xticks([]); ax_img.set_yticks([])
        ax_img.set_title(f"gaze  |  true={names[y]}  pred={names[pr]} {ok}  "
                         f"|  img {k+1}  tick {rec.ticks[fr]}", fontsize=10)
        # --- retina views ---
        ax_fov.imshow(rec.retina_fov[fr], cmap="gray", interpolation="nearest")
        ax_fov.set_title("fovea (sharp)", fontsize=9)
        ax_per.imshow(rec.retina_per[fr], cmap="gray", interpolation="nearest")
        ax_per.set_title("periphery (gist)", fontsize=9)
        ax_fov.set_xticks([]); ax_fov.set_yticks([]); ax_per.set_xticks([]); ax_per.set_yticks([])
        # --- pool spatial map (membrane, spikes highlighted) ---
        ps = S[fr, pool_mask]; po = O[fr, pool_mask]
        grid = np.full(ph * pw, np.nan); grid[:ps.size] = ps
        ax_map.imshow(grid.reshape(ph, pw), cmap="viridis", interpolation="nearest",
                      vmin=float(np.nanpercentile(S[:, pool_mask], 5)),
                      vmax=float(np.nanpercentile(S[:, pool_mask], 98)) + 1e-6)
        spk = np.where(po > 0)[0]
        if spk.size:
            ax_map.plot(spk % pw, spk // pw, "s", mfc="none", mec="#ff2e63", ms=6, mew=1.4)
        ax_map.set_title(f"pool liquid state (S + spikes)  |  active {int(po.sum())}/{po.size}",
                         fontsize=9)
        ax_map.set_xticks([]); ax_map.set_yticks([])
        # --- raster ---
        lo = max(0, fr - raster_window + 1)
        win = O[lo:fr + 1][:, order]                 # time x neuron(sorted)
        tt, nn = np.where(win > 0)
        ax_ras.scatter(tt + lo, nn, s=2.2, c="k", marker="|")
        for b in bounds[1:-1]:
            ax_ras.axhline(b, color="#bbbbbb", lw=0.7)
        for L, b0, b1 in zip(layer_ids, bounds[:-1], bounds[1:]):
            ax_ras.text(lo + 0.2, (b0 + b1) / 2, lname.get(L, str(L)), fontsize=8,
                        va="center", color="#0057b8", rotation=90)
        ax_ras.set_xlim(lo, max(lo + raster_window, fr + 1))
        ax_ras.set_ylim(0, len(order)); ax_ras.set_ylabel("neuron", fontsize=8)
        ax_ras.set_title("spike raster (recent)", fontsize=9)
        ax_ras.axvline(fr, color="#ff2e63", lw=0.8, alpha=0.6)
        # --- dynamics ---
        xs = np.arange(fr + 1)
        cols = {layer_ids[0]: "#888", }
        palette = ["#888888", "#00a0a0", "#0057b8"]
        for L, c in zip(layer_ids, palette):
            ax_dyn.plot(xs, part[L][:fr + 1], color=c, lw=1.2, label=lname.get(L, str(L)))
        ax_dyn.plot(xs, m1[:fr + 1], color="#2ca02c", lw=1.0, ls="--", label="m1 reward")
        ax_dyn.plot(xs, m0[:fr + 1], color="#d62728", lw=1.0, ls=":", label="m0 stress")
        ax_dyn.set_xlim(0, Nf); ax_dyn.set_ylim(0, None)
        ax_dyn.legend(fontsize=7, ncol=3, loc="upper left")
        ax_dyn.set_title("participation per layer + teacher neuromod", fontsize=9)
        ax_dyn.set_xlabel("tick", fontsize=8)
        # image boundaries
        for bnd in np.where(np.diff(rec.img_idx[:fr + 1]) != 0)[0]:
            ax_dyn.axvline(bnd, color="#dddddd", lw=0.6)
        # --- decoder posterior ---
        p = proba[fr]
        bar_c = ["#0057b8"] * len(p); bar_c[y] = "#2ca02c"
        if pr != y: bar_c[pr] = "#d62728"
        ax_dec.bar(range(len(p)), p, color=bar_c)
        ax_dec.set_xticks(range(len(p)))
        ax_dec.set_xticklabels(names, rotation=90, fontsize=6)
        ax_dec.set_ylim(0, 1); ax_dec.set_title("decoder posterior", fontsize=9)
        return []

    anim = animation.FuncAnimation(fig, draw, frames=Nf, interval=1000 / fps, blit=False)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    try:
        matplotlib.rcParams["animation.ffmpeg_path"] = _find_ffmpeg()
        anim.save(out_path, writer=animation.FFMpegWriter(fps=fps, bitrate=2400))
    except Exception as e:
        print(f"[render] ffmpeg failed ({e}); writing gif")
        out_path = os.path.splitext(out_path)[0] + ".gif"
        anim.save(out_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return out_path


def _find_ffmpeg():
    import shutil
    return shutil.which("ffmpeg") or "ffmpeg"
