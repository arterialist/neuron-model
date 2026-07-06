"""Interactive fovea + neuron-dynamics viewer.

Watch a PAULA perception substrate look at an image through a multi-resolution
retina (sharp fovea + blurred periphery), with LOCAL PLASTICITY ON, while you
drag the gaze around. Every neuron is shown individually (state grid + spike
raster) alongside population dynamics.

Two modes:
  play    (default) — interactive: drag the gaze in the image panel, the net
          runs continuously. Keys: [space] pause · [l] toggle learning ·
          [r] reset net · [n] next image · [c] recenter · [+/-] gain.
  record  — headless (Agg): a scripted gaze path is swept and written to mp4,
          for verification / sharing (no display needed).

Run:
  .venv/bin/python -m snn_classification_realtime.foveation.viewer
  .venv/bin/python -m snn_classification_realtime.foveation.viewer --record out.mp4
"""

from __future__ import annotations

import argparse
import os
from collections import deque

import numpy as np
import torch

from snn_classification_realtime.activity_dataset_builder.vision_datasets import (
    load_dataset_by_name,
)
from snn_classification_realtime.foveation.retina import Retina
from snn_classification_realtime.foveation.perception import (
    PerceptionNetwork,
    build_fovea_network_json,
)


def _grid_shape(n: int) -> tuple[int, int]:
    r = int(np.ceil(np.sqrt(n)))
    c = int(np.ceil(n / r))
    return r, c


def _to_grid(vec: np.ndarray, r: int, c: int) -> np.ndarray:
    out = np.full(r * c, np.nan, dtype=np.float32)
    out[: len(vec)] = vec
    return out.reshape(r, c)


class ViewerState:
    """Holds the sim + retina and advances one tick, exposing artist data."""

    def __init__(self, args):
        self.args = args
        self.ds_cfg = load_dataset_by_name(args.dataset_name, train=True)
        self.ds_cfg.signal_gain = args.gain
        self.ds = self.ds_cfg.dataset
        img0, _ = self.ds[0]
        self.base_c, self.H, self.W = img0.shape
        self.retina = Retina(self.H, self.W, grid=args.grid,
                             fovea_extent=args.fovea_extent,
                             periph_extent=args.periph_extent,
                             periph_bottleneck=args.periph_bottleneck)
        chans = self.base_c * self.retina.out_channels_factor
        layers = [{"type": "conv", "kernel_size": 4, "stride": 2, "filters": args.filters}]
        net_path = os.path.join(args.output_dir, f"viewer_net_{chans}x{args.grid}.json")
        os.makedirs(args.output_dir, exist_ok=True)
        build_fovea_network_json(net_path, channels=chans, size=args.grid,
                                 layers=layers, seed=args.seed)
        self.perc = PerceptionNetwork(net_path, self.ds_cfg, ablation=args.ablation)
        self.learning = True  # directive: no fixed weights
        self.paused = False
        self.img_idx = 0
        self.image = None
        self.label = None
        self.tick = 0
        self._load_image(args.img_index)

        n = self.perc.num_neurons
        self.gr, self.gc = _grid_shape(n)
        self.raster = deque(maxlen=args.raster_ticks)
        self.dyn = {k: deque(maxlen=args.dyn_ticks)
                    for k in ("participation", "meanS", "meanT", "efficacy")}
        self._eff = self.perc.mean_efficacy()

    def _load_image(self, idx):
        self.img_idx = idx % len(self.ds)
        image, label = self.ds[self.img_idx]
        self.image = image
        self.label = int(label)
        self.perc.reset()
        self.tick = 0

    def next_image(self):
        self._load_image(self.img_idx + 1)

    def reset_net(self):
        self.perc.reset()
        self.tick = 0

    def toggle_learning(self):
        self.learning = not self.learning
        self.perc.set_learning(self.learning)

    def set_gaze(self, cy, cx):
        self.retina.set_center(cy, cx)

    def bump_gain(self, factor):
        self.ds_cfg.signal_gain = float(np.clip(self.ds_cfg.signal_gain * factor, 0.1, 50))

    def step(self):
        """Advance one tick; return a dict of arrays for the artists."""
        retina_img = self.retina.render(self.image)
        sig = self.perc.patch_to_signals(retina_img)
        st = self.perc.step(sig)
        self.tick += 1
        part = float(st.O.mean())
        self.raster.append(st.O.copy())
        self.dyn["participation"].append(part)
        self.dyn["meanS"].append(float(st.S.mean()))
        self.dyn["meanT"].append(float(st.t_ref.mean()))
        if self.tick % self.args.efficacy_every == 0:
            self._eff = self.perc.mean_efficacy()
        self.dyn["efficacy"].append(self._eff)
        return {
            "retina": retina_img.detach().cpu().numpy(),
            "S": st.S, "O": st.O, "t_ref": st.t_ref,
            "part": part,
        }


# ----------------------------------------------------------------------------
def _build_figure(state, plt):
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])
    ax_img = fig.add_subplot(gs[0, 0])
    ax_fov = fig.add_subplot(gs[0, 1])
    ax_per = fig.add_subplot(gs[0, 2])
    ax_S = fig.add_subplot(gs[1, 0])
    ax_rast = fig.add_subplot(gs[1, 1])
    ax_dyn = fig.add_subplot(gs[1, 2])
    for a in (ax_img, ax_fov, ax_per, ax_S, ax_rast):
        a.axis("off")

    from matplotlib.patches import Rectangle
    img_np = state.image.mean(0).cpu().numpy() if state.base_c > 1 else state.image[0].cpu().numpy()
    art = {}
    art["img"] = ax_img.imshow(img_np, cmap="gray")
    fy, fx, fe = state.retina.fovea_box()
    py, px, pe = state.retina.periph_box()
    art["fbox"] = Rectangle((fx, fy), fe, fe, fill=False, ec="cyan", lw=2)
    art["pbox"] = Rectangle((px, py), pe, pe, fill=False, ec="yellow", lw=1, ls="--")
    ax_img.add_patch(art["fbox"]); ax_img.add_patch(art["pbox"])
    ax_img.set_title(f"gaze (drag) · label {state.label}", fontsize=9)

    art["fov"] = ax_fov.imshow(np.zeros((state.args.grid, state.args.grid)), cmap="gray")
    ax_fov.set_title("fovea (sharp)", fontsize=9)
    art["per"] = ax_per.imshow(np.zeros((state.args.grid, state.args.grid)), cmap="gray")
    ax_per.set_title("periphery (blurred)", fontsize=9)

    art["S"] = ax_S.imshow(np.full((state.gr, state.gc), np.nan), cmap="viridis")
    ax_S.set_title("per-neuron S (all neurons)", fontsize=9)
    art["rast"] = ax_rast.imshow(np.zeros((state.perc.num_neurons, 1)),
                                 cmap="gray_r", aspect="auto", vmin=0, vmax=1)
    ax_rast.set_title("spike raster (neurons x time)", fontsize=9)

    art["ax_dyn"] = ax_dyn
    art["ln_part"], = ax_dyn.plot([], [], label="participation", color="tab:blue")
    art["ln_eff"], = ax_dyn.plot([], [], label="mean efficacy (norm)", color="tab:green")
    art["ln_t"], = ax_dyn.plot([], [], label="mean t_ref (norm)", color="tab:red")
    ax_dyn.set_ylim(0, 1); ax_dyn.legend(fontsize=7, loc="upper left")
    ax_dyn.set_title("dynamics", fontsize=9)
    art["fig"] = fig
    art["ax_img"] = ax_img
    return fig, art


def _update_artists(state, art, frame):
    img_np = state.image.mean(0).cpu().numpy() if state.base_c > 1 else state.image[0].cpu().numpy()
    art["img"].set_data(img_np)
    art["img"].set_clim(img_np.min(), img_np.max())
    fy, fx, fe = state.retina.fovea_box()
    py, px, pe = state.retina.periph_box()
    art["fbox"].set_xy((fx, fy)); art["fbox"].set_width(fe); art["fbox"].set_height(fe)
    art["pbox"].set_xy((px, py)); art["pbox"].set_width(pe); art["pbox"].set_height(pe)

    ret = frame["retina"]
    C = state.base_c
    fov = ret[:C].mean(0) if C > 1 else ret[0]
    per = ret[C:].mean(0) if C > 1 else ret[C]
    art["fov"].set_data(fov); art["fov"].set_clim(fov.min(), fov.max())
    art["per"].set_data(per); art["per"].set_clim(per.min(), per.max())

    Sg = _to_grid(frame["S"], state.gr, state.gc)
    art["S"].set_data(Sg)
    finite = Sg[np.isfinite(Sg)]
    if finite.size:
        art["S"].set_clim(float(finite.min()), float(finite.max()) + 1e-6)

    if state.raster:
        R = np.stack(state.raster, axis=1)  # (N, T)
        art["rast"].set_data(R)
        art["rast"].set_extent([0, R.shape[1], R.shape[0], 0])

    def norm(seq, lo, hi):
        a = np.array(seq, dtype=np.float32)
        return (a - lo) / (hi - lo + 1e-9)
    d = state.dyn
    x = np.arange(len(d["participation"]))
    art["ln_part"].set_data(x, np.array(d["participation"]))
    if d["efficacy"]:
        eff = np.array(d["efficacy"]); art["ln_eff"].set_data(x, norm(eff, eff.min(), eff.max()))
    if d["meanT"]:
        t = np.array(d["meanT"]); art["ln_t"].set_data(x, norm(t, t.min(), t.max()))
    art["ax_dyn"].set_xlim(0, max(10, len(x)))
    lrn = "ON" if state.learning else "OFF"
    art["ax_img"].set_title(
        f"gaze (drag) · label {state.label} · learn {lrn} · gain "
        f"{state.ds_cfg.signal_gain:.1f} · part {frame['part']:.2f}", fontsize=9)


def run_play(state):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    fig, art = _build_figure(state, plt)

    drag = {"on": False}

    def on_press(e):
        if e.inaxes is art["ax_img"] and e.xdata is not None:
            drag["on"] = True
            state.set_gaze(e.ydata, e.xdata)

    def on_release(e):
        drag["on"] = False

    def on_motion(e):
        if drag["on"] and e.inaxes is art["ax_img"] and e.xdata is not None:
            state.set_gaze(e.ydata, e.xdata)

    def on_key(e):
        if e.key == " ":
            state.paused = not state.paused
        elif e.key == "l":
            state.toggle_learning()
        elif e.key == "r":
            state.reset_net()
        elif e.key == "n":
            state.next_image()
        elif e.key == "c":
            state.retina.center()
        elif e.key in ("+", "="):
            state.bump_gain(1.3)
        elif e.key == "-":
            state.bump_gain(1 / 1.3)

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("key_press_event", on_key)

    def update(_):
        if not state.paused:
            frame = state.step()
            _update_artists(state, art, frame)
        return []

    _anim = FuncAnimation(fig, update, interval=state.args.interval_ms, blit=False)
    plt.tight_layout()
    plt.show()


def run_record(state, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    fig, art = _build_figure(state, plt)
    T = state.args.record_ticks
    # Scripted Lissajous gaze path across the image.
    ts = np.arange(T)
    cy = state.H / 2 + 0.42 * state.H * np.sin(2 * np.pi * ts / 90)
    cx = state.W / 2 + 0.42 * state.W * np.sin(2 * np.pi * ts / 130 + 1.0)
    writer = FFMpegWriter(fps=state.args.fps, bitrate=3200)
    with writer.saving(fig, out_path, dpi=90):
        for t in range(T):
            state.set_gaze(cy[t], cx[t])
            frame = state.step()
            _update_artists(state, art, frame)
            writer.grab_frame()
    plt.close(fig)
    print(f"Saved {out_path}  (ticks={T}, final participation {frame['part']:.3f}, "
          f"efficacy {state.dyn['efficacy'][-1]:.4f})")


def main():
    p = argparse.ArgumentParser(description="Interactive fovea + neuron viewer")
    p.add_argument("--dataset-name", default="cifar10_grayscale")
    p.add_argument("--record", default="", help="path to mp4 (headless); empty = interactive")
    p.add_argument("--img-index", type=int, default=0)
    p.add_argument("--grid", type=int, default=16)
    p.add_argument("--fovea-extent", type=int, default=8)
    p.add_argument("--periph-extent", type=int, default=0, help="0 = whole image")
    p.add_argument("--periph-bottleneck", type=int, default=6)
    p.add_argument("--filters", type=int, default=4)
    p.add_argument("--gain", type=float, default=2.0)
    p.add_argument("--interval-ms", type=int, default=40)
    p.add_argument("--raster-ticks", type=int, default=120)
    p.add_argument("--dyn-ticks", type=int, default=200)
    p.add_argument("--efficacy-every", type=int, default=5)
    p.add_argument("--record-ticks", type=int, default=300)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--ablation", default="none")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="foveation_results")
    args = p.parse_args()

    state = ViewerState(args)
    print(f"{state.perc.num_neurons} neurons | retina {state.base_c}x{state.H}x{state.W} "
          f"-> ({state.base_c * 2},{args.grid},{args.grid}) | learning ON")
    if args.record:
        run_record(state, args.record)
    else:
        run_play(state)


if __name__ == "__main__":
    main()
