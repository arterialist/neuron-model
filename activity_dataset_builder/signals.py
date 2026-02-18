"""Signal grid computation and visualization."""

import os
from typing import TYPE_CHECKING, Any

import numpy as np

from neuron.network import NeuronNetwork

if TYPE_CHECKING:
    import torch

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def collect_tick_snapshot(
    network_sim: NeuronNetwork,
    layers: list[list[int]],
    tick_index: int,
) -> dict[str, Any]:
    """Collect per-layer arrays of neuron metrics for this tick."""
    net = network_sim.network
    layer_snapshots: list[dict[str, Any]] = []
    for layer_idx, layer_ids in enumerate(layers):
        layer_S: list[float] = []
        layer_F_avg: list[float] = []
        layer_t_ref: list[float] = []
        layer_fire: list[int] = []
        for nid in layer_ids:
            neuron = net.neurons[nid]
            layer_S.append(float(neuron.S))
            layer_F_avg.append(float(neuron.F_avg))
            layer_t_ref.append(float(neuron.t_ref))
            layer_fire.append(1 if neuron.O > 0 else 0)
        layer_snapshots.append({
            "layer_index": layer_idx,
            "neuron_ids": layer_ids,
            "S": layer_S,
            "F_avg": layer_F_avg,
            "t_ref": layer_t_ref,
            "fired": layer_fire,
        })
    return {"tick": tick_index, "layers": layer_snapshots}


def compute_signal_grid(image_tensor: "torch.Tensor") -> np.ndarray:
    """Compute a 2D grid of signal strengths matching the image's spatial size.

    Works for MNIST (1x28x28) and CIFAR (3x32x32) tensors normalized to [-1, 1].
    """
    arr = image_tensor.detach().cpu().numpy().astype(np.float32)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    if arr.ndim != 3:
        raise ValueError(f"Unsupported image tensor shape: {arr.shape}")
    arr01 = (arr + 1.0) * 0.5
    max_val = float(np.max(arr01))
    if max_val > 0:
        arr01 = arr01 / max_val
    return np.mean(arr01, axis=0)


def save_signal_plot(
    grid: np.ndarray,
    label: int,
    img_idx: int,
    out_dir: str = "plots",
) -> str:
    """Save a grayscale plot of the signal grid. Returns the saved file path."""
    if plt is None:
        return ""
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(grid, cmap="gray", vmin=0.0, vmax=1.0)
    ax.set_xticks([])
    ax.set_yticks([])

    h, w = grid.shape
    for y in range(h):
        for x in range(w):
            val = float(grid[y, x])
            txt_color = "black" if val > 0.6 else "white"
            ax.text(
                x, y, f"{val:.2f}",
                ha="center", va="center",
                fontsize=5, color=txt_color,
            )

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"signals_label_{label}_idx_{img_idx}.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path
