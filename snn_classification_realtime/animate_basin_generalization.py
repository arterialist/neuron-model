"""Animate class-basin sharpening from a run_basin_generalization npz.

Produces, over training checkpoints (n_train growing):
  * basin_<tag>.mp4  -- top row: shared-PCA scatter of the held-out TEST descriptors,
    colored by class, with per-class centroids, for each probe mode (frozen | on).
    Bottom row: leave-one-out nearest-own-centroid accuracy and own-vs-other margin vs
    n_train, drawn up to the current checkpoint, for both modes.
  * basin_<tag>_metrics.png -- static final curves (acc + margin, frozen vs on).

A single global PCA basis (fit on all test descriptors pooled across checkpoints and
modes, after global standardization) keeps every frame in the same projection so the
motion is real, not a per-frame re-embedding. CLI-only.
"""
import argparse
import os

import numpy as np


def load(npz):
    z = np.load(npz, allow_pickle=True)
    modes = [str(m) for m in z["modes"]]
    return dict(
        n_train=z["n_train"], weight=z["weight"], test_lab=z["test_lab"],
        classes=z["classes"].tolist(), modes=modes, ticks=int(z["ticks"]),
        desc={m: z[f"desc_{m}"] for m in modes},        # (C, n_test, D)
        metrics={m: z[f"metrics_{m}"] for m in modes},   # (C, 3) acc,margin,betw/within
    )


def global_pca(desc_by_mode):
    X = np.concatenate([d.reshape(-1, d.shape[-1]) for d in desc_by_mode.values()], 0)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd
    gm = Xs.mean(0)
    _, _, Vt = np.linalg.svd(Xs - gm, full_matrices=False)
    return mu, sd, gm, Vt[:2]


def project(d, mu, sd, gm, V):
    return ((d - mu) / sd - gm) @ V.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=int, default=4)
    ap.add_argument("--dpi", type=int, default=120)
    a = ap.parse_args()
    R = load(a.npz)
    tag = os.path.splitext(os.path.basename(a.npz))[0]
    out = a.out or os.path.dirname(a.npz) or "."
    os.makedirs(out, exist_ok=True)
    modes = R["modes"]; classes = R["classes"]; lab = R["test_lab"]
    C = len(R["n_train"]); chance = 1.0 / len(classes)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    cmap = plt.get_cmap("tab10")
    ccolor = {c: cmap(i % 10) for i, c in enumerate(classes)}
    pt_col = np.array([ccolor[int(c)] for c in lab])

    mu, sd, gm, V = global_pca(R["desc"])
    emb = {m: np.stack([project(R["desc"][m][ci], mu, sd, gm, V) for ci in range(C)]) for m in modes}
    # fixed axis limits across all frames/modes
    allE = np.concatenate([emb[m].reshape(-1, 2) for m in modes], 0)
    pad = 0.08 * (allE.max(0) - allE.min(0) + 1e-9)
    xlim = (allE[:, 0].min() - pad[0], allE[:, 0].max() + pad[0])
    ylim = (allE[:, 1].min() - pad[1], allE[:, 1].max() + pad[1])

    fig = plt.figure(figsize=(12, 8.5))
    gs = fig.add_gridspec(2, len(modes), height_ratios=[3, 1.7], hspace=0.28, wspace=0.15)
    sc_ax = [fig.add_subplot(gs[0, j]) for j in range(len(modes))]
    m_ax = fig.add_subplot(gs[1, :])
    m_ax2 = m_ax.twinx()

    def draw(ci):
        for j, m in enumerate(modes):
            ax = sc_ax[j]; ax.clear()
            E = emb[m][ci]
            ax.scatter(E[:, 0], E[:, 1], c=pt_col, s=34, edgecolors="white", linewidths=0.4, zorder=3)
            for c in classes:
                mm = lab == c
                if mm.any():
                    ct = E[mm].mean(0)
                    ax.scatter(*ct, color=ccolor[c], s=240, marker="*",
                               edgecolors="black", linewidths=0.9, zorder=4)
                    ax.annotate(str(c), ct, fontsize=11, fontweight="bold", zorder=5,
                                ha="center", va="center")
            acc, margin, bw = R["metrics"][m][ci]
            ax.set_xlim(*xlim); ax.set_ylim(*ylim)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{m}   NC_acc={acc:.2f}  margin={margin:+.2f}  betw/within={bw:.2f}",
                         fontsize=11)
        m_ax.clear(); m_ax2.clear()
        nt = R["n_train"][:ci + 1]
        for m, ls in zip(modes, ["-", "--"]):
            met = R["metrics"][m][:ci + 1]
            m_ax.plot(nt, met[:, 0], ls, color="#0d818f", marker="o", ms=3, label=f"{m} NC_acc")
            m_ax2.plot(nt, met[:, 1], ls, color="#b45309", marker="s", ms=3, label=f"{m} margin")
        m_ax.axhline(chance, color="k", ls=":", lw=1)
        m_ax.text(R["n_train"][0], chance + 0.005, f"chance {chance:.2f}", fontsize=8)
        m_ax.set_xlim(R["n_train"][0], R["n_train"][-1] + 1e-9)
        m_ax.set_ylim(0, max(0.6, R["metrics"][modes[0]][:, 0].max() * 1.15))
        m_ax.set_xlabel("training samples seen (basin formation)")
        m_ax.set_ylabel("nearest-own-centroid acc", color="#0d818f")
        m_ax2.yaxis.set_label_position("right"); m_ax2.yaxis.tick_right()
        m_ax2.set_ylabel("own-vs-other margin", color="#b45309")
        h1, l1 = m_ax.get_legend_handles_labels(); h2, l2 = m_ax2.get_legend_handles_labels()
        m_ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left", ncol=2)
        fig.suptitle(f"{tag}: do unseen test samples fall nearer their OWN class basin as "
                     f"training proceeds?  (n_train={R['n_train'][ci]})", fontsize=13)

    anim = animation.FuncAnimation(fig, draw, frames=C, interval=1000 // a.fps)
    mp4 = os.path.join(out, f"basin_{tag}.mp4")
    try:
        anim.save(mp4, writer=animation.FFMpegWriter(fps=a.fps, bitrate=2400), dpi=a.dpi)
        print(f"wrote {mp4}")
    except Exception as e:
        gif = os.path.join(out, f"basin_{tag}.gif")
        anim.save(gif, writer=animation.PillowWriter(fps=a.fps), dpi=a.dpi)
        print(f"ffmpeg failed ({e}); wrote {gif}")
    plt.close(fig)

    # static final metric comparison
    fig2, ax = plt.subplots(figsize=(9, 5)); ax2 = ax.twinx()
    for m, ls in zip(modes, ["-", "--"]):
        met = R["metrics"][m]
        ax.plot(R["n_train"], met[:, 0], ls, color="#0d818f", marker="o", ms=4, label=f"{m} NC_acc")
        ax2.plot(R["n_train"], met[:, 1], ls, color="#b45309", marker="s", ms=4, label=f"{m} margin")
    ax.axhline(chance, color="k", ls=":", lw=1); ax.text(R["n_train"][0], chance + 0.005, f"chance {chance:.2f}", fontsize=8)
    ax.set_xlabel("training samples seen"); ax.set_ylabel("nearest-own-centroid acc", color="#0d818f")
    ax2.set_ylabel("own-vs-other margin", color="#b45309")
    ax.set_title(f"{tag}: class-basin sharpening for genuinely-unseen test samples")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9, loc="best")
    fig2.tight_layout(); png = os.path.join(out, f"basin_{tag}_metrics.png")
    fig2.savefig(png, dpi=130); plt.close(fig2); print(f"wrote {png}")


if __name__ == "__main__":
    main()
