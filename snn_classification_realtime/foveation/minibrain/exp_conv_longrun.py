"""Long-timescale crux test WITH retinal convolutions.

Short runs showed the teacher <= unsupervised control (both pinned near the single-
glimpse ceiling). Two things changed: (1) retinal-conv front-end lifts the ceiling
(gray 0.22->0.34), giving the teacher headroom; (2) run LONG so slow teacher-driven
memory can form. Probe the frozen-eval pool separability TRAJECTORY over a long
teacher-on stream vs an unsupervised control, and see whether the teacher pulls ahead.

Frozen reservoir (plasticity rails it); input layer + decoder plastic. rpe teacher.
"""
from __future__ import annotations

import argparse, os, time
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from snn_classification_realtime.activity_dataset_builder.vision_datasets import (
    load_dataset_by_name,
)
from snn_classification_realtime.foveation.minibrain.core import MiniBrainConfig, MiniBrain

Cs = (0.0005, 0.002, 0.01, 0.05)
def best_acc(X, ys):
    X = np.asarray(X, float); n = len(ys); ntr = int(n * 0.7)
    mu, sd = X[:ntr].mean(0), X[:ntr].std(0) + 1e-6; Xn = (X - mu) / sd
    return max(LogisticRegression(max_iter=1500, C=C).fit(Xn[:ntr], ys[:ntr]).score(Xn[ntr:], ys[ntr:]) for C in Cs)

def probe(brain, ds, idxs):
    brain.sub.set_learning(False)
    m = brain._readout_mask; per = max(1, brain.cfg.input_period)
    X, Y = [], []
    for i in idxs:
        img, y = ds[i]; Y.append(int(y))
        sig = brain.sub.patch_to_signals(brain._encode(img))
        st = [brain.sub.step((sig if t % per == 0 else []) + brain._tonic) for t in range(brain.cfg.dwell)]
        S = np.mean([s.S[m] for s in st], 0); F = np.mean([s.F_avg[m] for s in st], 0)
        O = np.mean([s.O[m] for s in st], 0)
        X.append(np.concatenate([S, F, O]))
    brain.sub.set_learning(True)
    return best_acc(X, np.array(Y))

def run_arm(cfg, ds, order, probe_idx, probe_every, learn, teach, tag, log):
    # decoder trains in EVERY arm (train_decoder=True) so the control is clean.
    brain = MiniBrain(cfg, load_dataset_by_name(cfg.dataset_name, train=True))
    traj_t, traj_s, traj_oa = [], [], []
    base = probe(brain, ds, probe_idx)
    traj_t.append(0); traj_s.append(base); traj_oa.append(0.0)
    log(f"    [{tag}] t=0 baseline sep {base:.3f}")
    correct = 0
    for i, idx in enumerate(order):
        img, y = ds[idx]; y = int(y)
        _, pred = brain.present(img, y, learn=learn, teach=teach, train_decoder=True)
        correct += int(pred == y)
        if (i + 1) % probe_every == 0:
            s = probe(brain, ds, probe_idx)
            oa = correct / probe_every; correct = 0
            traj_t.append(i + 1); traj_s.append(s); traj_oa.append(oa)
            log(f"    [{tag}] {i+1:5d} sep {s:.3f} online {oa:.3f} "
                f"m0/m1 {getattr(brain,'_last_nm',(0,0))[0]:.2f}/{getattr(brain,'_last_nm',(0,0))[1]:.2f}")
    return brain, (traj_t, traj_s, traj_oa)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-name", default="cifar10_grayscale")
    p.add_argument("--images", type=int, default=2400)
    p.add_argument("--dwell", type=int, default=50)
    p.add_argument("--input-period", type=int, default=1,
                   help=">1 = intermittent input: silent gaps drive LTD (constructive?)")
    p.add_argument("--probe-every", type=int, default=400)
    p.add_argument("--probe-n", type=int, default=300)
    p.add_argument("--reward-gain", type=float, default=3.0)
    p.add_argument("--stress-gain", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="foveation_results/minibrain/conv_longrun")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    lf = open(os.path.join(args.out, f"{args.dataset_name}.log"), "a")
    def log(m): print(m, flush=True); lf.write(m + "\n"); lf.flush()

    ds = load_dataset_by_name(args.dataset_name, train=True).dataset
    rng = np.random.RandomState(args.seed)
    order = rng.randint(0, len(ds), size=args.images)
    probe_idx = rng.randint(0, len(ds), size=args.probe_n)

    def mk(mode):
        return MiniBrainConfig(dataset_name=args.dataset_name, retinal_conv=True,
                               tonic_drive=0.15, target_participation=0.05,
                               dwell=args.dwell, freeze_reservoir=True,
                               input_period=args.input_period,
                               teacher_mode=mode, reward_gain=args.reward_gain,
                               stress_gain=args.stress_gain, seed=args.seed)
    t0 = time.time()
    log(f"\n=== conv long-run | {args.dataset_name} | {args.images} imgs dwell {args.dwell} ===")
    # three clean arms; decoder trains in all three.
    log("--- FROZEN substrate + trained decoder (reservoir computing) ---")
    _, frz = run_arm(mk("rpe"), ds, order, probe_idx, args.probe_every, False, False, "FROZ ", log)
    log("--- PLASTIC control (substrate learns, NO teacher) ---")
    _, ctrl = run_arm(mk("rpe"), ds, order, probe_idx, args.probe_every, True, False, "CTRL ", log)
    log("--- TEACHER (substrate learns + rpe neuromod) ---")
    _, teach = run_arm(mk("rpe"), ds, order, probe_idx, args.probe_every, True, True, "TEACH", log)
    log(f"[{time.time()-t0:.0f}s] FINAL sep  frozen {frz[1][-1]:.3f}  plastic-ctrl "
        f"{ctrl[1][-1]:.3f}  teacher {teach[1][-1]:.3f}")
    log(f"          FINAL online-acc  frozen {frz[2][-1]:.3f}  plastic-ctrl "
        f"{ctrl[2][-1]:.3f}  teacher {teach[2][-1]:.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for tr, c, lab in [(frz, "tab:blue", "frozen"), (ctrl, "tab:gray", "plastic ctrl"),
                       (teach, "tab:green", "teacher rpe")]:
        ax[0].plot(tr[0], tr[1], "-o", color=c, ms=3, label=lab)
        ax[1].plot(tr[0], tr[2], "-o", color=c, ms=3, label=lab)
    ax[0].axhline(0.1, color="k", ls=":", lw=1, label="chance")
    ax[0].set_title(f"substrate separability (frozen probe) — {args.dataset_name}")
    ax[0].set_xlabel("images seen"); ax[0].set_ylabel("logreg test acc"); ax[0].legend(fontsize=8)
    ax[1].axhline(0.1, color="k", ls=":", lw=1)
    ax[1].set_title("online decoder acc (trained in ALL arms)")
    ax[1].set_xlabel("images seen"); ax[1].legend(fontsize=8)
    fig.tight_layout()
    png = os.path.join(args.out, f"{args.dataset_name}_trajectory.png")
    fig.savefig(png, dpi=120); log(f"saved {png}")

if __name__ == "__main__":
    main()
