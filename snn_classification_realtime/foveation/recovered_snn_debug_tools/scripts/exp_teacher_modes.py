"""Does a self-balancing RPE teacher make the substrate MORE separable, where the
all-stress 'graded' teacher made it LESS? Frozen reservoir; only input-layer plastic.

Arms (fresh brain, same seed each): control (teacher off), graded, rpe.
Measure pool-readout linear separability BEFORE vs AFTER a teacher-on stream.
The load-bearing hypothesis: teacher feedback should RAISE separability.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from snn_classification_realtime.activity_dataset_builder.vision_datasets import (
    load_dataset_by_name,
)
from snn_classification_realtime.foveation.minibrain.core import MiniBrainConfig, MiniBrain

def main():
    ds_cfg = load_dataset_by_name("cifar10_grayscale", train=True)
    ds = ds_cfg.dataset
    Cs = (0.0005, 0.002, 0.01, 0.05)

    def best_acc(X, ys):
        X = np.asarray(X, float); n = len(ys); ntr = int(n*0.7)
        mu, sd = X[:ntr].mean(0), X[:ntr].std(0)+1e-6; Xn = (X-mu)/sd
        return max(LogisticRegression(max_iter=2000, C=C).fit(Xn[:ntr], ys[:ntr]).score(Xn[ntr:], ys[ntr:]) for C in Cs)

    def probe(brain, idxs):
        brain.sub.set_learning(False)
        m = brain._readout_mask; per = max(1, brain.cfg.input_period)
        X, Y = [], []
        for i in idxs:
            img, y = ds[i]; Y.append(int(y))
            sig = brain.sub.patch_to_signals(brain.retina.render(img))
            st = [brain.sub.step((sig if t % per == 0 else []) + brain._tonic) for t in range(brain.cfg.dwell)]
            m_ = brain._readout_mask
            S=np.mean([s.S[m_] for s in st],0); F=np.mean([s.F_avg[m_] for s in st],0); O=np.mean([s.O[m_] for s in st],0)
            X.append(np.concatenate([S,F,O]))
        brain.sub.set_learning(True)
        return best_acc(X, np.array(Y))

    rng = np.random.RandomState(0)
    probe_idx = rng.randint(0, len(ds), size=300)
    train_idx = rng.randint(0, len(ds), size=700)

    def make(mode):
        return MiniBrainConfig(dataset_name="cifar10_grayscale", tonic_drive=0.15,
                               target_participation=0.05, dwell=40, seed=0,
                               freeze_reservoir=True, teacher_mode=mode,
                               reward_gain=3.0, stress_gain=3.0)  # rpe deltas are small -> scale up

    # before (same for all: same seed/init)
    b0 = MiniBrain(make("rpe"), ds_cfg)
    before = probe(b0, probe_idx)
    print(f"pool separability BEFORE any training: {before:.3f}  (retina input ~0.18)")

    for mode, teach in [("control", False), ("graded", True), ("rpe", True)]:
        tm = mode if teach else "rpe"
        brain = MiniBrain(make(tm), ds_cfg)
        last = (0.0, 0.0)
        for idx in train_idx:
            img, y = ds[idx]
            brain.present(img, int(y), learn=True, teach=teach)
            last = getattr(brain, "_last_nm", last)
        after = probe(brain, probe_idx)
        print(f"  {mode:8s}: AFTER {after:.3f}   delta {after-before:+.3f}   "
              f"last m0/m1 {last[0]:.2f}/{last[1]:.2f}")


if __name__ == "__main__":
    main()
