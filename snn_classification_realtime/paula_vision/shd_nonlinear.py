"""DECISIVE real-data test: does PAULA's nonlinear temporal computation ADD value on REAL SHD
spikes where a linear readout provably CANNOT? Label = fL XOR fH computed from the real spike
stream: fL = (spikes in channels 0-349 during FIRST half of time) > median; fH = (spikes in
channels 350-699 during SECOND half) > median. Both fL,fH are individually LINEAR projections of
the raw spike features, but their XOR is NOT -> raw readout ~0.5. A reservoir that holds fL
(early/low) and nonlinearly combines with fH (late/high) can. Real spike statistics, nonlinear-
temporal label. Reuses shd_online's real-spike loader + parallel frozen-reservoir extraction."""
import argparse
import numpy as np, multiprocessing as mp
from snn_classification_realtime.paula_vision.shd_online import load_shd, _init, _feat, raw_feat
from snn_classification_realtime.paula_vision.separability_probe import ridge_acc
from snn_classification_realtime.paula_vision.online_decoder import online_reward_competitive

def xor_label(sample, T, dt, medL, medH):
    times, units = sample; thalf = T * dt / 2
    fL = sum(1 for tm, u in zip(times, units) if tm < thalf and u < 350)
    fH = sum(1 for tm, u in zip(times, units) if thalf <= tm < T * dt and u >= 350)
    return int(fL > medL) ^ int(fH > medH), fL, fH

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True); ap.add_argument("--chan", required=True)
    ap.add_argument("--T", type=int, default=100); ap.add_argument("--dt", type=float, default=0.007)
    ap.add_argument("--settle", type=int, default=8); ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--ntr", type=int, default=80); ap.add_argument("--nte", type=int, default=40)
    ap.add_argument("--workers", type=int, default=12); ap.add_argument("--tag", default="shdxor")
    a = ap.parse_args()
    classes = list(range(20))
    tr_s, _ = load_shd("data/shd/shd_train.h5", a.ntr, classes)
    te_s, _ = load_shd("data/shd/shd_test.h5", a.nte, classes)
    # thresholds (medians) from TRAIN raw feature values
    Ls = []; Hs = []
    for s in tr_s:
        _, fL, fH = xor_label(s, a.T, a.dt, 0, 0); Ls.append(fL); Hs.append(fH)
    medL = float(np.median(Ls)); medH = float(np.median(Hs))
    def balance(samples):
        # subsample to EQUAL (fL,fH) combo counts so XOR label is exactly 50/50 AND fL,fH are each
        # uncorrelated with the label -> a linear readout provably cannot exceed chance via any
        # correlation; only genuine XOR (nonlinear) can. Removes the spike-budget anti-correlation.
        info = [(s, *xor_label(s, a.T, a.dt, medL, medH)) for s in samples]
        buckets = {(0,0):[], (0,1):[], (1,0):[], (1,1):[]}
        for s, lab, fL, fH in info: buckets[(int(fL>medL), int(fH>medH))].append((s, lab))
        m = min(len(v) for v in buckets.values())
        rng = np.random.RandomState(0); out = []
        for v in buckets.values():
            idx = rng.choice(len(v), m, replace=False); out += [v[i] for i in idx]
        rng.shuffle(out)
        return [o[0] for o in out], np.array([o[1] for o in out])
    tr_s, ytr = balance(tr_s); te_s, yte = balance(te_s)
    # balance check: 4 combos
    combos = np.array([(int(xor_label(s,a.T,a.dt,medL,medH)[1]>medL), int(xor_label(s,a.T,a.dt,medL,medH)[2]>medH)) for s in tr_s])
    print(f"[{a.tag}] real SHD spikes, label=fL XOR fH. train={len(tr_s)} test={len(te_s)} "
          f"medL={medL:.0f} medH={medH:.0f} label-balance tr={ytr.mean():.2f} te={yte.mean():.2f}", flush=True)
    print(f"[{a.tag}] (fL,fH) combo counts: 00={((combos==[0,0]).all(1)).sum()} 01={((combos==[0,1]).all(1)).sum()} "
          f"10={((combos==[1,0]).all(1)).sum()} 11={((combos==[1,1]).all(1)).sum()}", flush=True)
    # PAULA reservoir features (parallel, frozen)
    with mp.Pool(a.workers, initializer=_init, initargs=(a.net, a.chan, a.T, a.dt, a.settle, a.K)) as pool:
        Xtr = np.stack(pool.map(_feat, tr_s)); Xte = np.stack(pool.map(_feat, te_s))
    cf = ridge_acc(Xtr, ytr, Xte, yte)
    on, curve = online_reward_competitive(Xtr, ytr, Xte, yte, 2, eta=0.02, rh_decay=0.1, epochs=15)
    # raw baseline
    Rtr = np.stack([raw_feat(s, a.T, a.dt, a.K) for s in tr_s]); Rte = np.stack([raw_feat(s, a.T, a.dt, a.K) for s in te_s])
    cfr = ridge_acc(Rtr, ytr, Rte, yte)
    onr, curver = online_reward_competitive(Rtr, ytr, Rte, yte, 2, eta=0.02, rh_decay=0.1, epochs=15)
    print(f"[{a.tag}] RAW baseline (must ~=0.5, XOR not linearly separable): ridge={cfr:.3f} online={curver[-1]:.3f}", flush=True)
    print(f"[{a.tag}] PAULA reservoir (nonlinear temporal): ridge={cf:.3f} online={curve[-1]:.3f} (chance 0.5)", flush=True)
    print(f"[{a.tag}] SUBSTRATE ADDS on REAL data: ridge={cf-cfr:+.3f} online={curve[-1]-curver[-1]:+.3f}", flush=True)
