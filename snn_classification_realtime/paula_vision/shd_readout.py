"""Honest SHD train->test readout: record reservoir multi-window features for the SHD TRAIN split
and the held-out TEST split, fit closed-form (ncc/lda/ridge, no gradient) on train, score test.
Reuses shd_eval's frozen reservoir worker. This is the generalization number that matters."""
import argparse, time
import numpy as np, h5py
from collections import defaultdict
import multiprocessing as mp
from snn_classification_realtime.paula_vision import shd_eval as SE
from snn_classification_realtime.paula_vision.separability_probe import lda_acc, ridge_acc


def _load(path, n_per_class, classes):
    f = h5py.File(path, "r"); labels = f["labels"][:]
    by = defaultdict(list)
    for i in range(len(labels)):
        c = int(labels[i])
        if (classes is None or c in classes) and len(by[c]) < n_per_class: by[c].append(i)
    idx = [i for c in sorted(by) for i in by[c]]
    times = f["spikes"]["times"]; units = f["spikes"]["units"]
    return [(np.asarray(times[i]), np.asarray(units[i]), int(labels[i])) for i in idx]


def ncc_fit_eval(Xtr, ytr, Xte, yte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    R = (Xtr - mu) / sd; Q = (Xte - mu) / sd
    cls = np.unique(ytr); cents = np.stack([R[ytr == c].mean(0) for c in cls])
    d = np.linalg.norm(Q[:, None] - cents[None], axis=2)
    return float((cls[d.argmin(1)] == yte).mean())


def record(path, n_per_class, classes, init_args, workers, label):
    samples = _load(path, n_per_class, classes)
    t0 = time.time()
    with mp.Pool(workers, initializer=SE._init, initargs=init_args) as pool:
        out = pool.map(SE._feat, samples)
    X = np.stack([o[0] for o in out]); y = np.array([o[1] for o in out])
    print(f"  {label}: {len(X)} feat={X.shape[1]} ({time.time()-t0:.0f}s)", flush=True)
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True); ap.add_argument("--chan", required=True)
    ap.add_argument("--T", type=int, default=100); ap.add_argument("--dt", type=float, default=0.007)
    ap.add_argument("--settle", type=int, default=5); ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--train-per-class", type=int, default=120); ap.add_argument("--test-per-class", type=int, default=100)
    ap.add_argument("--classes", default=""); ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--tag", default="shd_ro")
    a = ap.parse_args()
    classes = [int(x) for x in a.classes.split(",")] if a.classes else None
    ncl = len(classes) if classes else 20
    init_args = (a.net, a.chan, a.T, a.dt, a.settle, a.K, "x")
    print(f"[{a.tag}] SHD train->test readout, {ncl} classes, K={a.K}", flush=True)
    Xtr, ytr = record("data/shd/shd_train.h5", a.train_per_class, classes, init_args, a.workers, "TRAIN")
    Xte, yte = record("data/shd/shd_test.h5", a.test_per_class, classes, init_args, a.workers, "TEST")
    accs = {"ncc": ncc_fit_eval(Xtr, ytr, Xte, yte), "lda": lda_acc(Xtr, ytr, Xte, yte),
            "ridge": ridge_acc(Xtr, ytr, Xte, yte)}
    best = max(accs.values())
    print(f"[{a.tag}] HELD-OUT TEST top-1 ({ncl}-class chance {1/ncl:.3f}): "
          + " ".join(f"{k}={v:.3f}" for k, v in accs.items()), flush=True)
    print(f"[{a.tag}] BEST = {best*100:.1f}%", flush=True)


if __name__ == "__main__":
    main()
