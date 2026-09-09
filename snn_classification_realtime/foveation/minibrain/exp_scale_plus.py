"""Part A+ -- push the encoder-ceiling CIFAR accuracy past the 0.566 (40k-sample) result.

Levers, all within the fixed-PAULA-retinal-conv-front-end + trained-readout spirit:
  * FULL unique train set (50k, no sampling-with-replacement duplicates) + optional
    horizontal-flip augmentation (2x, the single most reliable fixed-feature CIFAR trick).
  * A bigger regularized MLP readout (hidden 1024, dropout) trained longer on MPS, with
    minibatches STREAMED to the GPU so the ~3.7GB feature matrix never has to be resident.

Features are the raw retinal-conv vector (convfeat / encoder ceiling), computed on the fly,
never dumped to disk (storage-lean). Reports test accuracy.

    PYTHONPATH=. .venv/bin/python -m \
        snn_classification_realtime.foveation.minibrain.exp_scale_plus \
        --aug --hidden 1024 --dropout 0.3 --epochs 250 --batch 512 --device mps
"""
from __future__ import annotations
import argparse, os, json, time
import numpy as np
import torch
from tqdm import tqdm

from snn_classification_realtime.activity_dataset_builder.vision_datasets import load_dataset_by_name
from snn_classification_realtime.foveation.minibrain.core import MiniBrainConfig, MiniBrain
from snn_classification_realtime.foveation.minibrain.heads import TorchMLPHead


def extract(brain, ds, idxs, aug, desc):
    X, Y = [], []
    for i in tqdm(idxs, desc=desc, ncols=90):
        img, y = ds[int(i)]
        X.append(brain._encode(img).flatten().numpy()); Y.append(int(y))
        if aug:
            X.append(brain._encode(torch.flip(img, dims=[-1])).flatten().numpy()); Y.append(int(y))
    return np.asarray(X, np.float32), np.asarray(Y, np.int64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--grid", type=int, default=16); p.add_argument("--conv-bank", default="rich")
    p.add_argument("--samples", type=int, default=0, help="0 = full unique train set")
    p.add_argument("--test", type=int, default=10000)
    p.add_argument("--aug", action="store_true", help="add horizontal-flip augmentation (2x train)")
    p.add_argument("--hidden", type=int, default=1024); p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--epochs", type=int, default=250); p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-3); p.add_argument("--wd", type=float, default=2e-3)
    p.add_argument("--device", default="mps"); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="plus"); p.add_argument("--out", default="foveation_results/minibrain/scale")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True); t0 = time.time()
    ds = load_dataset_by_name(a.dataset, train=True).dataset
    dste = load_dataset_by_name(a.dataset, train=False).dataset
    cfg = MiniBrainConfig(dataset_name=a.dataset, fovea=32, periph=32, grid=a.grid,
                          conv_bank=a.conv_bank, readout="all",
                          output_dir=os.path.join(a.out, f"net_{a.tag}"))
    brain = MiniBrain(cfg, load_dataset_by_name(a.dataset, train=True))

    ntr = len(ds) if a.samples == 0 else min(a.samples, len(ds))
    tr_idx = np.arange(ntr)                              # unique, no replacement
    te_idx = np.arange(min(a.test, len(dste)))
    Xtr, Ytr = extract(brain, ds, tr_idx, a.aug, f"{a.tag} train{'+aug' if a.aug else ''}")
    Xte, Yte = extract(brain, dste, te_idx, False, f"{a.tag} test")
    print(f"[{time.time()-t0:.0f}s] features: train {Xtr.shape} test {Xte.shape}", flush=True)

    head = TorchMLPHead(Xtr.shape[1], 10, hidden=a.hidden, seed=a.seed,
                        device=a.device, dropout=a.dropout)
    best_val = head.fit(Xtr, Ytr, epochs=a.epochs, lr=a.lr, wd=a.wd, batch=a.batch, stream=True)
    acc = head.score(Xte, Yte)
    out = dict(exp="scale_plus", tag=a.tag, dataset=a.dataset, grid=a.grid, aug=bool(a.aug),
               dim=int(Xtr.shape[1]), n_train=int(Xtr.shape[0]), n_test=int(Xte.shape[0]),
               hidden=a.hidden, dropout=a.dropout, epochs=a.epochs,
               best_val=round(float(best_val), 4), acc=float(acc), complete=True)
    json.dump(out, open(os.path.join(a.out, f"{a.tag}.json"), "w"))
    print(f"\n[{time.time()-t0:.0f}s] {a.tag}: aug={a.aug} hidden={a.hidden} n_train={Xtr.shape[0]} "
          f"-> TEST ACC {acc:.4f} (val {best_val:.3f})", flush=True)


if __name__ == "__main__":
    main()
