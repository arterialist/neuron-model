"""Part A -- push CIFAR-10 separability to >=0.50, storage-lean.

The original pipeline hit ~0.5 with conv-PAULA + trained SNN readout on the FULL 32x32
image, 2000 samples/label -- but dumped ~25GB. Here we compute readout FEATURES on the
fly (mean [S,F_avg,O] per sample, a few KB) and never store trajectories, so 10k+
samples cost tens of MB. Levers swept: full-image vs foveated, substrate size, sample
count, readout (linear/MLP), and a `convfeat` mode that reads the retinal-conv features
DIRECTLY (bypassing the substrate) to measure the encoder ceiling.
"""
from __future__ import annotations
import argparse, os, json, time
import numpy as np
from tqdm import tqdm

from snn_classification_realtime.activity_dataset_builder.vision_datasets import load_dataset_by_name
from snn_classification_realtime.foveation.minibrain.core import MiniBrainConfig, MiniBrain
from snn_classification_realtime.foveation.minibrain.exp_arch import best_linear
from snn_classification_realtime.foveation.minibrain.heads import TorchMLPHead


def feats(brain, ds, idxs, dwell, mode, desc):
    """mode='substrate' -> mean [S,F_avg,O] of readout layer; 'convfeat' -> the raw
    retinal-conv feature vector (encoder ceiling, no substrate)."""
    per = max(1, brain.cfg.input_period); m = brain._readout_mask
    X, Y = [], []
    brain.sub.set_learning(False)
    for i in tqdm(idxs, desc=desc, ncols=90):
        img, y = ds[int(i)]; Y.append(int(y))
        if mode == "convfeat":
            X.append(brain._encode(img).flatten().numpy())
            continue
        sig = brain.sub.patch_to_signals(brain._encode(img))
        st = [brain.sub.step((sig if t % per == 0 else []) + brain._tonic) for t in range(dwell)]
        X.append(np.concatenate([np.mean([s.S[m] for s in st], 0),
                                 np.mean([s.F_avg[m] for s in st], 0),
                                 np.mean([s.O[m] for s in st], 0)]))
    brain.sub.set_learning(True)
    return np.array(X, np.float32), np.array(Y)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--fovea", type=int, default=32); p.add_argument("--periph", type=int, default=32)
    p.add_argument("--grid", type=int, default=16)
    p.add_argument("--conv-bank", default="rich")
    p.add_argument("--substrate", default="reservoir", choices=["reservoir", "conv"])
    p.add_argument("--wiring", default="retinotopic")
    p.add_argument("--n-in", type=int, default=200); p.add_argument("--n-buffer", type=int, default=200)
    p.add_argument("--n-pool", type=int, default=400)
    p.add_argument("--samples", type=int, default=4000); p.add_argument("--test", type=int, default=1500)
    p.add_argument("--dwell", type=int, default=100); p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--mode", default="substrate", choices=["substrate", "convfeat"])
    p.add_argument("--head", default="mlp", choices=["linear", "mlp"])
    p.add_argument("--mlp-hidden", type=int, default=256)
    p.add_argument("--batch", type=int, default=0, help="0=full-batch; else minibatch size")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--device", default="cpu", help="cpu | mps | cuda | auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="scale"); p.add_argument("--out", default="foveation_results/minibrain/scale")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True); t0 = time.time()
    dstr = load_dataset_by_name(a.dataset, train=True); ds = dstr.dataset
    dste = load_dataset_by_name(a.dataset, train=False).dataset
    cfg = MiniBrainConfig(dataset_name=a.dataset, fovea=a.fovea, periph=a.periph, grid=a.grid,
                          conv_bank=a.conv_bank, substrate_type=a.substrate, wiring=a.wiring,
                          n_in=a.n_in, n_buffer=a.n_buffer, n_pool=a.n_pool, readout="all",
                          dwell=a.dwell, seed=a.seed, output_dir=os.path.join(a.out, f"net_{a.tag}"))
    brain = MiniBrain(cfg, dstr)
    if a.mode == "substrate" and a.warmup > 0:
        per = max(1, cfg.input_period); done = 0; i = 0; brain.sub.set_learning(False)
        while done < a.warmup:
            sig = brain.sub.patch_to_signals(brain._encode(ds[i % 50][0]))
            for t in range(cfg.dwell):
                brain.sub.step((sig if t % per == 0 else []) + brain._tonic); done += 1
                if done >= a.warmup: break
            i += 1
        brain.sub.set_learning(True)
    tr = np.random.RandomState(a.seed).randint(0, len(ds), size=a.samples)
    te = np.random.RandomState(a.seed + 1).randint(0, len(dste), size=a.test)
    Xtr, Ytr = feats(brain, ds, tr, a.dwell, a.mode, f"{a.tag} train")
    Xte, Yte = feats(brain, dste, te, a.dwell, a.mode, f"{a.tag} test")
    # normalize on train
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr_n = (Xtr - mu) / sd; Xte_n = (Xte - mu) / sd
    if a.head == "mlp":
        head = TorchMLPHead(Xtr.shape[1], 10, hidden=a.mlp_hidden, seed=a.seed, device=a.device)
        head.fit(Xtr, Ytr, epochs=a.epochs, batch=(a.batch or None))  # fit sets mu/sd internally
        acc = head.score(Xte, Yte)
    else:
        from sklearn.linear_model import LogisticRegression
        acc = max(LogisticRegression(max_iter=2000, C=C).fit(Xtr_n, Ytr).score(Xte_n, Yte)
                  for C in (0.002, 0.01, 0.05, 0.2))
    out = dict(exp="scale", tag=a.tag, dataset=a.dataset, mode=a.mode, head=a.head,
               fovea=a.fovea, grid=a.grid, substrate=a.substrate, n_pool=a.n_pool,
               dim=int(Xtr.shape[1]), n_neurons=int(brain.n_neurons), samples=a.samples,
               dwell=a.dwell, acc=float(acc), complete=True)
    json.dump(out, open(os.path.join(a.out, f"{a.tag}.json"), "w"))
    print(f"\n[{time.time()-t0:.0f}s] {a.tag}: mode={a.mode} head={a.head} dim={Xtr.shape[1]} "
          f"samples={a.samples} -> TEST ACC {acc:.3f}", flush=True)


if __name__ == "__main__":
    main()
