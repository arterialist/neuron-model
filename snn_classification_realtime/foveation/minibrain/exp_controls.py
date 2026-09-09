"""PAULA ATTRIBUTION LADDER -- is the spiking substrate doing real work, or is it the
fixed retinal convolution + the MLP readout?

For one fixed sample set we build FOUR representations and read each with BOTH a LINEAR
decoder and an MLP:

  1. pixel    : raw image                         -> readout floor (no retina, no PAULA)
  2. retina   : fixed DoG/Gabor conv features     -> what the hand-designed front-end buys
  3. substrate: PAULA spiking activity (mean [S,F_avg,O] over the readout layer after
                `dwell` ticks) -- the ONLY stage that is PAULA
  4. randproj : tanh(W.retina) with W fixed random, dim-matched to the substrate feature
                -> a GENERIC nonlinear expansion of the same size (control for "any
                   nonlinearity of this width would do")

Decisive verdicts (LINEAR readouts, because an MLP can mask bad features):
  * paula_lifts_linear = L(substrate) - L(retina)   >0 => PAULA ADDS linear separability
                                                    <0 => PAULA is LOSSY (destroys signal)
  * paula_vs_random    = L(substrate) - L(randproj) >0 => it's the DYNAMICS, not just width
  * mlp_compensation   = M(substrate) - L(substrate) how much the MLP is carrying
  * retina_over_pixel  = L(retina)   - L(pixel)      the front-end's own contribution

Storage-lean: features computed on the fly, kept in RAM (a few hundred MB), never dumped.
One process per substrate arch (parallel). Resumable (skips complete arm JSON).

    PYTHONPATH=. OMP_NUM_THREADS=1 .venv/bin/python -m \
        snn_classification_realtime.foveation.minibrain.exp_controls \
        --archs conv,reservoir_random,reservoir_retino --samples 1500 --dwell 300
"""
from __future__ import annotations
import argparse, os, json, time
from multiprocessing import Process
import numpy as np
from tqdm import tqdm

from snn_classification_realtime.activity_dataset_builder.vision_datasets import load_dataset_by_name
from snn_classification_realtime.foveation.minibrain.core import MiniBrainConfig, MiniBrain
from snn_classification_realtime.foveation.minibrain.exp_arch import best_linear
from snn_classification_realtime.foveation.minibrain.heads import TorchMLPHead


def make_cfg(arch, dataset, out_dir, seed, dwell):
    substrate = "conv" if arch == "conv" else "reservoir"
    wiring = "retinotopic" if arch == "reservoir_retino" else "random"
    return MiniBrainConfig(dataset_name=dataset, dwell=dwell, seed=seed, substrate_type=substrate,
                           wiring=wiring, conv_bank="rich", readout="all",
                           fovea=32, periph=32, grid=16, output_dir=out_dir)


def warmup(brain, ds, ticks, seed):
    order = np.random.RandomState(seed).randint(0, len(ds), size=50)
    per = max(1, brain.cfg.input_period); done = 0; i = 0
    brain.sub.set_learning(False)
    while done < ticks:
        sig = brain.sub.patch_to_signals(brain._encode(ds[int(order[i % 50])][0]))
        for t in range(brain.cfg.dwell):
            brain.sub.step((sig if t % per == 0 else []) + brain._tonic); done += 1
            if done >= ticks:
                break
        i += 1
    brain.sub.set_learning(True)


def mlp_acc(X, Y, split, seed, device):
    n = len(Y); ntr = int(n * split)
    head = TorchMLPHead(X.shape[1], 10, hidden=256, seed=seed, device=device)
    head.fit(X[:ntr], Y[:ntr], epochs=150, batch=256)
    return float(head.score(X[ntr:], Y[ntr:]))


def shard_extract(arch, dataset, seed, idx_slice, dwell, warm, spath, pos, desc):
    """Compute pixel/retina/substrate features for a slice of images; save npz. The
    substrate step-loop is the expensive part, so we parallelize it across shards."""
    if os.path.exists(spath):
        return
    ds = load_dataset_by_name(dataset, train=True).dataset
    cfg = make_cfg(arch, dataset, spath + "_net", seed, dwell)
    brain = MiniBrain(cfg, load_dataset_by_name(dataset, train=True))
    warmup(brain, ds, warm, seed)
    per = max(1, cfg.input_period); m = brain._readout_mask
    brain.sub.set_learning(False)
    Xpix, Xret, Xsub, Y = [], [], [], []
    for i in tqdm(idx_slice, desc=desc, position=pos, ncols=90, leave=False):
        img, y = ds[int(i)]; Y.append(int(y))
        Xpix.append(np.asarray(img).ravel())
        ret = brain._encode(img).flatten().numpy(); Xret.append(ret)
        sig = brain.sub.patch_to_signals(brain._encode(img))
        st = [brain.sub.step((sig if t % per == 0 else []) + brain._tonic) for t in range(dwell)]
        Xsub.append(np.concatenate([np.mean([s.S[m] for s in st], 0),
                                    np.mean([s.F_avg[m] for s in st], 0),
                                    np.mean([s.O[m] for s in st], 0)]))
    np.savez_compressed(spath, Xpix=np.asarray(Xpix, np.float32), Xret=np.asarray(Xret, np.float32),
                        Xsub=np.asarray(Xsub, np.float32), Y=np.asarray(Y))


def run_arm(arch, dataset, samples, test, dwell, warm, seed, split, device, out, shards=3):
    tag = f"ctrl_{arch}_{dataset.replace('cifar10', 'c10')}"
    jpath = os.path.join(out, f"{tag}.json")
    if os.path.exists(jpath):
        try:
            if json.load(open(jpath)).get("complete"):
                print(f"[skip] {tag}", flush=True); return
        except Exception:
            pass
    t0 = time.time()
    n = samples + test
    idxs = np.random.RandomState(seed + 3).permutation(
        len(load_dataset_by_name(dataset, train=True).dataset))[:n]
    sdir = os.path.join(out, tag + "_shards"); os.makedirs(sdir, exist_ok=True)
    slices = np.array_split(idxs, shards); procs = []
    for si, sl in enumerate(slices):
        spath = os.path.join(sdir, f"sh{si}.npz")
        pr = Process(target=shard_extract, args=(arch, dataset, seed, sl, dwell, warm,
                                                 spath, si, f"{tag} sh{si}"))
        pr.start(); procs.append(pr)
    for pr in procs:
        pr.join()
    parts = [np.load(os.path.join(sdir, f"sh{si}.npz")) for si in range(shards)]
    Xpix = np.concatenate([p["Xpix"] for p in parts]); Xret = np.concatenate([p["Xret"] for p in parts])
    Xsub = np.concatenate([p["Xsub"] for p in parts]); Y = np.concatenate([p["Y"] for p in parts])
    N = int(Xsub.shape[1] // 3)

    # dim-matched random nonlinear projection of the retina features -> substrate feature dim
    rng = np.random.RandomState(seed + 11)
    W = rng.randn(Xsub.shape[1], Xret.shape[1]).astype(np.float32) / np.sqrt(Xret.shape[1])
    b = rng.uniform(-0.1, 0.1, Xsub.shape[1]).astype(np.float32)
    Xrnd = np.tanh(Xret @ W.T + b)

    reps = {"pixel": Xpix, "retina": Xret, "substrate": Xsub, "randproj": Xrnd}
    res = {}
    for name, X in reps.items():
        L = round(best_linear(X, Y, split), 3)
        M = round(mlp_acc(X, Y, split, seed, device), 3)
        res[name] = dict(linear=L, mlp=M, dim=int(X.shape[1]))
        print(f"  {tag} {name:9s} dim={X.shape[1]:5d} linear={L:.3f} mlp={M:.3f}", flush=True)

    verdict = dict(
        paula_lifts_linear=round(res["substrate"]["linear"] - res["retina"]["linear"], 3),
        paula_vs_random=round(res["substrate"]["linear"] - res["randproj"]["linear"], 3),
        mlp_compensation=round(res["substrate"]["mlp"] - res["substrate"]["linear"], 3),
        retina_over_pixel=round(res["retina"]["linear"] - res["pixel"]["linear"], 3))
    json.dump(dict(exp="controls", tag=tag, arch=arch, dataset=dataset, n_neurons=N,
                   samples=samples, test=test, dwell=dwell, split=split, reps=res,
                   verdict=verdict, secs=round(time.time() - t0), complete=True),
              open(jpath, "w"))
    print(f"[{tag}] VERDICT {json.dumps(verdict)}", flush=True)
    import glob as _g
    for f in _g.glob(os.path.join(sdir, "*.npz")):
        os.remove(f)


def aggregate(out, dataset):
    import glob
    runs = [json.load(open(f)) for f in glob.glob(os.path.join(out, "ctrl_*.json"))]
    if not runs:
        print("no runs"); return
    print("\n=== PAULA ATTRIBUTION LADDER ===")
    hdr = f"{'arch':18s} {'pixel L/M':>12s} {'retina L/M':>12s} {'PAULA L/M':>12s} {'rand L/M':>12s}"
    print(hdr)
    summ = {}
    for d in sorted(runs, key=lambda r: r["arch"]):
        r = d["reps"]
        def lm(k): return f"{r[k]['linear']:.2f}/{r[k]['mlp']:.2f}"
        print(f"{d['arch']:18s} {lm('pixel'):>12s} {lm('retina'):>12s} {lm('substrate'):>12s} {lm('randproj'):>12s}")
        summ[d["arch"]] = d["verdict"]
    print("\nVerdicts (LINEAR-based; the load-bearing ones):")
    for a, v in summ.items():
        print(f"  {a:18s} PAULA lifts linear over retina: {v['paula_lifts_linear']:+.3f} | "
              f"vs random proj: {v['paula_vs_random']:+.3f} | MLP compensation: {v['mlp_compensation']:+.3f}")
    json.dump(summ, open(os.path.join(out, "AGG_controls_summary.json"), "w"), indent=2)
    print(f"\nsaved {out}/AGG_controls_summary.json")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--archs", default="conv,reservoir_random,reservoir_retino")
    p.add_argument("--samples", type=int, default=1500); p.add_argument("--test", type=int, default=600)
    p.add_argument("--dwell", type=int, default=300); p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--split", type=float, default=0.72); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu"); p.add_argument("--par", type=int, default=3)
    p.add_argument("--shards", type=int, default=3, help="parallel shard-procs per arm")
    p.add_argument("--out", default="foveation_results/minibrain/controls")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    archs = a.archs.split(","); t0 = time.time()
    print(f"CONTROLS: archs={archs} samples={a.samples} test={a.test} dwell={a.dwell} out={a.out}", flush=True)
    # split=samples/(samples+test) so 'test' images are the held-out set
    split = a.samples / (a.samples + a.test)
    # arms run sequentially; each arm shards its substrate extraction across all cores
    for arch in archs:
        print(f"--- arm {arch} ({a.shards} shards) ---", flush=True)
        run_arm(arch, a.dataset, a.samples, a.test, a.dwell, a.warmup, a.seed, split,
                a.device, a.out, a.shards)
    print(f"\nall arms done in {time.time()-t0:.0f}s", flush=True)
    aggregate(a.out, a.dataset)


if __name__ == "__main__":
    main()
