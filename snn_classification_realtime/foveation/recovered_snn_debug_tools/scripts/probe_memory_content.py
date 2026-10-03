"""What does 'more memory' mean? Decompose the learned weight change Delta_w into
magnitude vs direction, sparse vs dense. (measurement only, no model edits)

For each connectivity and each of K images: build a FRESH net (same seed -> same
init weights w0), hold the image constant for T ticks so weights settle, take
w_img = efficacy_vector(). Delta = w_img - w0 is what that image WROTE.

Discriminating question:
  - if Delta_i for different images are PARALLEL (high pairwise cosine) -> every
    image pushes weights the same way = 'more of the same signal' (magnitude/gain).
  - if Delta_i are DISTINCT (low cosine) -> each image writes its own pattern =
    content-specific 'detail'.
  Compare sparse vs dense: is dense more content-specific, or just larger |Delta|?
"""
import tempfile
from pathlib import Path

import numpy as np
from snn_classification_realtime.activity_dataset_builder.vision_datasets import load_dataset_by_name
from snn_classification_realtime.activity_dataset_builder.drive_calibration import estimate_input_drive
from snn_classification_realtime.foveation.experiment_tref_stability import build_mlp_json, run_arm
from snn_classification_realtime.foveation.perception import PerceptionNetwork

def main():
    with tempfile.TemporaryDirectory(prefix="paula-memory-content-") as temp_dir:
        network_path = str(Path(temp_dir) / "network.json")
        T = 2500
        K = 10                       # one image per CIFAR class
        conns = [0.1, 0.5]
        dc = load_dataset_by_name("cifar10_grayscale", train=True)
        ds = dc.dataset
        # pick one image of each of the first 10 distinct labels
        by_label, idxs = {}, []
        i = 0
        while len(idxs) < K and i < len(ds):
            _, y = ds[i]
            y = int(y)
            if y not in by_label:
                by_label[y] = i; idxs.append(i)
            i += 1
        imgs = [ds[j][0] for j in idxs]

        def cos(a, b):
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            return float(a @ b / (na * nb)) if na > 1e-9 and nb > 1e-9 else 0.0

        for p in conns:
            build_mlp_json(network_path, channels=1, size=32, input_size=64, hidden=3,
                           sizes=64, connectivity=p, seed=0)
            # calibrate gain once (input layer identical across conn)
            perc0 = PerceptionNetwork(network_path, dc)
            rep = estimate_input_drive(perc0.sim, perc0.input_layer_ids,
                                       perc0.input_synapses_per_neuron, dc, list(range(10)), 1.8)
            dc.signal_gain = float(rep.suggested_gain)
            w0 = perc0.efficacy_vector().copy()            # init weights (same seed -> same for all)
            n_syn = w0.size

            W = np.zeros((K, n_syn), np.float32)
            for k, im in enumerate(imgs):
                perc = PerceptionNetwork(network_path, dc)
                run_arm(perc, perc.patch_to_signals(im), T, 0.0)
                W[k] = perc.efficacy_vector()
            D = W - w0[None, :]                             # learned change per image

            # magnitude
            mag = np.linalg.norm(D, axis=1)                 # ||Delta|| per image
            meaneff = W.mean(axis=1)                        # mean weight per image
            grown = (D > 0.05).mean(axis=1)                 # fraction of synapses potentiated
            # direction / content-specificity
            off = [cos(D[a], D[b]) for a in range(K) for b in range(a + 1, K)]
            off = np.array(off)
            # same-image reproducibility control: re-adapt image 0 with a 2nd fresh net
            perc = PerceptionNetwork(network_path, dc)
            run_arm(perc, perc.patch_to_signals(imgs[0]), T, 0.0)
            d0b = perc.efficacy_vector() - w0
            repro_cos = cos(D[0], d0b)                      # should be ~1 (deterministic)
            # how much of Delta is the COMMON (shared) component vs image-specific?
            Dmean = D.mean(axis=0)
            shared_frac = np.mean([np.linalg.norm(np.dot(D[k], Dmean) / (np.dot(Dmean, Dmean) + 1e-9) * Dmean)
                                   / (np.linalg.norm(D[k]) + 1e-9) for k in range(K)])

            print(f"\n=== connectivity {p} ===")
            print(f"  mean |Delta| (magnitude of learned change) : {mag.mean():.3f}  (per-image spread {mag.std():.3f})")
            print(f"  mean weight after adapt                    : {meaneff.mean():.3f}")
            print(f"  fraction of synapses potentiated (>0.05)   : {grown.mean():.3f}")
            print(f"  pairwise cosine of Delta across images     : {off.mean():.3f}  (min {off.min():.3f}, max {off.max():.3f})")
            print(f"     -> ~1.0 = all images write the SAME direction (magnitude/gain);  ~0 = content-specific")
            print(f"  same-image reproducibility cosine (control): {repro_cos:.3f}  (expect ~1.0)")
            print(f"  shared-component fraction of each Delta     : {shared_frac:.3f}  (1=pure common gain, <1=image-specific part)")


if __name__ == "__main__":
    main()
