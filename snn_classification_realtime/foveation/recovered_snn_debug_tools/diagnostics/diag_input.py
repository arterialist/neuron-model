"""Why does the input layer never fire? Report its S vs its firing threshold."""
import numpy as np
from snn_classification_realtime.activity_dataset_builder.vision_datasets import (
    load_dataset_by_name,
)
from snn_classification_realtime.foveation.minibrain.core import MiniBrainConfig, MiniBrain

ds_cfg = load_dataset_by_name("cifar10_grayscale", train=True)
ds = ds_cfg.dataset
cfg = MiniBrainConfig(dataset_name="cifar10_grayscale", tonic_drive=0.15, seed=0)
brain = MiniBrain(cfg, ds_cfg)
lay = brain.sub.layer_of_pos
neurons = list(brain.sub.sim.network.neurons.values())
input_L = brain._input_layer
sig = brain.sub.patch_to_signals(brain.retina.render(ds[0][0]))

brain.sub.reset()
for t in range(120):
    brain.sub.step(sig + brain._tonic)          # continuous image
S = np.array([n.S for n in neurons])
rbase = np.array([n.params.r_base for n in neurons])
r_now = np.array([float(np.atleast_1d(n.r)[0]) for n in neurons])
inp = (lay == input_L)
print(f"input layer ({inp.sum()} neurons) under CONTINUOUS image:")
print(f"  S:      mean {S[inp].mean():.3f}  max {S[inp].max():.3f}  min {S[inp].min():.3f}")
print(f"  r_base: mean {rbase[inp].mean():.3f}  (firing needs S >= r)")
print(f"  r_now:  mean {r_now[inp].mean():.3f}")
print(f"  fraction of input neurons with S >= r_now: {(S[inp] >= r_now[inp]).mean():.3f}")
print(f"\nsaturation ceiling S_max ~ 1/(1-exp(-c/lambda)) = "
      f"{1/(1-np.exp(-neurons[0].params.c/neurons[0].params.lambda_param)):.3f}")
# what gain would be needed for input S to reach r?  ratio of r to current S
print(f"\ncurrent signal_gain = {ds_cfg.signal_gain:.4f}")
print(f"to reach r, input S must ~3-4x -> gain would need to rise similarly, "
      f"but that re-saturates. Better: lower input r_base via calibration.")
