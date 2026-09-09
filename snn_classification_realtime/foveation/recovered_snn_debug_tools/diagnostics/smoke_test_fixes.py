"""Smoke tests for the 4 CIFAR fixes + drive calibration."""

import sys

sys.path.insert(0, "/Users/arterialist/Projects/agi-research/neuron-model")

import numpy as np
import torch

from snn_classification_realtime.network_builder import build_network
from snn_classification_realtime.activity_dataset_builder.vision_datasets import (
    load_dataset_by_name,
)
from snn_classification_realtime.core.network_utils import (
    infer_layers_from_metadata,
    determine_input_mapping,
)
from snn_classification_realtime.core.input_mapping import image_to_signals
from snn_classification_realtime.activity_dataset_builder.drive_calibration import (
    estimate_input_drive,
    format_drive_report,
    probe_image_indices,
)

ok = True


def check(name, cond, detail=""):
    global ok
    status = "PASS" if cond else "FAIL"
    if not cond:
        ok = False
    print(f"[{status}] {name} {detail}")


# ---------- Fix 1: cifar10_grayscale is real 1-channel luminance ----------
cfg_gray = load_dataset_by_name("cifar10_grayscale", train=True)
img0, _ = cfg_gray.dataset[0]
check(
    "cifar10_grayscale shape",
    tuple(img0.shape) == (1, 32, 32),
    f"shape={tuple(img0.shape)}, vec={cfg_gray.image_vector_size}",
)

# CNN grayscale-averaging fallback: in_channels=1 network fed a 3-channel image
net_gray_cnn = build_network(
    {
        "dataset": "cifar10_grayscale",
        "layers": [
            {"type": "conv", "kernel_size": 3, "stride": 2, "filters": 1},
            {"type": "dense", "size": 10},
        ],
    }
)
layers = infer_layers_from_metadata(net_gray_cnn)
in_ids, syn_per = determine_input_mapping(net_gray_cnn, layers)
cfg_rgb = load_dataset_by_name("cifar10", train=True)
rgb_img, _ = cfg_rgb.dataset[0]  # 3x32x32
sig_rgb_in = image_to_signals(rgb_img, in_ids, syn_per, net_gray_cnn, cfg_rgb)
# expected strength at any synapse = mean over channels, not red channel
first_nid = in_ids[0]
n0 = net_gray_cnn.network.neurons[first_nid]
m = n0.metadata
y0, x0 = int(m["y"]) * int(m["stride"]), int(m["x"]) * int(m["stride"])
expected = (float(rgb_img[:, y0, x0].mean()) + 1.0) * 0.5
got = [s for (nid, sid, s) in sig_rgb_in if nid == first_nid and sid == 0][0]
red_only = (float(rgb_img[0, y0, x0]) + 1.0) * 0.5
check(
    "CNN grayscale averages RGB (not red channel)",
    abs(got - expected) < 1e-6,
    f"got={got:.4f} expected_mean={expected:.4f} red_only={red_only:.4f}",
)

# ---------- Fix 2: colored dense mapping keeps all pixels ----------
net_color_dense = build_network(
    {
        "dataset": "cifar10_color",
        "input_size": 100,
        "layers": [{"type": "dense", "size": 32}, {"type": "dense", "size": 10}],
    }
)
layers_c = infer_layers_from_metadata(net_color_dense)
in_ids_c, syn_per_c = determine_input_mapping(net_color_dense, layers_c)
cfg_color = load_dataset_by_name("cifar10_color", train=True)
cimg, _ = cfg_color.dataset[0]
sigs_c = image_to_signals(cimg, in_ids_c, syn_per_c, net_color_dense, cfg_color)
check(
    "colored dense mapping covers all 3072 values",
    len(sigs_c) == 3072,
    f"signals={len(sigs_c)} (was 3000 before fix), syn_per={syn_per_c}",
)
check(
    "colored dense synapse ids within bounds",
    all(sid < syn_per_c for (_, sid, _) in sigs_c),
)

# ---------- Fix 3a: rgb_separate dense capacity ----------
net_sep = build_network(
    {
        "dataset": "cifar10_color",
        "input_size": 256,
        "rgb_separate_neurons": True,
        "layers": [{"type": "dense", "size": 32}, {"type": "dense", "size": 10}],
    }
)
layers_s = infer_layers_from_metadata(net_sep)
in_ids_s, syn_per_s = determine_input_mapping(net_sep, layers_s)
capacity = len(in_ids_s) * syn_per_s
check(
    "rgb_separate capacity >= 3072",
    capacity >= 3072,
    f"neurons={len(in_ids_s)}, syn_per={syn_per_s}, capacity={capacity}",
)
sigs_s = image_to_signals(cimg, in_ids_s, syn_per_s, net_sep, cfg_color)
check(
    "rgb_separate mapping covers all 3072 values",
    len(sigs_s) == 3072,
    f"signals={len(sigs_s)}",
)

# ---------- Fix 3b: conv-after-rgb-separate uses all filters ----------
net_sep_conv = build_network(
    {
        "dataset": "cifar10_color",
        "rgb_separate_neurons": True,
        "layers": [
            {"type": "conv", "kernel_size": 3, "stride": 2, "filters": 2},
            {"type": "conv", "kernel_size": 3, "stride": 2, "filters": 2},
            {"type": "dense", "size": 10},
        ],
    }
)
layers_sc = infer_layers_from_metadata(net_sep_conv)
l0_ids, l1_ids = set(layers_sc[0]), set(layers_sc[1])
nrn = net_sep_conv.network
l0_filters_used = {
    int(nrn.neurons[c[0]].metadata.get("filter", -1))
    for c in nrn.connections
    if c[0] in l0_ids and c[2] in l1_ids
}
l1_in_ch = int(nrn.neurons[next(iter(l1_ids))].metadata.get("in_channels", -1))
check(
    "conv-after-rgb-separate connects all layer-0 filters",
    l0_filters_used == {0, 1},
    f"filters_used={sorted(l0_filters_used)}",
)
check(
    "conv-after-rgb-separate in_channels = filters*3",
    l1_in_ch == 6,
    f"in_channels={l1_in_ch}",
)
max_syn_id = max(
    c[3] for c in nrn.connections if c[0] in l0_ids and c[2] in l1_ids
)
l1_syns = len(nrn.neurons[next(iter(l1_ids))].postsynaptic_points)
check(
    "layer-1 synapse ids within allocated range",
    max_syn_id < l1_syns,
    f"max_syn_id={max_syn_id}, allocated={l1_syns}",
)

# ---------- Fix 4: avg_t_ref uses full time series ----------
from snn_classification_realtime.activity_preparer.hdf5_features import (
    extract_features_from_hdf5_sample,
)

t = 5
n = 4
sample = {
    "u": torch.rand(t, n),
    "t_ref": torch.arange(t * n, dtype=torch.float32).reshape(t, n),
    "fr": torch.rand(t, n),
    "spikes": torch.zeros((0, 2), dtype=torch.int32),
    "label": 0,
    "neuron_ids": list(range(n)),
}
feat = extract_features_from_hdf5_sample(sample, ["avg_t_ref"])
check(
    "avg_t_ref keeps temporal variation",
    torch.equal(feat, sample["t_ref"]) and not torch.equal(feat[0], feat[1]),
    f"shape={tuple(feat.shape)}",
)

# ---------- Calibration: MNIST vs CIFAR dense, gain knob ----------
def drive_for(dataset_name, input_size=100, gain=1.0, auto_target=None):
    net = build_network(
        {
            "dataset": dataset_name,
            "input_size": input_size,
            "layers": [{"type": "dense", "size": 32}, {"type": "dense", "size": 10}],
        }
    )
    lyrs = infer_layers_from_metadata(net)
    ids, sp = determine_input_mapping(net, lyrs)
    dcfg = load_dataset_by_name(dataset_name, train=True)
    dcfg.signal_gain = gain
    label_to_indices = {i: [] for i in range(dcfg.num_classes)}
    for idx in range(200):
        _, lbl = dcfg.dataset[idx]
        label_to_indices[int(lbl)].append(idx)
    probes = probe_image_indices(label_to_indices, per_label=2)
    rep = estimate_input_drive(net, ids, sp, dcfg, probes, target_ratio=1.5)
    if auto_target is not None:
        dcfg.signal_gain = float(f"{rep.suggested_gain:.6g}")
        rep = estimate_input_drive(net, ids, sp, dcfg, probes, target_ratio=1.5)
    return rep, dcfg.signal_gain


# Dense-first layer metadata fix: input layer must contain exactly input_size neurons
layers_dense = infer_layers_from_metadata(net_color_dense)
check(
    "dense-first input layer isolated from first hidden layer",
    len(layers_dense[0]) == 100 and len(layers_dense[1]) == 32,
    f"layer_sizes={[len(l) for l in layers_dense]}",
)

# Direct JSON builder parity for the same fixes
from snn_classification_realtime.network_builder_direct import (
    build_network_config_direct,
)

cfg_direct = build_network_config_direct(
    {
        "dataset": "cifar10_color",
        "rgb_separate_neurons": True,
        "layers": [
            {"type": "conv", "kernel_size": 3, "stride": 2, "filters": 2},
            {"type": "conv", "kernel_size": 3, "stride": 2, "filters": 2},
            {"type": "dense", "size": 10},
        ],
    }
)
by_id = {n["id"]: n for n in cfg_direct["neurons"]}
l0d = {n["id"] for n in cfg_direct["neurons"] if n["metadata"]["layer"] == 0}
l1d = {n["id"] for n in cfg_direct["neurons"] if n["metadata"]["layer"] == 1}
filters_used_d = {
    by_id[c["source_neuron"]]["metadata"].get("filter", -1)
    for c in cfg_direct["connections"]
    if c["source_neuron"] in l0d and c["target_neuron"] in l1d
}
l1_in_ch_d = by_id[next(iter(l1d))]["metadata"].get("in_channels", -1)
check(
    "direct builder: conv-after-rgb-separate all filters + in_channels",
    filters_used_d == {0, 1} and l1_in_ch_d == 6,
    f"filters={sorted(filters_used_d)}, in_channels={l1_in_ch_d}",
)

cfg_direct_dense = build_network_config_direct(
    {
        "dataset": "mnist",
        "input_size": 100,
        "layers": [{"type": "dense", "size": 32}, {"type": "dense", "size": 10}],
    }
)
sizes_d: dict[int, int] = {}
for n in cfg_direct_dense["neurons"]:
    sizes_d[n["metadata"]["layer"]] = sizes_d.get(n["metadata"]["layer"], 0) + 1
check(
    "direct builder: dense-first layer indices",
    sizes_d == {0: 100, 1: 32, 2: 10},
    f"sizes={sizes_d}",
)

rep_mnist, _ = drive_for("mnist")
print()
print("MNIST dense (input_size=100):")
print(format_drive_report(rep_mnist, 1.0))
print()
rep_cifar, _ = drive_for("cifar10")
print("CIFAR10 RGB dense (input_size=100):")
print(format_drive_report(rep_cifar, 1.0))
print()
rep_cifar_auto, auto_gain = drive_for("cifar10", auto_target=1.5)
print(f"CIFAR10 RGB dense with --auto-gain 1.5 (effective gain {auto_gain}):")
print(format_drive_report(rep_cifar_auto, auto_gain))
print()
check(
    "CIFAR saturates without gain, MNIST does not",
    rep_cifar.ratio_mean > 2.5 > rep_mnist.ratio_mean,
    f"cifar_ratio={rep_cifar.ratio_mean:.2f}, mnist_ratio={rep_mnist.ratio_mean:.2f}",
)
check(
    "auto-gain brings mean ratio to ~1.5",
    abs(rep_cifar_auto.ratio_mean - 1.5) < 0.05,
    f"ratio_after={rep_cifar_auto.ratio_mean:.3f}",
)
# Gain must actually scale the emitted signals
cfg_rgb.signal_gain = 0.25
sigs_g = image_to_signals(rgb_img, in_ids_c, syn_per_c, net_color_dense, cfg_rgb)
cfg_rgb.signal_gain = 1.0
sigs_1 = image_to_signals(rgb_img, in_ids_c, syn_per_c, net_color_dense, cfg_rgb)
ratio = sum(s for *_, s in sigs_g) / max(sum(s for *_, s in sigs_1), 1e-9)
check("signal_gain scales strengths linearly", abs(ratio - 0.25) < 1e-6, f"ratio={ratio:.4f}")

print()
print("ALL PASS" if ok else "SOME FAILURES")
sys.exit(0 if ok else 1)
