"""Main orchestration logic for building activity datasets."""

import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Manager, Pool

from neuron.nn_core import NNCore
from neuron.network_config import NetworkConfig
from neuron.ablation_registry import get_neuron_class_for_ablation
from neuron import setup_neuron_logger
from cli.web_viz.config import WebVizConfig
from cli.web_viz.server import NeuralNetworkWebServer

from snn_classification_realtime.core.config import DatasetConfig
from snn_classification_realtime.activity_dataset_builder.datasets import HDF5TensorRecorder
from snn_classification_realtime.activity_dataset_builder.vision_datasets import (
    load_dataset_by_name,
)
from snn_classification_realtime.activity_dataset_builder.network_utils import (
    infer_layers_from_metadata,
    compute_default_ticks_per_image,
    determine_input_mapping,
    pre_run_compatibility_check,
)
from snn_classification_realtime.core.input_mapping import image_to_signals
from snn_classification_realtime.activity_dataset_builder.worker import process_single_image_worker
from snn_classification_realtime.activity_dataset_builder.resume_state import (
    build_per_label_plan,
    deterministic_shuffle_seed,
    infer_resume_from_row_count,
    load_manifest,
    save_manifest,
)
from snn_classification_realtime.activity_dataset_builder.drive_calibration import (
    estimate_input_drive,
    format_drive_report,
    probe_image_indices,
)

import h5py


def _run_identity(
    *,
    shuffle_seed: int,
    ticks_per_image: int,
    images_per_label: int,
    dataset_name: str,
    num_classes: int,
    network_path: str,
    ablation: str,
    cifar10_color_normalization_factor: float,
    signal_gain: float,
    tick_ms: int,
    use_multiprocessing: bool,
    fresh_run_per_label: bool,
    fresh_run_per_image: bool,
    export_network_states: bool,
    start_web_server: bool,
    dataset_base: str,
) -> dict[str, Any]:
    """Snapshot of CLI-affecting inputs; resume must match this exactly."""
    return {
        "shuffle_seed": int(shuffle_seed),
        "ticks_per_image": int(ticks_per_image),
        "images_per_label": int(images_per_label),
        "dataset_name": str(dataset_name),
        "num_classes": int(num_classes),
        "network_path": os.path.abspath(network_path),
        "ablation": str(ablation),
        "cifar10_color_normalization_factor": float(cifar10_color_normalization_factor),
        "signal_gain": float(signal_gain),
        "tick_ms": int(tick_ms),
        "use_multiprocessing": bool(use_multiprocessing),
        "fresh_run_per_label": bool(fresh_run_per_label),
        "fresh_run_per_image": bool(fresh_run_per_image),
        "export_network_states": bool(export_network_states),
        "start_web_server": bool(start_web_server),
        "dataset_base": str(dataset_base),
    }


def _manifest_skeleton(identity: dict[str, Any]) -> dict[str, Any]:
    m = dict(identity)
    m["next_label_idx"] = 0
    m["committed_global_samples"] = 0
    return m


def _expected_committed_at_label(plan: list[tuple[int, list[int]]], next_label_idx: int) -> int:
    return sum(len(chosen) for lbl, chosen in plan if lbl < next_label_idx)


def _identity_mismatch_errors(
    interrupted: dict[str, Any], current: dict[str, Any]
) -> list[str]:
    """Compare interrupted-run identity (manifest/HDF5) to this process."""
    errs: list[str] = []
    for key, cur_val in current.items():
        if key not in interrupted:
            errs.append(f"interrupted run record missing field {key!r}")
            continue
        prev_val = interrupted[key]
        if isinstance(cur_val, bool):
            ok = bool(prev_val) == cur_val
        elif isinstance(cur_val, float):
            ok = abs(float(prev_val) - cur_val) <= 1e-9
        else:
            ok = prev_val == cur_val
        if not ok:
            errs.append(
                f"{key}: interrupted run has {prev_val!r}, this command has {cur_val!r}"
            )
    return errs


def _identity_from_h5_attrs(hf: Any) -> dict[str, Any] | None:
    """Reconstruct run identity from HDF5 attrs; None if attrs are incomplete."""
    required = (
        "build_shuffle_seed",
        "build_ticks_per_image",
        "build_images_per_label",
        "build_dataset_name",
        "build_num_classes",
        "build_network_path",
        "build_cifar10_color_norm",
        "build_ablation",
        "build_tick_ms",
        "build_use_multiprocessing",
        "build_fresh_run_per_label",
        "build_fresh_run_per_image",
        "build_export_network_states",
        "build_start_web_server",
        "build_dataset_base",
    )
    attrs = hf.attrs
    for k in required:
        if k not in attrs:
            return None
    return {
        "shuffle_seed": int(attrs["build_shuffle_seed"]),
        "ticks_per_image": int(attrs["build_ticks_per_image"]),
        "images_per_label": int(attrs["build_images_per_label"]),
        "dataset_name": str(attrs["build_dataset_name"]),
        "num_classes": int(attrs["build_num_classes"]),
        "network_path": os.path.abspath(str(attrs["build_network_path"])),
        "ablation": str(attrs["build_ablation"]),
        "cifar10_color_normalization_factor": float(attrs["build_cifar10_color_norm"]),
        # Datasets built before the gain knob existed implicitly used gain 1.0
        "signal_gain": float(attrs.get("build_signal_gain", 1.0)),
        "tick_ms": int(attrs["build_tick_ms"]),
        "use_multiprocessing": bool(int(attrs["build_use_multiprocessing"])),
        "fresh_run_per_label": bool(int(attrs["build_fresh_run_per_label"])),
        "fresh_run_per_image": bool(int(attrs["build_fresh_run_per_image"])),
        "export_network_states": bool(int(attrs["build_export_network_states"])),
        "start_web_server": bool(int(attrs["build_start_web_server"])),
        "dataset_base": str(attrs["build_dataset_base"]),
    }


def _manifest_identity(manifest: dict[str, Any]) -> dict[str, Any] | None:
    """Extract comparable identity from manifest; None if incomplete."""
    keys = (
        "shuffle_seed",
        "ticks_per_image",
        "images_per_label",
        "dataset_name",
        "num_classes",
        "network_path",
        "ablation",
        "cifar10_color_normalization_factor",
        "signal_gain",
        "tick_ms",
        "use_multiprocessing",
        "fresh_run_per_label",
        "fresh_run_per_image",
        "export_network_states",
        "start_web_server",
        "dataset_base",
    )
    out: dict[str, Any] = {}
    # Manifests written before the gain knob existed implicitly used gain 1.0
    manifest = {**{"signal_gain": 1.0}, **manifest}
    for k in keys:
        if k not in manifest:
            return None
        v = manifest[k]
        if k == "signal_gain":
            out[k] = float(v)
            continue
        if k == "network_path":
            out[k] = os.path.abspath(str(v))
        elif k == "use_multiprocessing":
            out[k] = bool(v)
        elif k in (
            "fresh_run_per_label",
            "fresh_run_per_image",
            "export_network_states",
            "start_web_server",
        ):
            out[k] = bool(v)
        elif k == "cifar10_color_normalization_factor":
            out[k] = float(v)
        elif k == "shuffle_seed":
            out[k] = int(v)
        elif k in ("ticks_per_image", "images_per_label", "num_classes", "tick_ms"):
            out[k] = int(v)
        else:
            out[k] = v
    return out


def _export_network_state(
    net_path: str,
    neuron_cls: type,
    ds_train: Any,
    img_idx: int,
    input_layer_ids: list[int],
    input_synapses_per_neuron: int,
    ticks_per_image: int,
    img_pos: int,
    actual_label: int,
    state_filename: Path,
    dataset_config: DatasetConfig,
) -> None:
    """Export network state after simulating a single image."""
    network_sim_state = NetworkConfig.load_network_config(
        net_path, neuron_class=neuron_cls
    )
    nn_core_state = NNCore()
    nn_core_state.neural_net = network_sim_state
    setup_neuron_logger("CRITICAL")
    nn_core_state.set_log_level("CRITICAL")

    network_sim_state.reset_simulation()
    nn_core_state.state.current_tick = 0
    network_sim_state.current_tick = 0

    img_tensor, _ = ds_train[img_idx]
    signals = image_to_signals(
        img_tensor,
        input_layer_ids,
        input_synapses_per_neuron,
        network_sim_state,
        dataset_config,
    )

    for _ in range(ticks_per_image):
        nn_core_state.send_batch_signals(signals)
        nn_core_state.do_tick()

    NetworkConfig.save_network_config(
        network_sim_state,
        state_filename,
        metadata={
            "sample_index": img_pos,
            "image_index": int(img_idx),
            "label": int(actual_label),
            "ticks_simulated": ticks_per_image,
            "final_tick": int(network_sim_state.current_tick),
        },
    )


def run_build(config: dict[str, Any]) -> None:
    """Run the activity dataset build workflow. Requires config from CLI."""
    cfg = config
    silent = cfg.get("silent", False)

    if not silent:
        print("--- Build Activity Dataset from Network Simulation ---")

    net_path: str = cfg.get("network_path") or ""
    if not net_path or not os.path.isfile(net_path):
        if not silent:
            print(f"Network file not found: {net_path}")
        return

    try:
        network_sim = NetworkConfig.load_network_config(net_path)
    except Exception as e:
        if not silent:
            print(f"Failed to load network: {e}")
        return

    ablation_name = cfg.get("ablation", "none")
    neuron_cls = get_neuron_class_for_ablation(ablation_name)
    network_sim = NetworkConfig.load_network_config(net_path, neuron_class=neuron_cls)
    if not silent:
        print(f"Using neuron model: ablation={ablation_name}")

    nn_core = NNCore()
    nn_core.neural_net = network_sim
    try:
        setup_neuron_logger("CRITICAL")
    except Exception:
        pass
    try:
        nn_core.set_log_level("CRITICAL")
    except Exception:
        pass

    layers = infer_layers_from_metadata(network_sim)
    if not silent:
        print(f"Detected {len(layers)} layers. Sizes: {[len(layer) for layer in layers]}")

    if cfg.get("start_web_server", False):
        web_config = WebVizConfig()
        if not silent:
            print(
                "Starting web visualization server on "
                f"http://{web_config.host}:{web_config.port} ..."
            )
        web_server = NeuralNetworkWebServer(
            nn_core,
            host=web_config.host,
            port=web_config.port,
            debug=web_config.debug,
        )
        t = threading.Thread(target=web_server.run, daemon=True)
        t.start()
        time.sleep(1.5)

    dataset_name = cfg.get("dataset_name", "mnist")
    cifar10_color_norm = cfg.get("cifar10_color_normalization_factor", 0.165)
    dataset_config = load_dataset_by_name(
        dataset_name,
        cifar10_color_normalization_factor=cifar10_color_norm,
    )
    ds_train = dataset_config.dataset

    if not silent:
        print("Preparation complete.")

    default_ticks = compute_default_ticks_per_image(network_sim, layers)
    _ticks = cfg.get("ticks_per_image")
    ticks_per_image: int = default_ticks if _ticks is None else _ticks
    images_per_label = cfg.get("images_per_label", 5)
    tick_ms = cfg.get("tick_ms", 0)
    use_multiprocessing = cfg.get("use_multiprocessing", False)
    fresh_run_per_label = cfg.get("fresh_run_per_label", True)
    fresh_run_per_image = cfg.get("fresh_run_per_image", False)
    export_network_states = cfg.get("export_network_states", False)

    if fresh_run_per_label and fresh_run_per_image:
        if not silent:
            print(
                "Note: Both per-label and per-image fresh runs enabled. Per-image will take precedence."
            )
        fresh_run_per_label = False

    network_base = os.path.splitext(os.path.basename(net_path))[0]
    dataset_base = cfg.get("dataset_base") or network_base

    input_layer_ids, input_synapses_per_neuron = determine_input_mapping(
        network_sim, layers
    )
    network_sim, input_layer_ids, dataset_config = pre_run_compatibility_check(
        network_sim, input_layer_ids, dataset_config
    )
    nn_core.neural_net = network_sim
    ds_train = dataset_config.dataset
    input_synapses_per_neuron = max(
        1,
        len(network_sim.network.neurons[input_layer_ids[0]].postsynaptic_points),
    )

    label_to_indices: dict[int, list[int]] = {
        i: [] for i in range(dataset_config.num_classes)
    }
    for idx in range(len(ds_train)):
        _img, lbl = ds_train[idx]
        label_to_indices[int(lbl)].append(idx)

    num_classes = dataset_config.num_classes
    ds_name = dataset_config.dataset_name
    resume_dir = cfg.get("resume_dir")
    shuffle_seed_cli = cfg.get("shuffle_seed")
    start_web_server = cfg.get("start_web_server", False)

    # --- Input drive calibration ---
    # Deterministic probe (first N images per label) so auto-gain reproduces
    # the same effective gain on resume.
    signal_gain = float(cfg.get("signal_gain", 1.0))
    auto_gain_target = cfg.get("auto_gain_target")
    probe_per_label = int(cfg.get("calibration_probe_per_label", 2))
    dataset_config.signal_gain = signal_gain
    probe_indices = probe_image_indices(label_to_indices, per_label=probe_per_label)
    target_ratio = float(auto_gain_target) if auto_gain_target is not None else 1.5

    if probe_indices:
        report = estimate_input_drive(
            network_sim,
            input_layer_ids,
            input_synapses_per_neuron,
            dataset_config,
            probe_indices,
            target_ratio=target_ratio,
        )
        if auto_gain_target is not None:
            # Round for a stable value in the manifest/HDF5 identity
            dataset_config.signal_gain = float(f"{report.suggested_gain:.6g}")
            report = estimate_input_drive(
                network_sim,
                input_layer_ids,
                input_synapses_per_neuron,
                dataset_config,
                probe_indices,
                target_ratio=target_ratio,
            )
        if not silent:
            print(format_drive_report(report, dataset_config.signal_gain))
    signal_gain = dataset_config.signal_gain

    shuffle_seed_effective = (
        int(shuffle_seed_cli)
        if shuffle_seed_cli is not None
        else deterministic_shuffle_seed(
            net_path,
            ds_name,
            images_per_label,
            dataset_base,
            ticks_per_image,
            ablation_name,
            cifar10_color_norm,
        )
    )

    current_identity = _run_identity(
        shuffle_seed=shuffle_seed_effective,
        ticks_per_image=ticks_per_image,
        images_per_label=images_per_label,
        dataset_name=ds_name,
        num_classes=num_classes,
        network_path=net_path,
        ablation=ablation_name,
        cifar10_color_normalization_factor=cifar10_color_norm,
        signal_gain=signal_gain,
        tick_ms=tick_ms,
        use_multiprocessing=use_multiprocessing,
        fresh_run_per_label=fresh_run_per_label,
        fresh_run_per_image=fresh_run_per_image,
        export_network_states=export_network_states,
        start_web_server=start_web_server,
        dataset_base=dataset_base,
    )

    if resume_dir:
        dataset_dir = os.path.abspath(resume_dir)
        if not os.path.isdir(dataset_dir):
            if not silent:
                print(f"Resume directory not found: {dataset_dir}")
            return
        h5_path = os.path.join(dataset_dir, "activity_dataset.h5")
        if not os.path.isfile(h5_path):
            if not silent:
                print(f"Cannot resume: missing {h5_path}")
            return

        manifest = load_manifest(dataset_dir)
        interrupted: dict[str, Any] | None = None

        with h5py.File(h5_path, "r") as hf:
            n_rows = int(hf["u"].shape[0])
            interrupted_h5 = _identity_from_h5_attrs(hf)

        if manifest:
            interrupted = _manifest_identity(manifest)
            if interrupted is None:
                if not silent:
                    print(
                        "Resume refused: activity_build_state.json is missing required "
                        "fields (tick_ms, use_multiprocessing, …). Delete it only if "
                        "activity_dataset.h5 has the full set of build_* attributes."
                    )
                return
            errs = _identity_mismatch_errors(interrupted, current_identity)
            if errs:
                if not silent:
                    print("Resume refused: this command does not match the interrupted run:")
                    for e in errs:
                        print(f"  - {e}")
                return
            shuffle_seed = int(interrupted["shuffle_seed"])
            resume_next_label_idx = int(manifest["next_label_idx"])
            committed = int(manifest["committed_global_samples"])
        else:
            interrupted = interrupted_h5
            if interrupted is None:
                if not silent:
                    print(
                        "Resume refused: missing activity_build_state.json and HDF5 does not "
                        "contain the full set of build_* attributes (need a dataset started with "
                        "the current tool), or pass a matching manifest."
                    )
                return
            errs = _identity_mismatch_errors(interrupted, current_identity)
            if errs:
                if not silent:
                    print("Resume refused: this command does not match the interrupted run:")
                    for e in errs:
                        print(f"  - {e}")
                return
            shuffle_seed = int(interrupted["shuffle_seed"])
            plan_infer = build_per_label_plan(
                num_classes, label_to_indices, images_per_label, shuffle_seed
            )
            committed, resume_next_label_idx = infer_resume_from_row_count(
                n_rows, plan_infer
            )
            if not silent:
                print(
                    f"No manifest; inferred checkpoint from HDF5 rows={n_rows} -> "
                    f"committed={committed}, resume_label={resume_next_label_idx}. "
                    "Writing activity_build_state.json."
                )
            manifest = {**interrupted, "next_label_idx": resume_next_label_idx, "committed_global_samples": committed}
            save_manifest(dataset_dir, manifest)

        plan = build_per_label_plan(
            num_classes, label_to_indices, images_per_label, shuffle_seed
        )
        expected = _expected_committed_at_label(plan, resume_next_label_idx)
        if committed != expected:
            if not silent:
                print(
                    f"Warning: manifest committed_global_samples={committed} "
                    f"!= expected {expected} for next_label_idx={resume_next_label_idx}; "
                    "using manifest value for truncate."
                )
        if committed > n_rows:
            if not silent:
                print(
                    f"Cannot resume: manifest committed ({committed}) exceeds HDF5 rows ({n_rows})."
                )
            return

        recorder = HDF5TensorRecorder(
            dataset_dir, network_sim.network, ablation_name=ablation_name
        )
        try:
            recorder.open_existing_for_resume(
                truncate_to=committed,
                run_identity=current_identity,
            )
        except ValueError as e:
            if not silent:
                print(str(e))
            return
        if not silent:
            print(
                f"Resuming into {dataset_dir} (shuffle_seed={shuffle_seed}, "
                f"next_label={resume_next_label_idx}, committed={committed})."
            )
    else:
        ts = cfg.get("output_suffix") or str(int(time.time()))
        output_base = cfg.get("output_dir", "activity_datasets")
        dataset_dir = os.path.join(output_base, f"{dataset_base}_{ds_name}_{ts}")
        os.makedirs(dataset_dir, exist_ok=True)
        shuffle_seed = int(current_identity["shuffle_seed"])
        plan = build_per_label_plan(
            num_classes, label_to_indices, images_per_label, shuffle_seed
        )
        manifest = _manifest_skeleton(current_identity)
        save_manifest(dataset_dir, manifest)
        build_meta = dict(current_identity)
        recorder = HDF5TensorRecorder(
            dataset_dir,
            network_sim.network,
            ablation_name=ablation_name,
            build_meta=build_meta,
        )
        resume_next_label_idx = 0
        if not silent:
            print(
                f"Recording to HDF5: {dataset_dir}/activity_dataset.h5 "
                f"(shuffle_seed={shuffle_seed})"
            )

    if resume_next_label_idx >= num_classes:
        if not silent:
            print("Nothing to do: dataset build already finished for all labels.")
        recorder.close()
        return

    network_state_dir: Path | None = None
    if export_network_states:
        network_state_dir = Path("network_state") / Path(dataset_dir).name
        network_state_dir.mkdir(parents=True, exist_ok=True)
        if not silent:
            print(f"Network states will be exported to: {network_state_dir}")

    if not silent:
        print("Starting collection loop ...")

    labels_pbar = tqdm(plan, desc="Labels", leave=False, disable=silent)

    for label_idx, chosen in labels_pbar:
        labels_pbar.set_description(f"Label {label_idx}")

        if label_idx < resume_next_label_idx:
            continue

        if not chosen:
            manifest["committed_global_samples"] = _expected_committed_at_label(
                plan, label_idx + 1
            )
            manifest["next_label_idx"] = label_idx + 1
            save_manifest(dataset_dir, manifest)
            continue

        samples_pbar = tqdm(
            total=len(chosen),
            desc=f"Label {label_idx} samples",
            position=1,
            leave=False,
            disable=silent,
        )

        global_sample_counter = _expected_committed_at_label(plan, label_idx)

        label_tasks = []
        for img_pos, img_idx in enumerate(chosen):
            task = (
                img_idx,
                label_idx,
                ticks_per_image,
                net_path,
                input_layer_ids,
                input_synapses_per_neuron,
                layers,
                tick_ms,
                ds_train,
                dataset_config,
                ablation_name,
            )
            label_tasks.append((task, label_idx, img_pos, global_sample_counter))
            global_sample_counter += 1

        if use_multiprocessing and len(label_tasks) > 1:
            num_processes = min(mp.cpu_count(), len(label_tasks))
            manager = Manager()
            progress_queue = manager.Queue()

            with Pool(processes=num_processes) as pool:
                process_bars: dict[int, Any] = {}
                process_progress: dict[int, dict[str, Any]] = {}
                results = []
                task_idx = 0

                for task_data in label_tasks:
                    task, label, img_pos, global_sample_idx = task_data
                    process_id = task_idx % num_processes
                    updated_task = task + (process_id, progress_queue)
                    result = pool.apply_async(
                        process_single_image_worker, (updated_task,)
                    )
                    results.append(
                        (result, label, img_pos, global_sample_idx, task[0], process_id)
                    )
                    task_idx += 1

                completed_count = 0
                last_update = {proc_id: 0 for proc_id in range(num_processes)}

                while completed_count < len(results):
                    while not progress_queue.empty():
                        try:
                            progress_data = progress_queue.get_nowait()
                            process_id = progress_data["process_id"]
                            process_progress[process_id] = progress_data
                            if process_id not in process_bars:
                                img_idx = progress_data["img_idx"]
                                label = progress_data["label"]
                                total_ticks = progress_data["total_ticks"]
                                process_bars[process_id] = tqdm(
                                    total=total_ticks,
                                    desc=f"P{process_id + 1} L{label} I{img_idx}",
                                    leave=False,
                                    unit="ticks",
                                    disable=silent,
                                )
                        except Exception:
                            break

                    for proc_id in list(process_bars.keys()):
                        if proc_id in process_progress:
                            progress_data = process_progress[proc_id]
                            current_tick = progress_data["current_tick"]
                            total_ticks = progress_data["total_ticks"]
                            progress_increment = current_tick - last_update[proc_id]
                            if progress_increment > 0:
                                process_bars[proc_id].update(progress_increment)
                                last_update[proc_id] = current_tick
                            if progress_data.get("completed", False):
                                process_bars[proc_id].close()
                                del process_bars[proc_id]
                                if proc_id in process_progress:
                                    del process_progress[proc_id]

                    for result_info in results:
                        (
                            result_future,
                            label,
                            img_pos,
                            global_sample_idx,
                            img_idx,
                            process_id,
                        ) = result_info
                        if result_future.ready() and not hasattr(
                            result_future, "_collected"
                        ):
                            result = result_future.get()
                            result_future._collected = True
                            completed_count += 1
                            samples_pbar.update(1)

                            (
                                _result_img_idx,
                                actual_label,
                                u_buf,
                                t_ref_buf,
                                fr_buf,
                                spikes,
                            ) = result

                            spike_arr = (
                                np.array(spikes, dtype=np.int32).flatten()
                                if spikes
                                else np.zeros((0,), dtype=np.int32)
                            )
                            recorder.save_sample_from_buffers(
                                global_sample_idx,
                                int(actual_label),
                                u_buf,
                                t_ref_buf,
                                fr_buf,
                                spike_arr,
                            )
                            # ApplyResult keeps return value on _value; release so RAM
                            # stays flat across many samples within one label.
                            result_future._value = None
                            del result

                            if export_network_states and network_state_dir is not None:
                                label_dir = network_state_dir / f"label_{label}"
                                label_dir.mkdir(parents=True, exist_ok=True)
                                state_filename = (
                                    label_dir / f"sample_{img_pos}_img{img_idx}.json"
                                )
                                _export_network_state(
                                    net_path,
                                    neuron_cls,
                                    ds_train,
                                    img_idx,
                                    input_layer_ids,
                                    input_synapses_per_neuron,
                                    ticks_per_image,
                                    img_pos,
                                    actual_label,
                                    state_filename,
                                    dataset_config,
                                )

                            if process_id in process_progress:
                                del process_progress[process_id]
                            last_update[process_id] = 0

                    time.sleep(0.05)

                for bar in process_bars.values():
                    bar.close()

                if not silent:
                    tqdm.write(
                        f"Label {label_idx} processing complete - {completed_count} samples processed"
                    )

        else:
            for task_data in label_tasks:
                task, label, img_pos, global_sample_idx = task_data
                updated_task = task + (None, None)
                result = process_single_image_worker(updated_task)

                (
                    _result_img_idx,
                    actual_label,
                    u_buf,
                    t_ref_buf,
                    fr_buf,
                    spikes,
                ) = result

                spike_arr = (
                    np.array(spikes, dtype=np.int32).flatten()
                    if spikes
                    else np.zeros((0,), dtype=np.int32)
                )
                recorder.save_sample_from_buffers(
                    global_sample_idx,
                    int(actual_label),
                    u_buf,
                    t_ref_buf,
                    fr_buf,
                    spike_arr,
                )

                if export_network_states and network_state_dir is not None:
                    label_dir = network_state_dir / f"label_{label}"
                    label_dir.mkdir(parents=True, exist_ok=True)
                    state_filename = label_dir / f"sample_{img_pos}_img{img_idx}.json"
                    _export_network_state(
                        net_path,
                        neuron_cls,
                        ds_train,
                        task[0],
                        input_layer_ids,
                        input_synapses_per_neuron,
                        ticks_per_image,
                        img_pos,
                        actual_label,
                        state_filename,
                        dataset_config,
                    )

                samples_pbar.update(1)
                del result

        samples_pbar.close()

        manifest["committed_global_samples"] = _expected_committed_at_label(
            plan, label_idx + 1
        )
        manifest["next_label_idx"] = label_idx + 1
        save_manifest(dataset_dir, manifest)

    labels_pbar.close()

    if not silent:
        print("All data saved progressively during processing")

    sample_count = sum(
        min(images_per_label, len(indices))
        for indices in label_to_indices.values()
    )
    if not silent:
        print(
            f"HDF5 dataset complete -> {dataset_dir}/activity_dataset.h5 ({sample_count} compressed samples)"
        )
    recorder.close()

    if not silent:
        print("Done.")
