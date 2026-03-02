"""Main orchestration logic for building activity datasets."""

import os
import random
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
from snn_classification_realtime.activity_dataset_builder.signals import compute_signal_grid, save_signal_plot
from snn_classification_realtime.activity_dataset_builder.worker import process_single_image_worker


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
        if not silent:
            print("Starting web visualization server on http://127.0.0.1:5555 ...")
        web_server = NeuralNetworkWebServer(nn_core, host="127.0.0.1", port=5555)
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

    ts = int(time.time())
    output_base = cfg.get("output_dir", "activity_datasets")
    dataset_dir = os.path.join(output_base, f"{dataset_base}_{dataset_config.dataset_name}_{ts}")

    recorder = HDF5TensorRecorder(
        dataset_dir, network_sim.network, ablation_name=ablation_name
    )
    if not silent:
        print(f"Recording to HDF5: {dataset_dir}/activity_dataset.h5")

    network_state_dir: Path | None = None
    if export_network_states:
        network_state_dir = Path("network_state") / f"{dataset_base}_{dataset_config.dataset_name}_{ts}"
        network_state_dir.mkdir(parents=True, exist_ok=True)
        if not silent:
            print(f"Network states will be exported to: {network_state_dir}")

    if not silent:
        print("Starting collection loop ...")
    global_sample_counter = 0
    processed_results: list[tuple[Any, int, int, int, int]] = []
    num_classes = dataset_config.num_classes

    labels_pbar = tqdm(range(num_classes), desc="Labels", leave=False, disable=silent)

    for label_idx in range(num_classes):
        labels_pbar.update(1)
        labels_pbar.set_description(f"Label {label_idx}")

        indices = label_to_indices.get(label_idx, [])
        if not indices:
            continue

        chosen = random.sample(indices, min(images_per_label, len(indices)))

        samples_pbar = tqdm(
            total=len(chosen),
            desc=f"Label {label_idx} samples",
            position=1,
            leave=False,
            disable=silent,
        )

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

                label_processed_results = []
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
                            label_processed_results.append(
                                (result, label, img_pos, global_sample_idx, img_idx)
                            )
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
                        f"Label {label_idx} processing complete - {len(label_processed_results)} samples processed"
                    )
                processed_results.extend(label_processed_results)

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

                processed_results.append(
                    (result, label, img_pos, global_sample_idx, task[0])
                )
                samples_pbar.update(1)

        samples_pbar.close()

    labels_pbar.close()

    plotted_labels: set[int] = set()
    for _, label, _, _, img_idx in processed_results:
        if label not in plotted_labels:
            img_tensor, _ = ds_train[img_idx]
            grid = compute_signal_grid(img_tensor)
            save_signal_plot(grid, label, int(img_idx))
            plotted_labels.add(label)

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
