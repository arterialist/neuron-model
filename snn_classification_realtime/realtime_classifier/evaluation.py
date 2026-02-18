"""Evaluation mode: automated testing with progress tracking and JSONL streaming."""

from __future__ import annotations

import json
import os
import time
from typing import Any

import torch
import torch.nn as nn
from tqdm import tqdm

from snn_classification_realtime.realtime_classifier.features import (
    collect_features_consistently,
)
from snn_classification_realtime.realtime_classifier.scaling import (
    apply_scaling_to_snapshot,
)
from snn_classification_realtime.realtime_classifier.input_mapping import (
    image_to_signals,
)
from snn_classification_realtime.realtime_classifier.network_utils import (
    infer_layers_from_metadata,
    determine_input_mapping,
)
from snn_classification_realtime.core.config import DatasetConfig


def run_evaluation(
    *,
    dataset_config: DatasetConfig,
    network_config_path: str,
    neuron_cls: type,
    nn_core: Any,
    snn_model: nn.Module,
    scaler_state: dict[str, Any],
    feature_types: list[str],
    device: torch.device,
    args: Any,
) -> None:
    """Run evaluation mode with automatic testing and progress tracking."""
    softmax = nn.Softmax(dim=1)
    activity_buffer: list[list[float]] = []
    num_classes = dataset_config.num_classes
    dataset = dataset_config.dataset

    print(f"Starting evaluation mode with {args.eval_samples} samples...")

    eval_results: list[dict[str, Any]] = []
    label_errors = {i: 0 for i in range(num_classes)}
    label_errors_second = {i: 0 for i in range(num_classes)}
    label_errors_third = {i: 0 for i in range(num_classes)}
    label_errors_second_strict = {i: 0 for i in range(num_classes)}
    label_errors_third_strict = {i: 0 for i in range(num_classes)}
    label_totals = {i: 0 for i in range(num_classes)}

    num_samples = min(args.eval_samples, len(dataset))

    timestamp = int(time.time())
    model_dir = os.path.dirname(args.snn_model_path)
    model_dir_name = os.path.basename(model_dir)
    os.makedirs("evals", exist_ok=True)
    results_filename = f"evals/{model_dir_name}_eval_{timestamp}.jsonl"
    results_file = open(results_filename, "w", buffering=1)
    print(f"Streaming evaluation results to: {results_filename}")

    main_pbar = tqdm(total=num_samples, desc="Evaluation Progress", position=0, leave=True)

    base_ticks_per_image = args.ticks_per_image
    base_window_size = args.window_size
    max_ticks_to_add = int(base_ticks_per_image * (args.max_thinking_multiplier - 1.0))

    try:
        for i in range(num_samples):
            image_tensor, actual_label = dataset[i]
            actual_label = int(actual_label)
            label_totals[actual_label] += 1

            network_sim = _load_fresh_network(network_config_path, neuron_cls)
            nn_core.neural_net = network_sim
            layers = infer_layers_from_metadata(network_sim)
            input_layer_ids, synapses_per_neuron = determine_input_mapping(
                network_sim, layers
            )

            signals = image_to_signals(
                image_tensor,
                network_sim,
                input_layer_ids,
                synapses_per_neuron,
                dataset_config,
            )
            activity_buffer.clear()
            network_sim.reset_simulation()

            mem1 = snn_model.lif1.init_leaky()
            mem2 = snn_model.lif2.init_leaky()
            mem3 = snn_model.lif3.init_leaky()
            mem4 = snn_model.lif4.init_leaky()

            current_ticks_per_image = args.ticks_per_image
            current_window_size = args.window_size

            tick_pbar = tqdm(
                total=current_ticks_per_image,
                desc=f"Image {i + 1}/{num_samples} (Label: {actual_label})",
                position=1,
                leave=False,
            )

            result = _run_single_image_eval(
                nn_core=nn_core,
                network_sim=network_sim,
                layers=layers,
                signals=signals,
                activity_buffer=activity_buffer,
                snn_model=snn_model,
                softmax=softmax,
                scaler_state=scaler_state,
                feature_types=feature_types,
                device=device,
                actual_label=actual_label,
                base_ticks_per_image=base_ticks_per_image,
                base_window_size=base_window_size,
                max_ticks_to_add=max_ticks_to_add,
                max_ticks=base_ticks_per_image + max_ticks_to_add,
                tick_pbar=tick_pbar,
                args=args,
                image_idx=i,
                num_samples=num_samples,
            )

            tick_pbar.close()

            result["image_idx"] = i
            result["actual_label"] = actual_label

            is_correct = (
                result["predicted_label"] == actual_label
                if result["predicted_label"] is not None
                else False
            )
            if not is_correct and result["predicted_label"] is not None:
                label_errors[actual_label] += 1

            _update_label_errors(result, actual_label, label_errors_second,
                label_errors_third, label_errors_second_strict,
                label_errors_third_strict, num_classes)

            json.dump(result, results_file, default=str)
            results_file.write("\n")

            eval_results.append(result)
            _update_main_pbar(main_pbar, eval_results, args, num_classes)

    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
    finally:
        main_pbar.close()
        results_file.close()
        _print_evaluation_summary(
            eval_results,
            label_errors,
            label_errors_second,
            label_errors_third,
            label_errors_second_strict,
            label_errors_third_strict,
            label_totals,
            num_classes,
            args,
            model_dir_name,
            timestamp,
            results_filename,
            feature_types,
            device,
        )


def _load_fresh_network(network_config_path: str, neuron_cls: type) -> Any:
    from neuron.network_config import NetworkConfig
    return NetworkConfig.load_network_config(
        network_config_path, neuron_class=neuron_cls
    )


def _run_single_image_eval(
    *,
    nn_core: Any,
    network_sim: Any,
    layers: list[list[int]],
    signals: list[tuple[int, int, float]],
    activity_buffer: list[list[float]],
    snn_model: nn.Module,
    softmax: nn.Module,
    scaler_state: dict[str, Any],
    feature_types: list[str],
    device: torch.device,
    actual_label: int,
    base_ticks_per_image: int,
    base_window_size: int,
    max_ticks_to_add: int,
    max_ticks: int,
    tick_pbar: tqdm,
    args: Any,
    image_idx: int,
    num_samples: int,
) -> dict[str, Any]:
    """Run evaluation for a single image."""
    current_ticks_per_image = args.ticks_per_image
    current_window_size = args.window_size
    ticks_added = 0

    final_prediction = None
    final_confidence = 0.0
    final_second_prediction = None
    final_second_confidence = 0.0
    final_third_prediction = None
    final_third_confidence = 0.0
    first_correct_tick = None
    first_second_correct_tick = None
    first_third_correct_tick = None
    first_correct_appearance_tick = None
    base_time_prediction = None
    base_time_correct = False
    used_extended_thinking = False
    total_ticks_added = 0
    prediction_history: list[bool] = []

    tick = 0

    while tick < current_ticks_per_image and tick < max_ticks:
        nn_core.send_batch_signals(signals)
        nn_core.do_tick()

        snapshot = collect_features_consistently(network_sim, layers, feature_types)
        if scaler_state:
            snapshot = apply_scaling_to_snapshot(snapshot, scaler_state)
        activity_buffer.append(snapshot)
        if len(activity_buffer) > current_window_size:
            activity_buffer.pop(0)

        with torch.no_grad():
            input_sequence = (
                torch.tensor(activity_buffer, dtype=torch.float32)
                .unsqueeze(0)
                .to(device)
            )
            mem1 = snn_model.lif1.init_leaky()
            mem2 = snn_model.lif2.init_leaky()
            mem3 = snn_model.lif3.init_leaky()
            mem4 = snn_model.lif4.init_leaky()
            spk_rec = []
            for step in range(input_sequence.shape[1]):
                spk2_step, mem1, mem2, mem3, mem4 = snn_model(
                    input_sequence[:, step, :], mem1, mem2, mem3, mem4
                )
                spk_rec.append(spk2_step)
            spk_rec_tensor = torch.stack(spk_rec, dim=0)
            spike_counts = spk_rec_tensor.sum(dim=0)
            probabilities = softmax(spike_counts)

            top_prob, top_class = probabilities.max(1)
            current_prediction = top_class.item()
            current_confidence = top_prob.item()

            if base_time_prediction is None and tick >= base_ticks_per_image - 1:
                base_time_prediction = current_prediction
                base_time_correct = current_prediction == actual_label

            prediction_history.append(current_prediction == actual_label)
            if len(prediction_history) > 5:
                prediction_history.pop(0)

            final_prediction = current_prediction
            final_confidence = current_confidence
            sorted_probs = torch.sort(probabilities[0], descending=True)
            if len(sorted_probs[0]) > 1:
                final_second_prediction = int(sorted_probs[1][1].item())
                final_second_confidence = float(sorted_probs[0][1].item())
            if len(sorted_probs[0]) > 2:
                final_third_prediction = int(sorted_probs[1][2].item())
                final_third_confidence = float(sorted_probs[0][2].item())

        current_tick = tick + 1
        prob_diff = probabilities[0].max().item() - probabilities[0].min().item()
        has_meaningful_confidence = prob_diff > 0.01

        if has_meaningful_confidence:
            if first_correct_tick is None and final_prediction == actual_label and final_confidence > 0:
                first_correct_tick = current_tick
            if first_second_correct_tick is None and final_second_prediction == actual_label and final_second_confidence > 0:
                first_second_correct_tick = current_tick
            if first_third_correct_tick is None and final_third_prediction == actual_label and final_third_confidence > 0:
                first_third_correct_tick = current_tick
            if first_correct_appearance_tick is None and current_prediction == actual_label and current_confidence > 0:
                first_correct_appearance_tick = current_tick

        if (
            args.think_longer
            and len(prediction_history) >= 5
            and not all(prediction_history[-5:])
            and tick == current_ticks_per_image - 1
            and ticks_added < max_ticks_to_add
        ):
            used_extended_thinking = True
            total_ticks_added = ticks_added + 1
            ticks_added += 1
            current_ticks_per_image = base_ticks_per_image + ticks_added
            current_window_size = base_window_size + int(
                ticks_added * (base_window_size / base_ticks_per_image)
            )
            tick_pbar.total = current_ticks_per_image
            tick_pbar.set_description(
                f"Image {image_idx + 1}/{num_samples} (Label: {actual_label}) [Think +{ticks_added}]"
            )

        postfix_data = {
            "pred": final_prediction if final_prediction is not None else "N/A",
            "conf": f"{final_confidence:.2%}" if final_confidence > 0 else "N/A",
            "correct": (
                "✅" if final_prediction == actual_label
                else "❌" if final_prediction is not None
                else "⏳"
            ),
        }
        if first_correct_tick is not None:
            postfix_data["1st_tick"] = str(first_correct_tick)
        if first_second_correct_tick is not None:
            postfix_data["2nd_tick"] = str(first_second_correct_tick)
        if first_third_correct_tick is not None:
            postfix_data["3rd_tick"] = str(first_third_correct_tick)
        tick_pbar.set_postfix(postfix_data)
        tick_pbar.update(1)
        tick += 1

    had_correct_appearance_but_wrong_final = (
        first_correct_appearance_tick is not None and final_prediction != actual_label
    )

    is_second_correct = final_second_prediction == actual_label if final_second_prediction is not None else False
    is_third_correct = final_third_prediction == actual_label if final_third_prediction is not None else False
    is_second_correct_strict = (
        is_second_correct and final_second_confidence > 0.0
        if final_second_prediction is not None
        else False
    )
    is_third_correct_strict = (
        is_third_correct and final_third_confidence > 0.0
        if final_third_prediction is not None
        else False
    )

    is_bistability_rescue_correct = final_prediction == actual_label if final_prediction is not None else False
    if (
        args.bistability_rescue
        and not is_bistability_rescue_correct
        and final_second_prediction == actual_label
        and final_second_confidence is not None
        and final_confidence is not None
        and (final_confidence - final_second_confidence) < 0.05
    ):
        is_bistability_rescue_correct = True

    return {
        "predicted_label": final_prediction,
        "confidence": final_confidence,
        "correct": final_prediction == actual_label if final_prediction is not None else False,
        "bistability_rescue_correct": is_bistability_rescue_correct,
        "second_predicted_label": final_second_prediction,
        "second_confidence": final_second_confidence,
        "second_correct": is_second_correct,
        "second_correct_strict": is_second_correct_strict,
        "third_predicted_label": final_third_prediction,
        "third_confidence": final_third_confidence,
        "third_correct": is_third_correct,
        "third_correct_strict": is_third_correct_strict,
        "first_correct_tick": first_correct_tick,
        "first_correct_appearance_tick": first_correct_appearance_tick,
        "first_second_correct_tick": first_second_correct_tick,
        "first_third_correct_tick": first_third_correct_tick,
        "had_correct_appearance_but_wrong_final": had_correct_appearance_but_wrong_final,
        "used_extended_thinking": used_extended_thinking,
        "total_ticks_added": total_ticks_added,
        "base_ticks_per_image": base_ticks_per_image,
        "base_time_prediction": base_time_prediction,
        "base_time_correct": base_time_correct,
    }


def _update_label_errors(
    result: dict[str, Any],
    actual_label: int,
    label_errors_second: dict[int, int],
    label_errors_third: dict[int, int],
    label_errors_second_strict: dict[int, int],
    label_errors_third_strict: dict[int, int],
    num_classes: int,
) -> None:
    is_correct = result.get("correct", False)
    is_second_correct = result.get("second_correct", False)
    is_third_correct = result.get("third_correct", False)
    is_second_correct_strict = result.get("second_correct_strict", False)
    is_third_correct_strict = result.get("third_correct_strict", False)
    final_second = result.get("second_predicted_label")
    final_third = result.get("third_predicted_label")

    if not is_correct and not is_second_correct and final_second is not None:
        label_errors_second[actual_label] += 1
    if not is_correct and not is_second_correct and not is_third_correct and final_third is not None:
        label_errors_third[actual_label] += 1
    if not is_correct and not is_second_correct_strict and final_second is not None:
        label_errors_second_strict[actual_label] += 1
    if not is_correct and not is_second_correct_strict and not is_third_correct_strict and final_third is not None:
        label_errors_third_strict[actual_label] += 1


def _update_main_pbar(
    main_pbar: tqdm,
    eval_results: list[dict[str, Any]],
    args: Any,
    num_classes: int,
) -> None:
    """Update the main progress bar with current metrics."""
    if not eval_results:
        return
    current_accuracy = sum(1 for r in eval_results if r["correct"]) / len(eval_results) * 100
    current_bistability_rescue_accuracy = (
        sum(1 for r in eval_results if r.get("bistability_rescue_correct", r["correct"])) / len(eval_results) * 100
        if args.bistability_rescue else 0.0
    )
    second_choice_accuracy = (
        sum(1 for r in eval_results if r["correct"] or r["second_correct"]) / len(eval_results) * 100
    )
    third_choice_accuracy = (
        sum(1 for r in eval_results if r["correct"] or r["second_correct"] or r["third_correct"]) / len(eval_results) * 100
    )
    current_first_correct_ticks = [r["first_correct_tick"] for r in eval_results if r.get("first_correct_tick") is not None]
    current_second_correct_ticks = [r["first_second_correct_tick"] for r in eval_results if r.get("first_second_correct_tick") is not None]
    current_third_correct_ticks = [r["first_third_correct_tick"] for r in eval_results if r.get("first_third_correct_tick") is not None]
    current_correct_appearance_ticks = [r["first_correct_appearance_tick"] for r in eval_results if r.get("first_correct_appearance_tick") is not None]
    current_appeared_but_wrong_final = [r for r in eval_results if r.get("had_correct_appearance_but_wrong_final")]
    current_appeared_and_correct_final = [r for r in eval_results if r.get("first_correct_appearance_tick") is not None and r["correct"]]
    current_thinking_results = [r for r in eval_results if r.get("used_extended_thinking")]
    current_thinking_ticks = [r["total_ticks_added"] for r in current_thinking_results]
    current_processed = [r for r in eval_results if r.get("predicted_label") is not None]
    current_base_time_correct = [r for r in current_processed if r.get("base_time_correct", False)]
    current_final_correct = [r for r in current_processed if r["correct"]]

    main_pbar.set_postfix({
        "1st_acc": f"{current_accuracy:.1f}%",
        "2nd_acc": f"{second_choice_accuracy:.1f}%",
        "3rd_acc": f"{third_choice_accuracy:.1f}%",
        "1st_time": f"{sum(current_first_correct_ticks) / len(current_first_correct_ticks):.1f}" if current_first_correct_ticks else "N/A",
        "2nd_time": f"{sum(current_second_correct_ticks) / len(current_second_correct_ticks):.1f}" if current_second_correct_ticks else "N/A",
        "3rd_time": f"{sum(current_third_correct_ticks) / len(current_third_correct_ticks):.1f}" if current_third_correct_ticks else "N/A",
        "unstable": str(len(current_appeared_but_wrong_final)),
        "stable": str(len(current_appeared_and_correct_final)),
        "think_ticks": f"{sum(current_thinking_ticks) / len(current_thinking_ticks):.1f}" if current_thinking_ticks else "N/A",
        "base_acc": f"{len(current_base_time_correct) / len(current_processed) * 100:.1f}%" if current_processed else "N/A",
        "final_acc": f"{len(current_final_correct) / len(current_processed) * 100:.1f}%" if current_processed else "N/A",
        "bistab_acc": f"{current_bistability_rescue_accuracy:.1f}%" if args.bistability_rescue else "N/A",
        "correct": f"{sum(1 for r in eval_results if r['correct'])}/{len(eval_results)}",
    })
    main_pbar.update(1)


def _print_evaluation_summary(
    eval_results: list[dict[str, Any]],
    label_errors: dict[int, int],
    label_errors_second: dict[int, int],
    label_errors_third: dict[int, int],
    label_errors_second_strict: dict[int, int],
    label_errors_third_strict: dict[int, int],
    label_totals: dict[int, int],
    num_classes: int,
    args: Any,
    model_dir_name: str,
    timestamp: int,
    results_filename: str,
    feature_types: list[str],
    device: torch.device,
) -> None:
    """Print and save evaluation summary."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    total_samples = len(eval_results)
    if total_samples == 0:
        print("No results to display.")
        return

    total_correct = sum(1 for r in eval_results if r["correct"])
    total_second_correct = sum(1 for r in eval_results if r["correct"] or r["second_correct"])
    total_third_correct = sum(1 for r in eval_results if r["correct"] or r["second_correct"] or r["third_correct"])
    total_second_correct_strict = sum(1 for r in eval_results if r["correct"] or r["second_correct_strict"])
    total_third_correct_strict = sum(1 for r in eval_results if r["correct"] or r["second_correct_strict"] or r["third_correct_strict"])

    overall_accuracy = total_correct / total_samples * 100
    overall_second_accuracy = total_second_correct / total_samples * 100
    overall_third_accuracy = total_third_correct / total_samples * 100
    overall_second_accuracy_strict = total_second_correct_strict / total_samples * 100
    overall_third_accuracy_strict = total_third_correct_strict / total_samples * 100

    overall_bistability_rescue_accuracy = None
    bistability_rescue_improvement = None
    if args.bistability_rescue:
        total_bistability_rescue_correct = sum(
            1 for r in eval_results if r.get("bistability_rescue_correct", r["correct"])
        )
        overall_bistability_rescue_accuracy = total_bistability_rescue_correct / total_samples * 100
        bistability_rescue_improvement = overall_bistability_rescue_accuracy - overall_accuracy

    print(f"First Choice Accuracy: {overall_accuracy:.2f}% ({total_correct}/{total_samples})")
    if args.bistability_rescue:
        print(
            f"Bistability Rescue Accuracy: {overall_bistability_rescue_accuracy:.2f}% "
            f"({total_bistability_rescue_correct}/{total_samples}) "
            f"[+{bistability_rescue_improvement or 0:.2f}% improvement]"
        )
    print(f"Second Choice Accuracy: {overall_second_accuracy:.2f}% ({total_second_correct}/{total_samples})")
    print(f"Third Choice Accuracy: {overall_third_accuracy:.2f}% ({total_third_correct}/{total_samples})")
    print(f"Total Errors (1st choice): {total_samples - total_correct}")
    print(f"Total Errors (2nd choice): {total_samples - total_second_correct}")
    print(f"Total Errors (3rd choice): {total_samples - total_third_correct}")
    print()
    print("Strict Accuracy (excluding 0% confidence correct predictions):")
    print("-" * 60)
    print(f"Strict Second Choice Accuracy: {overall_second_accuracy_strict:.2f}% ({total_second_correct_strict}/{total_samples})")
    print(f"Strict Third Choice Accuracy: {overall_third_accuracy_strict:.2f}% ({total_third_correct_strict}/{total_samples})")
    print(f"Strict Total Errors (2nd choice): {total_samples - total_second_correct_strict}")
    print(f"Strict Total Errors (3rd choice): {total_samples - total_third_correct_strict}")
    print(f"Zero-confidence contribution (2nd): {total_second_correct - total_second_correct_strict} samples")
    print(f"Zero-confidence contribution (3rd): {total_third_correct - total_third_correct_strict} samples")
    print()

    first_correct_ticks = [r["first_correct_tick"] for r in eval_results if r.get("first_correct_tick") is not None]
    second_correct_ticks = [r["first_second_correct_tick"] for r in eval_results if r.get("first_second_correct_tick") is not None]
    third_correct_ticks = [r["first_third_correct_tick"] for r in eval_results if r.get("first_third_correct_tick") is not None]
    avg_first = sum(first_correct_ticks) / len(first_correct_ticks) if first_correct_ticks else 0
    avg_second = sum(second_correct_ticks) / len(second_correct_ticks) if second_correct_ticks else 0
    avg_third = sum(third_correct_ticks) / len(third_correct_ticks) if third_correct_ticks else 0

    print("Average Ticks to Correct Prediction:")
    print("-" * 40)
    print(f"1st Choice: {avg_first:.1f} ticks ({len(first_correct_ticks)}/{total_samples} samples)")
    print(f"2nd Choice: {avg_second:.1f} ticks ({len(second_correct_ticks)}/{total_samples} samples)")
    print(f"3rd Choice: {avg_third:.1f} ticks ({len(third_correct_ticks)}/{total_samples} samples)")
    print()

    correct_appearance_ticks = [r["first_correct_appearance_tick"] for r in eval_results if r.get("first_correct_appearance_tick") is not None]
    appeared_but_wrong_final = [r for r in eval_results if r.get("had_correct_appearance_but_wrong_final")]
    appeared_and_correct_final = [r for r in eval_results if r.get("first_correct_appearance_tick") is not None and r["correct"]]
    avg_correct_appearance = sum(correct_appearance_ticks) / len(correct_appearance_ticks) if correct_appearance_ticks else 0
    stability_rate = (
        len(appeared_and_correct_final) / len(correct_appearance_ticks) * 100
        if correct_appearance_ticks
        else None
    )

    print("Correct Prediction Appearance Analysis:")
    print("-" * 40)
    print(f"Avg ticks to first correct appearance: {avg_correct_appearance:.1f} ticks ({len(correct_appearance_ticks)}/{total_samples} samples)")
    print(f"Current avg ticks to sustained correct: {avg_first:.1f} ticks ({len(first_correct_ticks)}/{total_samples} samples)")
    print()
    print("Correct Appearance vs Final Result:")
    print("-" * 40)
    print(f"Correct appeared but final wrong: {len(appeared_but_wrong_final)} samples")
    print(f"Correct appeared and final correct: {len(appeared_and_correct_final)} samples")
    print(f"Total samples where correct appeared: {len(correct_appearance_ticks)} samples")
    if stability_rate is not None:
        print(f"Correct prediction stability rate: {stability_rate:.1f}%")
    print()

    processed_results = [r for r in eval_results if r.get("predicted_label") is not None]
    base_time_correct_results = [r for r in processed_results if r.get("base_time_correct", False)]
    final_correct_results = [r for r in processed_results if r["correct"]]
    base_time_accuracy = len(base_time_correct_results) / len(processed_results) * 100 if processed_results else 0
    final_accuracy = len(final_correct_results) / len(processed_results) * 100 if processed_results else 0
    thinking_results = [r for r in processed_results if r.get("used_extended_thinking")]
    avg_ticks_added = sum(r["total_ticks_added"] for r in thinking_results) / len(thinking_results) if thinking_results else 0

    print("Thinking Effort Analysis:")
    print("-" * 40)
    print(f"Average ticks added: {avg_ticks_added:.1f} ticks")
    print(f"Accuracy without thinking (base time): {base_time_accuracy:.1f}% ({len(base_time_correct_results)}/{len(processed_results)})")
    print(f"Accuracy with thinking (final): {final_accuracy:.1f}% ({len(final_correct_results)}/{len(processed_results)})")
    print(f"Accuracy improvement: {final_accuracy - base_time_accuracy:.1f}%")

    label_base_correct: dict[int, int] = {}
    label_final_correct: dict[int, int] = {}
    label_thinking_count: dict[int, int] = {}
    for result in processed_results:
        label = result["actual_label"]
        if result.get("base_time_correct", False):
            label_base_correct[label] = label_base_correct.get(label, 0) + 1
        if result["correct"]:
            label_final_correct[label] = label_final_correct.get(label, 0) + 1
        if result.get("used_extended_thinking"):
            label_thinking_count[label] = label_thinking_count.get(label, 0) + 1

    print()
    print("Labels Requiring Extended Thinking:")
    print("-" * 40)
    for label in sorted(set(list(label_base_correct.keys()) + list(label_final_correct.keys()) + list(label_thinking_count.keys()))):
        base_correct = label_base_correct.get(label, 0)
        final_correct = label_final_correct.get(label, 0)
        thinking_used = label_thinking_count.get(label, 0)
        total_for_label = len([r for r in processed_results if r["actual_label"] == label])
        if total_for_label > 0:
            base_acc = base_correct / total_for_label * 100
            final_acc = final_correct / total_for_label * 100
            print(f"Label {label:2d}: Base {base_acc:4.1f}% → Final {final_acc:4.1f}% (+{final_acc - base_acc:4.1f}%) | Thinking: {thinking_used:2d}/{total_for_label:2d}")

    print()
    for title, errors_dict in [
        ("First Choice", label_errors),
        ("Second Choice", label_errors_second),
        ("Third Choice", label_errors_third),
        ("Strict Second Choice", label_errors_second_strict),
        ("Strict Third Choice", label_errors_third_strict),
    ]:
        print(f"{title} Error Analysis by Label:")
        print("-" * 50)
        for label in range(num_classes):
            if label_totals[label] > 0:
                errors = errors_dict[label]
                total = label_totals[label]
                error_rate = errors / total * 100
                print(f"Label {label:2d}: {errors:3d}/{total:3d} errors ({error_rate:5.1f}%)")
            else:
                print(f"Label {label:2d}: No samples")
        print()

    print("Detailed Results (First 10 samples):")
    print("-" * 90)
    for i, result in enumerate(eval_results[:10]):
        status_1st = "✅" if result["correct"] else "❌"
        status_2nd = "✅" if result["second_correct"] else "❌"
        status_3rd = "✅" if result["third_correct"] else "❌"
        status_2nd_strict = "✅" if result["second_correct_strict"] else "❌"
        status_3rd_strict = "✅" if result["third_correct_strict"] else "❌"
        appearance_status = "🔄" if result.get("first_correct_appearance_tick") is not None else "➖"
        final_after_appearance = "😔" if result.get("had_correct_appearance_but_wrong_final") else "😊"
        second_pred = result.get("second_predicted_label", "N/A")
        second_conf = result.get("second_confidence", 0.0) or 0.0
        third_pred = result.get("third_predicted_label", "N/A")
        third_conf = result.get("third_confidence", 0.0) or 0.0
        appearance_tick = result.get("first_correct_appearance_tick") or "N/A"
        sustained_tick = result.get("first_correct_tick") or "N/A"
        print(
            f"{i + 1:2d}. Label {result['actual_label']} → 1st: {result['predicted_label']} "
            f"({result['confidence']:.2%}) {status_1st} | Appear@{appearance_tick}/Sustained@{sustained_tick} {appearance_status}{final_after_appearance} | 2nd: {second_pred} "
            f"({second_conf:.2%}) {status_2nd}/{status_2nd_strict} | 3rd: {third_pred} ({third_conf:.2%}) {status_3rd}/{status_3rd_strict}"
        )
    if len(eval_results) > 10:
        print(f"... and {len(eval_results) - 10} more samples")

    summary_filename = f"evals/{model_dir_name}_eval_{timestamp}_summary.json"
    total_bistability_rescue_correct = sum(
        1 for r in eval_results if r.get("bistability_rescue_correct", r["correct"])
    ) if args.bistability_rescue else 0

    results_data = {
        "evaluation_metadata": {
            "timestamp": timestamp,
            "dataset_name": args.dataset_name,
            "neuron_model_path": args.neuron_model_path,
            "snn_model_path": args.snn_model_path,
            "ablation": args.ablation or "none",
            "ticks_per_image": args.ticks_per_image,
            "window_size": args.window_size,
            "eval_samples": len(eval_results),
            "think_longer_enabled": args.think_longer,
            "max_thinking_multiplier": args.max_thinking_multiplier,
            "bistability_rescue_enabled": args.bistability_rescue,
            "feature_types": feature_types,
            "num_classes": num_classes,
            "device": str(device),
            "results_file": results_filename,
        },
        "calculated_metrics": {
            "accuracy_metrics": {
                "first_choice_accuracy": overall_accuracy,
                "second_choice_accuracy": overall_second_accuracy,
                "third_choice_accuracy": overall_third_accuracy,
                "strict_second_choice_accuracy": overall_second_accuracy_strict,
                "strict_third_choice_accuracy": overall_third_accuracy_strict,
                "bistability_rescue_accuracy": overall_bistability_rescue_accuracy,
                "bistability_rescue_improvement": bistability_rescue_improvement,
                "total_errors_first_choice": total_samples - total_correct,
                "total_errors_second_choice": total_samples - total_second_correct,
                "total_errors_third_choice": total_samples - total_third_correct,
                "strict_total_errors_second_choice": total_samples - total_second_correct_strict,
                "strict_total_errors_third_choice": total_samples - total_third_correct_strict,
                "zero_confidence_contribution_second": total_second_correct - total_second_correct_strict,
                "zero_confidence_contribution_third": total_third_correct - total_third_correct_strict,
            },
            "timing_metrics": {
                "avg_ticks_to_first_correct": avg_first,
                "avg_ticks_to_second_correct": avg_second,
                "avg_ticks_to_third_correct": avg_third,
                "avg_ticks_to_correct_appearance": avg_correct_appearance,
                "correct_prediction_stability_rate": stability_rate,
            },
            "thinking_effort_analysis": {
                "avg_ticks_added": avg_ticks_added,
                "base_time_accuracy": base_time_accuracy,
                "final_accuracy": final_accuracy,
                "accuracy_improvement": final_accuracy - base_time_accuracy,
            },
            "error_analysis_by_label": {
                "first_choice_errors": {str(l): {"errors": label_errors[l], "total": label_totals[l], "error_rate": label_errors[l] / label_totals[l] * 100 if label_totals[l] > 0 else 0} for l in range(num_classes)},
                "second_choice_errors": {str(l): {"errors": label_errors_second[l], "total": label_totals[l], "error_rate": label_errors_second[l] / label_totals[l] * 100 if label_totals[l] > 0 else 0} for l in range(num_classes)},
                "third_choice_errors": {str(l): {"errors": label_errors_third[l], "total": label_totals[l], "error_rate": label_errors_third[l] / label_totals[l] * 100 if label_totals[l] > 0 else 0} for l in range(num_classes)},
                "strict_second_choice_errors": {str(l): {"errors": label_errors_second_strict[l], "total": label_totals[l], "error_rate": label_errors_second_strict[l] / label_totals[l] * 100 if label_totals[l] > 0 else 0} for l in range(num_classes)},
                "strict_third_choice_errors": {str(l): {"errors": label_errors_third_strict[l], "total": label_totals[l], "error_rate": label_errors_third_strict[l] / label_totals[l] * 100 if label_totals[l] > 0 else 0} for l in range(num_classes)},
            },
            "per_label_thinking_impact": {
                str(label): {
                    "base_correct": label_base_correct.get(label, 0),
                    "final_correct": label_final_correct.get(label, 0),
                    "thinking_used": label_thinking_count.get(label, 0),
                    "total_samples": len([r for r in processed_results if r["actual_label"] == label]),
                    "base_accuracy": (label_base_correct.get(label, 0) / total_for_label * 100) if (total_for_label := len([r for r in processed_results if r["actual_label"] == label])) > 0 else 0,
                    "final_accuracy": (label_final_correct.get(label, 0) / total_for_label * 100) if (total_for_label := len([r for r in processed_results if r["actual_label"] == label])) > 0 else 0,
                }
                for label in sorted(set(list(label_base_correct.keys()) + list(label_final_correct.keys()) + list(label_thinking_count.keys())))
            },
            "appearance_vs_final_analysis": {
                "appeared_but_wrong_final_count": len(appeared_but_wrong_final),
                "appeared_and_correct_final_count": len(appeared_and_correct_final),
                "total_correct_appearances": len(correct_appearance_ticks),
            },
        },
    }

    try:
        with open(summary_filename, "w") as f:
            json.dump(results_data, f, indent=2, default=str)
        print(f"\nEvaluation summary saved to: {summary_filename}")
        print(f"Individual results streamed to: {results_filename}")
    except Exception as e:
        print(f"Warning: Failed to save evaluation summary to JSON: {e}")

    print("\nExiting evaluation mode.")
