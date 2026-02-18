"""Checkpoint and interrupted state handling."""

import datetime
import json
from typing import Any


def load_interrupted_state(config_path: str) -> dict[str, Any] | None:
    """Load interrupted training state from config file."""
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        if config.get("interrupted", False):
            return config
        return None
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None


def save_checkpoint(
    net: Any,
    epoch: int,
    epoch_losses: list[float],
    epoch_accuracies: list[float],
    test_accuracies: list[float],
    test_losses: list[float | None],
    config: "TrainConfig",
    dataset_basename: str,
    run_dir_path: str,
    model_save_path: str,
    device: Any,
    input_size: int,
    num_classes: int,
    feature_types: list[str],
    num_features: int,
    dataset_metadata: dict[str, Any],
    run_dir_name: str,
) -> tuple[str, str]:
    """Save a training checkpoint for the current epoch."""
    from snn_classification_realtime.snn_trainer.model import HIDDEN_SIZE

    checkpoint_model_path = model_save_path.replace(
        ".pth", f"_checkpoint_epoch_{epoch + 1}.pth"
    )
    import torch
    torch.save(net.state_dict(), checkpoint_model_path)

    checkpoint_config = {
        "dataset_dir": config.dataset_dir,
        "dataset_basename": dataset_basename,
        "run_dir_name": run_dir_name,
        "run_dir_path": run_dir_path,
        "model_save_path": checkpoint_model_path,
        "load_model_path": config.load_model_path,
        "output_dir": config.output_dir,
        "epochs": config.epochs,
        "completed_epochs": epoch + 1,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "test_every": config.test_every,
        "device": str(device),
        "input_size": input_size,
        "hidden_size": HIDDEN_SIZE,
        "output_size": num_classes,
        "optimizer": "Adam",
        "optimizer_betas": [0.9, 0.999],
        "loss_function": "CrossEntropyLoss",
        "neuron_type": "Leaky",
        "beta": 0.9,
        "training_timestamp": datetime.datetime.now().isoformat(),
        "interrupted": False,
        "checkpoint_epoch": epoch + 1,
        "epoch_losses": epoch_losses,
        "epoch_accuracies": epoch_accuracies,
        "test_accuracies": test_accuracies,
        "test_losses": test_losses,
        "feature_types": feature_types,
        "num_features": num_features,
        "dataset_metadata": dataset_metadata,
    }

    checkpoint_config_path = model_save_path.replace(
        ".pth", f"_checkpoint_epoch_{epoch + 1}_config.json"
    )
    with open(checkpoint_config_path, "w") as f:
        json.dump(checkpoint_config, f, indent=2)

    return checkpoint_model_path, checkpoint_config_path
