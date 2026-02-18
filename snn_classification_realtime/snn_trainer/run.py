"""Main orchestration for SNN classifier training."""

import datetime
import json
import os
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from snn_classification_realtime.snn_trainer.config import TrainConfig
from snn_classification_realtime.snn_trainer.dataset import ActivityDataset, collate_fn
from snn_classification_realtime.snn_trainer.model import SNNClassifier, HIDDEN_SIZE
from snn_classification_realtime.snn_trainer.metadata import load_dataset_metadata
from snn_classification_realtime.snn_trainer.checkpoint import (
    load_interrupted_state,
    save_checkpoint,
)
from snn_classification_realtime.snn_trainer.evaluation import test_model
from snn_classification_realtime.snn_trainer.plotting import save_loss_graph


def run_train(config: TrainConfig) -> None:
    """Run the full SNN classifier training workflow."""
    device = torch.device(
        config.device
        if config.device
        else (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    )
    print(f"Using device: {device}")

    dataset_basename = os.path.basename(os.path.normpath(config.dataset_dir))
    run_dir_name = (
        f"{dataset_basename}_e{config.epochs}_lr{config.learning_rate}_b{config.batch_size}"
    )
    run_dir_path = os.path.join(config.output_dir, run_dir_name)
    os.makedirs(run_dir_path, exist_ok=True)
    print(f"Created training run directory: {run_dir_path}")

    if config.model_save_path == "snn_model.pth":
        model_filename = "model.pth"
        model_save_path = os.path.join(run_dir_path, model_filename)
    else:
        model_filename = os.path.basename(config.model_save_path)
        model_save_path = os.path.join(run_dir_path, model_filename)

    train_dataset = ActivityDataset(
        os.path.join(config.dataset_dir, "train_data.pt"),
        os.path.join(config.dataset_dir, "train_labels.pt"),
    )
    test_dataset = ActivityDataset(
        os.path.join(config.dataset_dir, "test_data.pt"),
        os.path.join(config.dataset_dir, "test_labels.pt"),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    dataset_metadata = load_dataset_metadata(config.dataset_dir)
    feature_types = dataset_metadata.get("feature_types", ["firings"])
    num_features = dataset_metadata.get("num_features", 1)

    print(f"Dataset feature configuration: {feature_types}")
    print(f"Number of feature types: {num_features}")
    print("Architecture: SNN")

    sample_data, sample_label = train_dataset[0]
    input_size = sample_data.shape[1]
    num_classes = len(
        torch.unique(torch.cat([train_dataset.labels, test_dataset.labels]))
    )
    print(f"Input feature size: {input_size}, Number of classes: {num_classes}")

    print("Initializing standard SNN model...")
    net = SNNClassifier(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        output_size=num_classes,
    ).to(device)

    interrupted_state = None
    if config.load_model_path:
        print(f"Loading pre-trained model from {config.load_model_path}")
        net.load_state_dict(torch.load(config.load_model_path))
        config_path = config.load_model_path.replace(".pth", "_config.json")
        interrupted_state = load_interrupted_state(config_path)
        if interrupted_state:
            print(
                f"Detected interrupted training from epoch "
                f"{interrupted_state['interruption_epoch']}"
            )
            print("Resuming training from where it was interrupted...")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        net.parameters(), lr=config.learning_rate, betas=(0.9, 0.999)
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=20, gamma=0.75
    )

    if interrupted_state:
        epoch_losses = interrupted_state.get("epoch_losses", [])
        epoch_accuracies = interrupted_state.get("epoch_accuracies", [])
        test_accuracies = interrupted_state.get("test_accuracies", [])
        test_losses = interrupted_state.get("test_losses", [])
        start_epoch = interrupted_state.get("completed_epochs", 0)
        print(
            f"Resuming from epoch {start_epoch + 1}, continuing for "
            f"{config.epochs - start_epoch} more epochs"
        )
    else:
        epoch_losses = []
        epoch_accuracies = []
        test_accuracies = []
        test_losses: list[float | None] = []
        start_epoch = 0

    training_pbar = tqdm(
        range(start_epoch, config.epochs), desc="Training Progress", position=0
    )
    latest_train_acc = 0.0
    latest_test_acc = 0.0

    try:
        for epoch in training_pbar:
            net.train()
            total_loss = 0.0
            total_correct = 0
            total_samples = 0

            with tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{config.epochs}",
                position=1,
                leave=False,
            ) as epoch_pbar:
                for data, labels in epoch_pbar:
                    data = data.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()

                    mem1 = net.lif1.init_leaky()
                    mem2 = net.lif2.init_leaky()
                    mem3 = net.lif3.init_leaky()
                    mem4 = net.lif4.init_leaky()

                    spk_rec = []
                    for step in range(data.shape[1]):
                        spk2, mem1, mem2, mem3, mem4 = net(
                            data[:, step, :], mem1, mem2, mem3, mem4
                        )
                        spk_rec.append(spk2)

                    spk_rec = torch.stack(spk_rec, dim=1)
                    loss = criterion(spk_rec.sum(dim=1), labels)

                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    correct = (
                        (spk_rec.sum(dim=1).argmax(dim=1) == labels).sum().item()
                    )
                    total_correct += correct
                    total_samples += labels.size(0)

                    epoch_pbar.set_postfix(
                        loss=f"{loss.item():.4f}",
                        acc=f"{(100 * total_correct / total_samples):.2f}%",
                        lr=f"{scheduler.get_last_lr()[0]:.6f}",
                    )

            if scheduler.get_last_lr()[0] >= 0.00001:
                scheduler.step()

            avg_loss = total_loss / len(train_loader)
            train_acc = 100 * total_correct / total_samples
            epoch_losses.append(avg_loss)
            epoch_accuracies.append(train_acc)
            latest_train_acc = train_acc

            if config.test_every > 0 and (epoch + 1) % config.test_every == 0:
                with tqdm([0], desc="Testing", position=2, leave=False) as test_pbar:
                    test_pbar.set_description("Testing model...")
                    test_acc, test_loss = test_model(
                        net, test_loader, device, criterion, epoch + 1
                    )
                    test_accuracies.append(test_acc)
                    test_losses.append(test_loss)
                    latest_test_acc = test_acc
                    test_pbar.set_postfix(
                        accuracy=f"{test_acc:.2f}%",
                        loss=f"{(test_loss if test_loss is not None else float('nan')):.4f}",
                    )
                    test_pbar.update(1)

            postfix_dict = {
                "epoch": f"{epoch + 1}/{config.epochs}",
                "loss": f"{avg_loss:.4f}",
                "train_acc": f"{latest_train_acc:.2f}%",
            }
            if latest_test_acc > 0:
                postfix_dict["test_acc"] = f"{latest_test_acc:.2f}%"
            training_pbar.set_postfix(postfix_dict)

            save_checkpoint(
                net,
                epoch,
                epoch_losses,
                epoch_accuracies,
                test_accuracies,
                test_losses,
                config,
                dataset_basename,
                run_dir_path,
                model_save_path,
                device,
                input_size,
                num_classes,
                feature_types,
                num_features,
                dataset_metadata,
                run_dir_name,
            )

    except KeyboardInterrupt:
        _handle_interrupt(
            net,
            epoch,
            epoch_losses,
            epoch_accuracies,
            test_accuracies,
            test_losses,
            config,
            dataset_basename,
            run_dir_path,
            model_save_path,
            device,
            input_size,
            num_classes,
            feature_types,
            num_features,
            dataset_metadata,
            run_dir_name,
            train_dataset,
            test_dataset,
        )
        return

    print(f"Saving trained model to {model_save_path}")
    torch.save(net.state_dict(), model_save_path)

    graph_save_path = model_save_path.replace(".pth", "_loss_graph.png")
    save_loss_graph(
        epoch_losses,
        epoch_accuracies,
        test_accuracies,
        test_losses,
        graph_save_path,
        config.test_every,
    )
    print(f"Saved loss graph to {graph_save_path}")

    train_config_dict: dict[str, Any] = {
        "model_type": "SNN",
        "dataset_dir": config.dataset_dir,
        "dataset_basename": dataset_basename,
        "run_dir_name": run_dir_name,
        "run_dir_path": run_dir_path,
        "model_save_path": model_save_path,
        "load_model_path": config.load_model_path,
        "output_dir": config.output_dir,
        "epochs": config.epochs,
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
        "final_train_loss": epoch_losses[-1] if epoch_losses else None,
        "final_train_accuracy": epoch_accuracies[-1] if epoch_accuracies else None,
        "test_accuracies": test_accuracies,
        "test_losses": test_losses,
        "total_train_samples": len(train_dataset),
        "total_test_samples": len(test_dataset),
        "feature_types": feature_types,
        "num_features": num_features,
        "dataset_metadata": dataset_metadata,
    }

    config_save_path = model_save_path.replace(".pth", "_config.json")
    with open(config_save_path, "w") as f:
        json.dump(train_config_dict, f, indent=2)
    print(f"Saved training configuration to {config_save_path}")

    print("\n" + "=" * 50)
    print("FINAL TEST RESULTS")
    print("=" * 50)

    with tqdm([0], desc="Final Testing", position=0) as final_test_pbar:
        final_test_pbar.set_description("Running final test...")
        final_test_acc, final_test_loss = test_model(
            net, test_loader, device, criterion
        )
        final_test_pbar.set_postfix(
            accuracy=f"{final_test_acc:.2f}%",
            loss=f"{(final_test_loss if final_test_loss is not None else float('nan')):.4f}",
        )
        final_test_pbar.update(1)

    train_config_dict["final_test_accuracy"] = final_test_acc
    train_config_dict["final_test_loss"] = final_test_loss
    with open(config_save_path, "w") as f:
        json.dump(train_config_dict, f, indent=2)


def _handle_interrupt(
    net: Any,
    epoch: int,
    epoch_losses: list[float],
    epoch_accuracies: list[float],
    test_accuracies: list[float],
    test_losses: list[float | None],
    config: TrainConfig,
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
    train_dataset: ActivityDataset,
    test_dataset: ActivityDataset,
) -> None:
    """Handle KeyboardInterrupt: save intermediate model and config."""
    print(f"\n\nTraining interrupted at epoch {epoch + 1}")
    print("Saving intermediate results...")

    intermediate_model_path = model_save_path.replace(
        ".pth", f"_interrupted_epoch_{epoch + 1}.pth"
    )
    torch.save(net.state_dict(), intermediate_model_path)
    print(f"Saved interrupted model to {intermediate_model_path}")

    intermediate_config = {
        "dataset_dir": config.dataset_dir,
        "dataset_basename": dataset_basename,
        "run_dir_name": run_dir_name,
        "run_dir_path": run_dir_path,
        "model_save_path": intermediate_model_path,
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
        "interrupted": True,
        "interruption_epoch": epoch + 1,
        "epoch_losses": epoch_losses,
        "epoch_accuracies": epoch_accuracies,
        "test_accuracies": test_accuracies,
        "test_losses": test_losses,
        "total_train_samples": len(train_dataset),
        "total_test_samples": len(test_dataset),
        "feature_types": feature_types,
        "num_features": num_features,
        "dataset_metadata": dataset_metadata,
    }

    intermediate_config_path = model_save_path.replace(
        ".pth", f"_interrupted_epoch_{epoch + 1}_config.json"
    )
    with open(intermediate_config_path, "w") as f:
        json.dump(intermediate_config, f, indent=2)
    print(f"Saved interrupted configuration to {intermediate_config_path}")

    if epoch_losses:
        intermediate_graph_path = model_save_path.replace(
            ".pth", f"_interrupted_epoch_{epoch + 1}_loss_graph.png"
        )
        save_loss_graph(
            epoch_losses,
            epoch_accuracies,
            test_accuracies,
            test_losses,
            intermediate_graph_path,
            config.test_every,
            title_suffix=" (Interrupted)",
        )
        print(f"Saved interrupted loss graph to {intermediate_graph_path}")

    print("\nTo resume training, use:")
    remaining_epochs = config.epochs - (epoch + 1)
    if remaining_epochs > 0:
        print(
            f"python -m snn_classification_realtime.train_snn_classifier "
            f"--dataset-dir {config.dataset_dir} "
            f"--load-model-path {intermediate_model_path} "
            f"--epochs {remaining_epochs}"
        )
    else:
        print("Training was already complete or nearly complete.")
    print("\nTraining interrupted gracefully. Intermediate results saved.")
