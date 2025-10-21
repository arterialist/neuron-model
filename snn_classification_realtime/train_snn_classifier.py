import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import snntorch as snn
from snntorch import spikegen
from snntorch import functional as SF
import matplotlib.pyplot as plt
import json
import datetime
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SNN_HIDDEN_SIZE = 512


class ActivityDataset(Dataset):
    """Custom PyTorch Dataset for activity time-series data."""

    def __init__(self, data_path, labels_path):
        self.data = torch.load(data_path)
        self.labels = torch.load(labels_path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Data is a list of tensors with potentially variable length
        return self.data[idx], self.labels[idx]


def collate_fn(batch):
    """Custom collate function to handle variable length sequences by padding."""
    # Pad sequences to the length of the longest sequence in the batch
    data, labels = zip(*batch)
    padded_data = nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=0)
    return padded_data, torch.stack([torch.as_tensor(l) for l in labels], dim=0)


def test_model(net, test_loader, device, criterion=None, epoch=None):
    """Test the SNN model and return accuracy and average loss (if criterion provided)."""
    net.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            labels = labels.to(device)

            # Initialize membrane potentials for SNN
            mem1 = net.lif1.init_leaky()
            mem2 = net.lif2.init_leaky()
            mem3 = net.lif3.init_leaky()
            mem4 = net.lif4.init_leaky()

            # Record spikes for the entire sequence
            spk_rec = []

            # Process each time step explicitly
            for step in range(data.shape[1]):  # loop over time dimension
                spk2, mem1, mem2, mem3, mem4 = net(
                    data[:, step, :], mem1, mem2, mem3, mem4
                )
                spk_rec.append(spk2)

            # Stack the recorded spikes
            spk_rec = torch.stack(spk_rec, dim=0)

            # Compute test loss if criterion provided
            if criterion is not None:
                loss = criterion(spk_rec.sum(0), labels)
                total_loss += loss.item()
                num_batches += 1

            # Test accuracy
            correct = (spk_rec.sum(0).argmax(dim=1) == labels).sum().item()
            total_correct += correct
            total_samples += labels.size(0)

    test_acc = 100 * total_correct / total_samples
    test_loss = (
        (total_loss / num_batches)
        if (criterion is not None and num_batches > 0)
        else None
    )
    return test_acc, test_loss


def load_interrupted_state(config_path):
    """Load interrupted training state from config file."""
    try:
        with open(config_path, "r") as f:
            config = json.load(f)

        if config.get("interrupted", False):
            return config
        return None
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None


def load_dataset_metadata(dataset_dir):
    """Load dataset metadata to understand feature configuration."""
    metadata_path = os.path.join(dataset_dir, "dataset_metadata.json")
    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        return metadata
    except (FileNotFoundError, json.JSONDecodeError):
        # Fallback for datasets without metadata (backward compatibility)
        return {
            "feature_types": ["firings"],  # Default assumption
            "num_features": 1,
        }


# Define the SNN model
class SNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        beta = 0.9  # Neuron membrane potential decay rate

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.lif2 = snn.Leaky(beta=beta)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.lif3 = snn.Leaky(beta=beta)
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.lif4 = snn.Leaky(beta=beta)

    def forward(self, x, mem1=None, mem2=None, mem3=None, mem4=None):
        """
        Forward pass for a single time step.

        Args:
            x: Input tensor of shape (batch_size, input_size)
            mem1: Previous membrane potential for first layer (optional)
            mem2: Previous membrane potential for second layer (optional)

        Returns:
            spk2: Output spikes for current time step
            mem1: Updated membrane potential for first layer
            mem2: Updated membrane potential for second layer
        """
        # Initialize hidden states if not provided
        if mem1 is None:
            mem1 = self.lif1.init_leaky()
        if mem2 is None:
            mem2 = self.lif2.init_leaky()
        if mem3 is None:
            mem3 = self.lif3.init_leaky()
        if mem4 is None:
            mem4 = self.lif4.init_leaky()

        # Single forward pass
        cur1 = self.fc1(x)
        spk1, mem1 = self.lif1(cur1, mem1)
        cur2 = self.fc2(spk1)
        spk2, mem2 = self.lif2(cur2, mem2)
        cur3 = self.fc3(spk2)
        spk3, mem3 = self.lif3(cur3, mem3)
        cur4 = self.fc4(spk3)
        spk4, mem4 = self.lif4(cur4, mem4)

        return spk4, mem1, mem2, mem3, mem4


def main():
    parser = argparse.ArgumentParser(
        description="Train an SNN classifier on network activity data."
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Path to the directory containing the .pt files.",
    )
    parser.add_argument(
        "--model-save-path",
        type=str,
        default="snn_model.pth",
        help="Path to save the trained model. (default: snn_model.pth)",
    )
    parser.add_argument(
        "--load-model-path",
        type=str,
        help="(Optional) Path to a pre-trained model to load.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save trained models.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs. (default: 10)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for the optimizer. (default: 1e-3)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Training batch size. (default: 32)"
    )
    parser.add_argument(
        "--test-every",
        type=int,
        default=0,
        help="Test the model every N epochs during training. Set to 0 to disable. (default: 0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="Device to use for training. (default: auto)",
    )
    args = parser.parse_args()

    # works 2x faster on CPU than MPS
    device = torch.device(
        args.device
        if args.device
        else (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
    )
    print(f"Using device: {device}")

    # Create a unique directory for this training run
    dataset_basename = os.path.basename(os.path.normpath(args.dataset_dir))
    run_dir_name = (
        f"{dataset_basename}_e{args.epochs}_lr{args.learning_rate}_b{args.batch_size}"
    )
    run_dir_path = os.path.join(args.output_dir, run_dir_name)
    os.makedirs(run_dir_path, exist_ok=True)
    print(f"Created training run directory: {run_dir_path}")

    # Construct model save path within the run directory
    if args.model_save_path == "snn_model.pth":  # Check if default value is used
        model_filename = "model.pth"
        model_save_path = os.path.join(run_dir_path, model_filename)
    else:
        # If custom model path is provided, use the run directory as base
        model_filename = os.path.basename(args.model_save_path)
        model_save_path = os.path.join(run_dir_path, model_filename)

    # 1. Load Data
    train_dataset = ActivityDataset(
        os.path.join(args.dataset_dir, "train_data.pt"),
        os.path.join(args.dataset_dir, "train_labels.pt"),
    )
    test_dataset = ActivityDataset(
        os.path.join(args.dataset_dir, "test_data.pt"),
        os.path.join(args.dataset_dir, "test_labels.pt"),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Load dataset metadata to understand feature configuration
    dataset_metadata = load_dataset_metadata(args.dataset_dir)
    feature_types = dataset_metadata.get("feature_types", ["firings"])
    num_features = dataset_metadata.get("num_features", 1)

    print(f"Dataset feature configuration: {feature_types}")
    print(f"Number of feature types: {num_features}")
    print(f"Architecture: SNN")

    # Set architecture to SNN (only supported option)
    architecture = "snn"

    # Determine input and output sizes from data
    sample_data, sample_label = train_dataset[0]
    input_size = sample_data.shape[1]

    # Find the number of unique classes
    num_classes = len(
        torch.unique(torch.cat([train_dataset.labels, test_dataset.labels]))
    )

    print(f"Input feature size: {input_size}, Number of classes: {num_classes}")

    # 2. Initialize SNN Model
    print("Initializing standard SNN model...")
    net = SNNClassifier(
        input_size=input_size, hidden_size=SNN_HIDDEN_SIZE, output_size=num_classes
    ).to(device)

    # Check if we're resuming from interrupted training
    interrupted_state = None
    if args.load_model_path:
        print(f"Loading pre-trained model from {args.load_model_path}")
        net.load_state_dict(torch.load(args.load_model_path))

        # Try to load interrupted state from config
        config_path = args.load_model_path.replace(".pth", "_config.json")
        interrupted_state = load_interrupted_state(config_path)

        if interrupted_state:
            print(
                f"Detected interrupted training from epoch {interrupted_state['interruption_epoch']}"
            )
            print("Resuming training from where it was interrupted...")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        net.parameters(), lr=args.learning_rate, betas=(0.9, 0.999)
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.75)

    # 3. Training Loop
    # Initialize training state
    if interrupted_state:
        # Resume from interrupted state
        epoch_losses = interrupted_state.get("epoch_losses", [])
        epoch_accuracies = interrupted_state.get("epoch_accuracies", [])
        test_accuracies = interrupted_state.get("test_accuracies", [])
        test_losses = interrupted_state.get("test_losses", [])
        start_epoch = interrupted_state.get("completed_epochs", 0)
        print(
            f"Resuming from epoch {start_epoch + 1}, continuing for {args.epochs - start_epoch} more epochs"
        )
    else:
        # Start fresh training
        epoch_losses = []
        epoch_accuracies = []
        test_accuracies = []
        test_losses = []
        start_epoch = 0

    # Create overall training progress bar
    training_pbar = tqdm(
        range(start_epoch, args.epochs), desc="Training Progress", position=0
    )

    # Initialize variables to track latest metrics
    latest_train_acc = 0.0
    latest_test_acc = 0.0

    try:
        for epoch in training_pbar:
            net.train()
            total_loss = 0
            total_correct = 0
            total_samples = 0

            with tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{args.epochs}",
                position=1,
                leave=False,
            ) as epoch_pbar:
                for batch_idx, (data, labels) in enumerate(epoch_pbar):
                    data = data.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    # Initialize membrane potentials for SNN
                    mem1 = net.lif1.init_leaky()
                    mem2 = net.lif2.init_leaky()
                    mem3 = net.lif3.init_leaky()
                    mem4 = net.lif4.init_leaky()

                    # Record spikes for the entire sequence
                    spk_rec = []

                    # Process each time step explicitly
                    for step in range(data.shape[1]):  # loop over time dimension
                        spk2, mem1, mem2, mem3, mem4 = net(
                            data[:, step, :], mem1, mem2, mem3, mem4
                        )
                        spk_rec.append(spk2)

                    # Stack the recorded spikes
                    spk_rec = torch.stack(spk_rec, dim=1)

                    # Loss computation for SNN
                    loss = criterion(spk_rec.sum(dim=1), labels)  # SNN uses spike sum

                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                    # Training accuracy
                    correct = (spk_rec.sum(dim=1).argmax(dim=1) == labels).sum().item()
                    total_correct += correct
                    total_samples += labels.size(0)

                    epoch_pbar.set_postfix(
                        loss=f"{loss.item():.4f}",
                        acc=f"{(100*total_correct/total_samples):.2f}%",
                        lr=f"{scheduler.get_last_lr()[0]:.6f}",
                    )

            if scheduler.get_last_lr()[0] >= 0.00001:
                scheduler.step()
            avg_loss = total_loss / len(train_loader)
            train_acc = 100 * total_correct / total_samples
            epoch_losses.append(avg_loss)
            epoch_accuracies.append(train_acc)
            latest_train_acc = train_acc

            # Test the model every N epochs if specified
            if args.test_every > 0 and (epoch + 1) % args.test_every == 0:
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

            # Update overall training progress bar with both training and test accuracy
            postfix_dict = {
                "epoch": f"{epoch+1}/{args.epochs}",
                "loss": f"{avg_loss:.4f}",
                "train_acc": f"{latest_train_acc:.2f}%",
            }

            # Add test accuracy if we have test results
            if latest_test_acc > 0:
                postfix_dict["test_acc"] = f"{latest_test_acc:.2f}%"

            training_pbar.set_postfix(postfix_dict)

    except KeyboardInterrupt:
        print(f"\n\nTraining interrupted at epoch {epoch+1}")
        print("Saving intermediate results...")

        # Save intermediate model
        intermediate_model_path = model_save_path.replace(
            ".pth", f"_interrupted_epoch_{epoch+1}.pth"
        )
        torch.save(net.state_dict(), intermediate_model_path)
        print(f"Saved interrupted model to {intermediate_model_path}")

        # Save intermediate configuration
        intermediate_config = {
            "dataset_dir": args.dataset_dir,
            "dataset_basename": dataset_basename,
            "run_dir_name": run_dir_name,
            "run_dir_path": run_dir_path,
            "model_save_path": intermediate_model_path,
            "load_model_path": args.load_model_path,
            "output_dir": args.output_dir,
            "epochs": args.epochs,
            "completed_epochs": epoch + 1,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "test_every": args.test_every,
            "device": str(device),
            "input_size": input_size,
            "hidden_size": SNN_HIDDEN_SIZE,
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
            "total_train_samples": len(train_dataset),
            "total_test_samples": len(test_dataset),
            # Add feature configuration metadata
            "feature_types": feature_types,
            "num_features": num_features,
            "dataset_metadata": dataset_metadata,
            "test_losses": test_losses,
        }

        intermediate_config_path = model_save_path.replace(
            ".pth", f"_interrupted_epoch_{epoch+1}_config.json"
        )
        with open(intermediate_config_path, "w") as f:
            json.dump(intermediate_config, f, indent=2)
        print(f"Saved interrupted configuration to {intermediate_config_path}")

        # Save intermediate loss graph
        if epoch_losses:
            # If we have test accuracies (and maybe test losses), use a 4-plot layout; else keep 2 plots
            plt.figure(figsize=(24, 4) if test_accuracies else (12, 4))

            if test_accuracies:
                plt.subplot(1, 4, 1)
                plt.plot(
                    range(1, len(epoch_losses) + 1), epoch_losses, "b-", linewidth=2
                )
                plt.title("Training Loss Over Epochs (Interrupted)")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.grid(True, alpha=0.3)

                plt.subplot(1, 4, 2)
                plt.plot(
                    range(1, len(epoch_accuracies) + 1),
                    epoch_accuracies,
                    "r-",
                    linewidth=2,
                )
                plt.title("Training Accuracy Over Epochs (Interrupted)")
                plt.xlabel("Epoch")
                plt.ylabel("Accuracy (%)")
                plt.grid(True, alpha=0.3)

                plt.subplot(1, 4, 3)
                test_epochs = [
                    i * args.test_every for i in range(1, len(test_accuracies) + 1)
                ]
                plt.plot(test_epochs, test_accuracies, "g-", linewidth=2)
                plt.title("Test Accuracy Over Epochs (Interrupted)")
                plt.xlabel("Epoch")
                plt.ylabel("Accuracy (%)")
                plt.grid(True, alpha=0.3)

                plt.subplot(1, 4, 4)
                if "test_losses" in locals() and test_losses:
                    plt.plot(test_epochs, test_losses, "m-", linewidth=2)
                else:
                    plt.plot([], [])
                plt.title("Test Loss Over Epochs (Interrupted)")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.grid(True, alpha=0.3)
            else:
                plt.subplot(1, 2, 1)
                plt.plot(
                    range(1, len(epoch_losses) + 1), epoch_losses, "b-", linewidth=2
                )
                plt.title("Training Loss Over Epochs (Interrupted)")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.grid(True, alpha=0.3)

                plt.subplot(1, 2, 2)
                plt.plot(
                    range(1, len(epoch_accuracies) + 1),
                    epoch_accuracies,
                    "r-",
                    linewidth=2,
                )
                plt.title("Training Accuracy Over Epochs (Interrupted)")
                plt.xlabel("Epoch")
                plt.ylabel("Accuracy (%)")
                plt.grid(True, alpha=0.3)

            plt.tight_layout()

            intermediate_graph_path = model_save_path.replace(
                ".pth", f"_interrupted_epoch_{epoch+1}_loss_graph.png"
            )
            plt.savefig(intermediate_graph_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Saved interrupted loss graph to {intermediate_graph_path}")

        print(f"\nTo resume training, use:")
        remaining_epochs = args.epochs - (epoch + 1)
        if remaining_epochs > 0:
            print(
                f"python {__file__} --dataset-dir {args.dataset_dir} --load-model-path {intermediate_model_path} --epochs {remaining_epochs}"
            )
        else:
            print("Training was already complete or nearly complete.")
        print("\nTraining interrupted gracefully. Intermediate results saved.")
        return

    # 4. Save the model
    print(f"Saving trained model to {model_save_path}")
    torch.save(net.state_dict(), model_save_path)

    # 5. Save loss graph
    if test_accuracies:
        plt.figure(figsize=(24, 4))

        plt.subplot(1, 4, 1)
        plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, "b-", linewidth=2)
        plt.title("Training Loss Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 4, 2)
        plt.plot(
            range(1, len(epoch_accuracies) + 1), epoch_accuracies, "r-", linewidth=2
        )
        plt.title("Training Accuracy Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 4, 3)
        test_epochs = [i * args.test_every for i in range(1, len(test_accuracies) + 1)]
        plt.plot(test_epochs, test_accuracies, "g-", linewidth=2, marker="o")
        plt.title("Test Accuracy Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 4, 4)
        if "test_losses" in locals() and test_losses:
            plt.plot(test_epochs, test_losses, "m-", linewidth=2, marker="o")
        else:
            plt.plot([], [])
        plt.title("Test Loss Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.3)
    else:
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, "b-", linewidth=2)
        plt.title("Training Loss Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(
            range(1, len(epoch_accuracies) + 1), epoch_accuracies, "r-", linewidth=2
        )
        plt.title("Training Accuracy Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save loss graph with same base name as model
    graph_save_path = model_save_path.replace(".pth", "_loss_graph.png")
    plt.savefig(graph_save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved loss graph to {graph_save_path}")

    # 6. Save training configuration
    config = {
        "model_type": "SNN",
        "dataset_dir": args.dataset_dir,
        "dataset_basename": dataset_basename,
        "run_dir_name": run_dir_name,
        "run_dir_path": run_dir_path,
        "model_save_path": model_save_path,
        "load_model_path": args.load_model_path,
        "output_dir": args.output_dir,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "test_every": args.test_every,
        "device": str(device),
        "input_size": input_size,
        "hidden_size": SNN_HIDDEN_SIZE,
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
        "test_losses": test_losses if "test_losses" in locals() else [],
        "total_train_samples": len(train_dataset),
        "total_test_samples": len(test_dataset),
        # Add feature configuration metadata
        "feature_types": feature_types,
        "num_features": num_features,
        "dataset_metadata": dataset_metadata,
    }

    config_save_path = model_save_path.replace(".pth", "_config.json")
    with open(config_save_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved training configuration to {config_save_path}")

    # 7. Final test the model
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

    # Add final test accuracy to config if not already there
    if "final_test_accuracy" not in config:
        config["final_test_accuracy"] = final_test_acc
        config["final_test_loss"] = final_test_loss
        # Update the config file
        with open(config_save_path, "w") as f:
            json.dump(config, f, indent=2)


if __name__ == "__main__":
    main()
