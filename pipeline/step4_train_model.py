import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import snntorch as snn
import matplotlib.pyplot as plt
import json
import datetime
import sys
import yaml

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.config import TrainingConfig

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
    return padded_data, torch.stack([torch.as_tensor(label) for label in labels], dim=0)


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


def save_checkpoint(
    net,
    epoch,
    epoch_losses,
    epoch_accuracies,
    test_accuracies,
    test_losses,
    config: TrainingConfig,
    output_dir,
    device,
    input_size,
    num_classes,
    feature_types,
    num_features,
    dataset_metadata,
):
    """Save a training checkpoint for the current epoch."""
    # Save checkpoint model
    checkpoint_model_path = os.path.join(output_dir, f"model_checkpoint_epoch_{epoch + 1}.pth")
    torch.save(net.state_dict(), checkpoint_model_path)

    # Save checkpoint configuration
    checkpoint_config = {
        "output_dir": output_dir,
        "epochs": config.epochs,
        "completed_epochs": epoch + 1,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "test_every": config.test_every,
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
        "checkpoint_epoch": epoch + 1,
        "epoch_losses": epoch_losses,
        "epoch_accuracies": epoch_accuracies,
        "test_accuracies": test_accuracies,
        "test_losses": test_losses,
        "feature_types": feature_types,
        "num_features": num_features,
        "dataset_metadata": dataset_metadata,
    }

    checkpoint_config_path = os.path.join(output_dir, f"model_checkpoint_epoch_{epoch + 1}_config.json")
    with open(checkpoint_config_path, "w") as f:
        json.dump(checkpoint_config, f, indent=2)

    return checkpoint_model_path, checkpoint_config_path


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

def train_model(config: TrainingConfig, input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # works 2x faster on CPU than MPS
    device_str = config.device
    if device_str is None or device_str == "auto":
        device_str = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    device = torch.device(device_str)
    print(f"Using device: {device}")

    # 1. Load Data
    train_dataset = ActivityDataset(
        os.path.join(input_dir, "train_data.pt"),
        os.path.join(input_dir, "train_labels.pt"),
    )
    test_dataset = ActivityDataset(
        os.path.join(input_dir, "test_data.pt"),
        os.path.join(input_dir, "test_labels.pt"),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Load dataset metadata to understand feature configuration
    dataset_metadata = load_dataset_metadata(input_dir)
    feature_types = dataset_metadata.get("feature_types", ["firings"])
    num_features = dataset_metadata.get("num_features", 1)

    print(f"Dataset feature configuration: {feature_types}")
    print(f"Number of feature types: {num_features}")
    print("Architecture: SNN")

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

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        net.parameters(), lr=config.learning_rate, betas=(0.9, 0.999)
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.75)

    # 3. Training Loop
    epoch_losses = []
    epoch_accuracies = []
    test_accuracies = []
    test_losses = []
    start_epoch = 0

    # Create overall training progress bar
    training_pbar = tqdm(
        range(start_epoch, config.epochs), desc="Training Progress", position=0
    )

    latest_train_acc = 0.0
    latest_test_acc = 0.0

    model_save_path = os.path.join(output_dir, "model.pth")

    try:
        for epoch in training_pbar:
            net.train()
            total_loss = 0
            total_correct = 0
            total_samples = 0

            with tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{config.epochs}",
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

            # Test the model every N epochs if specified
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

            # Update overall training progress bar with both training and test accuracy
            postfix_dict = {
                "epoch": f"{epoch + 1}/{config.epochs}",
                "loss": f"{avg_loss:.4f}",
                "train_acc": f"{latest_train_acc:.2f}%",
            }

            # Add test accuracy if we have test results
            if latest_test_acc > 0:
                postfix_dict["test_acc"] = f"{latest_test_acc:.2f}%"

            training_pbar.set_postfix(postfix_dict)

            # Save checkpoint after each epoch
            save_checkpoint(
                net,
                epoch,
                epoch_losses,
                epoch_accuracies,
                test_accuracies,
                test_losses,
                config,
                output_dir,
                device,
                input_size,
                num_classes,
                feature_types,
                num_features,
                dataset_metadata,
            )

    except KeyboardInterrupt:
        print(f"\n\nTraining interrupted at epoch {epoch + 1}")

    # 4. Save the model
    print(f"Saving trained model to {model_save_path}")
    torch.save(net.state_dict(), model_save_path)

    # 5. Save loss graph
    if test_accuracies:
        plt.figure(figsize=(24, 4))
        # Plotting code simplified for headless
        # ... (visualization logic preserved but simplified)

        plt.subplot(1, 4, 1)
        plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, "b-", linewidth=2)
        plt.title("Training Loss")

        plt.subplot(1, 4, 2)
        plt.plot(range(1, len(epoch_accuracies) + 1), epoch_accuracies, "r-", linewidth=2)
        plt.title("Training Accuracy")

        plt.subplot(1, 4, 3)
        test_epochs = [i * config.test_every for i in range(1, len(test_accuracies) + 1)]
        plt.plot(test_epochs, test_accuracies, "g-", linewidth=2, marker="o")
        plt.title("Test Accuracy")

        plt.subplot(1, 4, 4)
        if "test_losses" in locals() and test_losses:
            plt.plot(test_epochs, test_losses, "m-", linewidth=2, marker="o")
        plt.title("Test Loss")
    else:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, "b-", linewidth=2)
        plt.title("Training Loss")

        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(epoch_accuracies) + 1), epoch_accuracies, "r-", linewidth=2)
        plt.title("Training Accuracy")

    plt.tight_layout()
    graph_save_path = os.path.join(output_dir, "loss_graph.png")
    plt.savefig(graph_save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved loss graph to {graph_save_path}")

    # 7. Final test the model
    print("\n" + "=" * 50)
    print("FINAL TEST RESULTS")
    print("=" * 50)

    final_test_acc, final_test_loss = test_model(
        net, test_loader, device, criterion
    )

    # 6. Save training configuration
    result_config = {
        "model_type": "SNN",
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "test_every": config.test_every,
        "device": str(device),
        "input_size": input_size,
        "hidden_size": SNN_HIDDEN_SIZE,
        "output_size": num_classes,
        "optimizer": "Adam",
        "training_timestamp": datetime.datetime.now().isoformat(),
        "final_train_loss": epoch_losses[-1] if epoch_losses else None,
        "final_train_accuracy": epoch_accuracies[-1] if epoch_accuracies else None,
        "final_test_accuracy": final_test_acc,
        "final_test_loss": final_test_loss,
        "total_train_samples": len(train_dataset),
        "total_test_samples": len(test_dataset),
        "feature_types": feature_types,
        "num_features": num_features,
        "dataset_metadata": dataset_metadata,
    }

    config_save_path = os.path.join(output_dir, "training_results.json")
    with open(config_save_path, "w") as f:
        json.dump(result_config, f, indent=2)
    print(f"Saved training results to {config_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
        train_cfg = TrainingConfig(**cfg_dict['training'])

    train_model(train_cfg, args.input_dir, args.output_dir)
