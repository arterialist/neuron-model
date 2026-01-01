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

SNN_HIDDEN_SIZE = 512


class FusionDataset(Dataset):
    def __init__(self, dir_a, dir_b, dir_c, dir_d, split="train"):
        self.split = split

        # Load Time-Series Data
        self.data_a = torch.load(os.path.join(dir_a, f"{split}_data.pt"))
        self.data_b = torch.load(os.path.join(dir_b, f"{split}_data.pt"))
        self.data_c = torch.load(os.path.join(dir_c, f"{split}_data.pt"))
        self.data_d = torch.load(os.path.join(dir_d, f"{split}_data.pt"))

        self.labels = torch.load(os.path.join(dir_a, f"{split}_labels.pt"))

        # Dimensions
        self.dim_a = self.data_a[0].shape[1]
        self.dim_b = self.data_b[0].shape[1]
        self.dim_c = self.data_c[0].shape[1]
        self.dim_d = self.data_d[0].shape[1]
        self.total_dim = self.dim_a + self.dim_b + self.dim_c + self.dim_d

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        ts_a = self.data_a[idx].float()
        ts_b = self.data_b[idx].float()
        ts_c = self.data_c[idx].float()
        ts_d = self.data_d[idx].float()

        # Synchronize Time (Trim to shortest among the 4 inputs)
        min_time = min(ts_a.size(0), ts_b.size(0), ts_c.size(0), ts_d.size(0))

        ts_a = ts_a[:min_time]
        ts_b = ts_b[:min_time]
        ts_c = ts_c[:min_time]
        ts_d = ts_d[:min_time]

        # Concatenate Features
        fusion_series = torch.cat([ts_a, ts_b, ts_c, ts_d], dim=1)

        return fusion_series, self.labels[idx]


def collate_fn(batch):
    """Pads sequences so they can be stacked in a batch."""
    data, labels = zip(*batch)
    padded_data = nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=0)
    # Ensure labels are stacked correctly
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
            # Record spikes for the entire sequence
            spk_rec = []

            # Process each time step explicitly
            for step in range(data.shape[1]):  # loop over time dimension
                spk5, mem1, mem2, mem3 = net(
                    data[:, step, :], mem1, mem2, mem3
                )
                spk_rec.append(spk5)

            # Stack the recorded spikes
            spk_rec = torch.stack(spk_rec, dim=0)

            # Compute test loss if criterion provided
            if criterion is not None:
                # spk_rec is [Time, Batch, Output], sum over time
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


def save_checkpoint(
    net,
    epoch,
    epoch_losses,
    epoch_accuracies,
    test_accuracies,
    test_losses,
    args,
    dataset_basename,
    run_dir_path,
    model_save_path,
    device,
    input_size,
    num_classes,
    dataset_metadata,
    run_dir_name,
):
    """Save a training checkpoint for the current epoch."""
    # Save checkpoint model
    checkpoint_model_path = model_save_path.replace(
        ".pth", f"_checkpoint_epoch_{epoch + 1}.pth"
    )
    torch.save(net.state_dict(), checkpoint_model_path)

    # Save checkpoint configuration
    checkpoint_config = {
        "dataset_dirs": {
            "dir_a": args.dir_a,
            "dir_b": args.dir_b,
            "dir_c": args.dir_c,
            "dir_d": args.dir_d,
        },
        "dataset_basename": dataset_basename,
        "run_dir_name": run_dir_name,
        "run_dir_path": run_dir_path,
        "model_save_path": checkpoint_model_path,
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
        "interrupted": False,
        "checkpoint_epoch": epoch + 1,
        "epoch_losses": epoch_losses,
        "epoch_accuracies": epoch_accuracies,
        "test_accuracies": test_accuracies,
        "test_losses": test_losses,
        "dataset_metadata": dataset_metadata,
    }

    checkpoint_config_path = model_save_path.replace(
        ".pth", f"_checkpoint_epoch_{epoch + 1}_config.json"
    )
    with open(checkpoint_config_path, "w") as f:
        json.dump(checkpoint_config, f, indent=2)

    return checkpoint_model_path, checkpoint_config_path


class FusionSNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        beta = 0.9  # Neuron membrane potential decay rate

        self.fc1 = nn.Linear(input_size, 1024)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(1024, hidden_size)
        self.lif2 = snn.Leaky(beta=beta)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.lif3 = snn.Leaky(beta=beta)

    def forward(self, x, mem1=None, mem2=None, mem3=None):
        """
        Forward pass for a single time step.
        """
        # Initialize hidden states if not provided
        if mem1 is None:
            mem1 = self.lif1.init_leaky()
        if mem2 is None:
            mem2 = self.lif2.init_leaky()
        if mem3 is None:
            mem3 = self.lif3.init_leaky()

        # Single forward pass
        cur1 = self.fc1(x)
        spk1, mem1 = self.lif1(cur1, mem1)

        cur2 = self.fc2(spk1)
        spk2, mem2 = self.lif2(cur2, mem2)
        
        cur3 = self.fc3(spk2)
        spk3, mem3 = self.lif3(cur3, mem3)

        return spk3, mem1, mem2, mem3


def main():
    parser = argparse.ArgumentParser(
        description="Train a Fusion SNN classifier on 4 network activity sources."
    )
    parser.add_argument(
        "--prefix", type=str, required=True, help="Prefix for run naming"
    )
    parser.add_argument("--dir_a", type=str, required=True)
    parser.add_argument("--dir_b", type=str, required=True)
    parser.add_argument("--dir_c", type=str, required=True)
    parser.add_argument("--dir_d", type=str, required=True)

    parser.add_argument(
        "--model-save-path",
        type=str,
        default="fusion_snn_model.pth",
        help="Path to save the trained model.",
    )
    parser.add_argument(
        "--load-model-path",
        type=str,
        help="(Optional) Path to a pre-trained model to load.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="fusion_models",
        help="Directory to save trained models.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--test-every",
        type=int,
        default=0,
        help="Test the model every N epochs. (default: 0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "mps", "cpu"],
    )
    args = parser.parse_args()

    # Device Setup
    device = torch.device(
        args.device
        if args.device
        else (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    )
    print(f"Using device: {device}")

    # Create directories
    dataset_basename = args.prefix
    run_dir_name = (
        f"{dataset_basename}_e{args.epochs}_lr{args.learning_rate}_b{args.batch_size}"
    )
    run_dir_path = os.path.join(args.output_dir, run_dir_name)
    os.makedirs(run_dir_path, exist_ok=True)
    print(f"Created training run directory: {run_dir_path}")

    # Model path setup
    if args.model_save_path == "fusion_snn_model.pth":
        model_filename = "model.pth"
        model_save_path = os.path.join(run_dir_path, model_filename)
    else:
        model_filename = os.path.basename(args.model_save_path)
        model_save_path = os.path.join(run_dir_path, model_filename)

    print("Loading Fusion Datasets...")
    train_dataset = FusionDataset(
        args.dir_a, args.dir_b, args.dir_c, args.dir_d, split="train"
    )
    test_dataset = FusionDataset(
        args.dir_a, args.dir_b, args.dir_c, args.dir_d, split="test"
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    dataset_metadata = {
        "dim_a": train_dataset.dim_a,
        "dim_b": train_dataset.dim_b,
        "dim_c": train_dataset.dim_c,
        "dim_d": train_dataset.dim_d,
        "total_dim": train_dataset.total_dim,
    }

    input_size = train_dataset.total_dim

    all_labels = torch.cat([train_dataset.labels, test_dataset.labels])
    num_classes = len(torch.unique(all_labels))

    print(f"Input Feature Size: {input_size} (Fused)")
    print(f"Number of classes: {num_classes}")
    print("Architecture: 4-Layer SNN (Matching train_snn_classifier.py)")

    net = FusionSNNClassifier(
        input_size=input_size, hidden_size=SNN_HIDDEN_SIZE, output_size=num_classes
    ).to(device)

    interrupted_state = None
    if args.load_model_path:
        print(f"Loading pre-trained model from {args.load_model_path}")
        net.load_state_dict(torch.load(args.load_model_path))

        config_path = args.load_model_path.replace(".pth", "_config.json")
        interrupted_state = load_interrupted_state(config_path)

        if interrupted_state:
            print(
                f"Detected interruption at epoch {interrupted_state['interruption_epoch']}"
            )
            print("Resuming...")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        net.parameters(), lr=args.learning_rate, betas=(0.9, 0.999)
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.75)

    if interrupted_state:
        epoch_losses = interrupted_state.get("epoch_losses", [])
        epoch_accuracies = interrupted_state.get("epoch_accuracies", [])
        test_accuracies = interrupted_state.get("test_accuracies", [])
        test_losses = interrupted_state.get("test_losses", [])
        start_epoch = interrupted_state.get("completed_epochs", 0)
        # Initialize best test loss tracking
        best_test_loss = min(test_losses) if test_losses else float("inf")
        best_test_loss_epoch = interrupted_state.get("best_test_loss_epoch", None)
    else:
        epoch_losses = []
        epoch_accuracies = []
        test_accuracies = []
        test_losses = []
        start_epoch = 0
        # Initialize best test loss tracking
        best_test_loss = float("inf")
        best_test_loss_epoch = None

    training_pbar = tqdm(
        range(start_epoch, args.epochs), desc="Training Progress", position=0
    )

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
                desc=f"Epoch {epoch + 1}/{args.epochs}",
                position=1,
                leave=False,
            ) as epoch_pbar:
                for batch_idx, (data, labels) in enumerate(epoch_pbar):
                    data = data.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    # Initialize membranes
                    mem1 = net.lif1.init_leaky()
                    mem2 = net.lif2.init_leaky()
                    mem3 = net.lif3.init_leaky()

                    spk_rec = []

                    for step in range(data.shape[1]):
                        spk3, mem1, mem2, mem3 = net(
                            data[:, step, :], mem1, mem2, mem3
                        )
                        spk_rec.append(spk3)

                    spk_rec = torch.stack(spk_rec, dim=1)

                    loss = criterion(spk_rec.sum(dim=1), labels)

                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

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

            if args.test_every > 0 and (epoch + 1) % args.test_every == 0:
                with tqdm([0], desc="Testing", position=2, leave=False) as test_pbar:
                    test_pbar.set_description("Testing model...")
                    test_acc, test_loss = test_model(
                        net, test_loader, device, criterion, epoch + 1
                    )
                    test_accuracies.append(test_acc)
                    test_losses.append(test_loss)
                    latest_test_acc = test_acc

                    # Track best test loss and epoch
                    if test_loss is not None and test_loss < best_test_loss:
                        best_test_loss = test_loss
                        best_test_loss_epoch = epoch + 1

                    test_pbar.update(1)

            postfix_dict = {
                "epoch": f"{epoch + 1}/{args.epochs}",
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
                args,
                dataset_basename,
                run_dir_path,
                model_save_path,
                device,
                input_size,
                num_classes,
                dataset_metadata,
                run_dir_name,
            )

    except KeyboardInterrupt:
        print(f"\n\nTraining interrupted at epoch {epoch + 1}")
        print("Saving intermediate results...")

        intermediate_model_path = model_save_path.replace(
            ".pth", f"_interrupted_epoch_{epoch + 1}.pth"
        )
        torch.save(net.state_dict(), intermediate_model_path)

        intermediate_config = {
            "dataset_dirs": {
                "dir_a": args.dir_a,
                "dir_b": args.dir_b,
                "dir_c": args.dir_c,
                "dir_d": args.dir_d,
            },
            "run_dir_name": run_dir_name,
            "interrupted": True,
            "interruption_epoch": epoch + 1,
            "epoch_losses": epoch_losses,
            "epoch_accuracies": epoch_accuracies,
            "test_accuracies": test_accuracies,
            "test_losses": test_losses,
            "best_test_loss": best_test_loss
            if best_test_loss != float("inf")
            else None,
            "best_test_loss_epoch": best_test_loss_epoch,
            "completed_epochs": epoch + 1,
        }

        intermediate_config_path = model_save_path.replace(
            ".pth", f"_interrupted_epoch_{epoch + 1}_config.json"
        )
        with open(intermediate_config_path, "w") as f:
            json.dump(intermediate_config, f, indent=2)

        if epoch_losses:
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
                ".pth", f"_interrupted_epoch_{epoch + 1}_loss_graph.png"
            )
            plt.savefig(intermediate_graph_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Saved interrupted loss graph to {intermediate_graph_path}")

        print("Saved interrupted state.")
        return

    print(f"Saving trained model to {model_save_path}")
    torch.save(net.state_dict(), model_save_path)

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
        if test_losses:
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

    graph_save_path = model_save_path.replace(".pth", "_loss_graph.png")
    plt.tight_layout()
    plt.savefig(graph_save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved graph to {graph_save_path}")

    print("\n" + "=" * 50)
    print("FINAL TEST RESULTS")
    print("=" * 50)

    final_test_acc, final_test_loss = test_model(net, test_loader, device, criterion)
    print(f"Final Test Acc: {final_test_acc:.2f}%")
    print(f"Final Test Loss: {final_test_loss:.4f}")

    config = {
        "model_type": "FusionSNN",
        "epochs": args.epochs,
        "final_train_loss": epoch_losses[-1] if epoch_losses else None,
        "final_train_accuracy": epoch_accuracies[-1] if epoch_accuracies else None,
        "final_test_accuracy": final_test_acc,
        "final_test_loss": final_test_loss,
        "best_test_loss": best_test_loss if best_test_loss != float("inf") else None,
        "best_test_loss_epoch": best_test_loss_epoch,
        "dataset_metadata": dataset_metadata,
    }
    config_save_path = model_save_path.replace(".pth", "_config.json")
    with open(config_save_path, "w") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    main()
