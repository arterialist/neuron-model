import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import json
import datetime
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
import sys


# ==========================================
# 1. Time-Series Fusion Dataset
# ==========================================
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

        # Synchronize Time (Trim to shortest)
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
    return padded_data, torch.stack(labels)


# ==========================================
# 2. The SNN Fusion Model (With Input Norm)
# ==========================================
class FusionSNN(nn.Module):
    def __init__(self, input_dim, num_classes=10, beta=0.9):
        super().__init__()
        self.beta = beta
        spike_grad = surrogate.fast_sigmoid()

        # --- THE FIX: Time-Distributed Batch Norm ---
        # Squashes the massive input current (24.0) down to (0.0)
        self.bn_input = nn.BatchNorm1d(input_dim)

        self.fc1 = nn.Linear(input_dim, 1024)
        self.lif1 = snn.Leaky(beta=self.beta, spike_grad=spike_grad)

        self.fc2 = nn.Linear(1024, 512)
        self.lif2 = snn.Leaky(beta=self.beta, spike_grad=spike_grad)

        self.fc3 = nn.Linear(512, num_classes)
        self.lif3 = snn.Leaky(beta=self.beta, spike_grad=spike_grad, output=True)

    def forward(self, x):
        # x: [Batch, Time, Features]

        # 1. Apply Batch Norm across (Batch * Time)
        # This treats every tick as an independent sample for normalization
        b, t, f = x.shape
        x_flat = x.reshape(b * t, f)
        x_norm = self.bn_input(x_flat)
        x = x_norm.reshape(b, t, f)

        # Initialize Membranes
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        spk3_rec = []

        # Time Loop
        for step in range(t):
            current_input = x[:, step, :]

            cur1 = self.fc1(current_input)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)

            spk3_rec.append(spk3)

        return torch.stack(spk3_rec, dim=0)


# ==========================================
# 3. Checkpoint Saving Function
# ==========================================
def save_checkpoint(
    model,
    optimizer,
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
    run_dir_name,
    train_ds,
):
    """Save a training checkpoint for the current epoch."""
    # Save checkpoint model
    checkpoint_model_path = model_save_path.replace(
        ".pth", f"_checkpoint_epoch_{epoch + 1}.pth"
    )
    torch.save(model.state_dict(), checkpoint_model_path)

    # Save checkpoint configuration
    checkpoint_config = {
        "dataset_dir_a": args.dir_a,
        "dataset_dir_b": args.dir_b,
        "dataset_dir_c": args.dir_c,
        "dataset_dir_d": args.dir_d,
        "dataset_basename": dataset_basename,
        "run_dir_name": run_dir_name,
        "run_dir_path": run_dir_path,
        "model_save_path": checkpoint_model_path,
        "load_model_path": None,
        "output_dir": "fusion_models",
        "epochs": args.epochs,
        "completed_epochs": epoch + 1,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "device": str(device),
        "input_size": input_size,
        "hidden_size": 512,
        "output_size": num_classes,
        "optimizer": "Adam",
        "optimizer_betas": [0.9, 0.999],
        "loss_function": "ce_rate_loss",
        "neuron_type": "Leaky",
        "beta": 0.9,
        "training_timestamp": datetime.datetime.now().isoformat(),
        "interrupted": False,
        "checkpoint_epoch": epoch + 1,
        "epoch_losses": epoch_losses,
        "epoch_accuracies": epoch_accuracies,
        "test_accuracies": test_accuracies,
        "test_losses": test_losses,
        "feature_types": ["firings", "avg_S"],
        "num_features": 2,
        "dataset_metadata": {
            "train_samples": len(train_ds),
            "feature_dimensions": {
                "dim_a": train_ds.dim_a,
                "dim_b": train_ds.dim_b,
                "dim_c": train_ds.dim_c,
                "dim_d": train_ds.dim_d,
                "total_dim": train_ds.total_dim,
            },
        },
    }

    checkpoint_config_path = model_save_path.replace(
        ".pth", f"_checkpoint_epoch_{epoch + 1}_config.json"
    )
    with open(checkpoint_config_path, "w") as f:
        json.dump(checkpoint_config, f, indent=2)

    return checkpoint_model_path, checkpoint_config_path


# ==========================================
# 4. Training Loop
# ==========================================
def train_fusion(args):
    import datetime

    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {DEVICE}")

    # Create a unique directory for this training run
    dataset_basename = os.path.basename(
        os.path.normpath(args.dir_a)
    )  # Use first dataset name as base
    run_dir_name = (
        f"{args.prefix}_e{args.epochs}_lr{args.learning_rate}_b{args.batch_size}"
    )
    run_dir_path = os.path.join("fusion_models", run_dir_name)
    os.makedirs(run_dir_path, exist_ok=True)
    print(f"Created training run directory: {run_dir_path}")

    # Construct model save path within the run directory
    model_filename = "model.pth"
    model_save_path = os.path.join(run_dir_path, model_filename)

    train_ds = FusionDataset(
        args.dir_a, args.dir_b, args.dir_c, args.dir_d, split="train"
    )
    test_ds = FusionDataset(
        args.dir_a, args.dir_b, args.dir_c, args.dir_d, split="test"
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    model = FusionSNN(input_dim=train_ds.total_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = SF.ce_rate_loss()

    print(f"Training SNN Fusion on {DEVICE} | Input Dim: {train_ds.total_dim}")

    # Initialize training state
    epoch_losses = []
    epoch_accuracies = []
    test_accuracies = []
    test_losses = []

    # Create overall training progress bar
    training_pbar = tqdm(range(args.epochs), desc="Training Progress", position=0)

    for epoch in training_pbar:
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        with tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
            position=1,
            leave=False,
        ) as epoch_pbar:
            for data, target in epoch_pbar:
                data, target = data.to(DEVICE), target.to(DEVICE)

                optimizer.zero_grad()
                spk_rec = model(data)
                loss = loss_fn(spk_rec, target)
                loss.backward()
                optimizer.step()

                _, idx = spk_rec.sum(dim=0).max(1)
                correct += idx.eq(target).sum().item()
                total += target.size(0)
                train_loss += loss.item()

                epoch_pbar.set_postfix(acc=correct / total * 100, loss=loss.item())

        # Calculate training metrics
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * correct / total
        epoch_losses.append(avg_train_loss)
        epoch_accuracies.append(train_acc)

        # Eval
        model.eval()
        test_correct = 0
        test_total = 0
        test_loss_total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                spk_rec = model(data)
                loss = loss_fn(spk_rec, target)
                test_loss_total += loss.item()

                _, idx = spk_rec.sum(dim=0).max(1)
                test_correct += idx.eq(target).sum().item()
                test_total += target.size(0)

        test_acc = 100 * test_correct / test_total
        avg_test_loss = test_loss_total / len(test_loader)
        test_accuracies.append(test_acc)
        test_losses.append(avg_test_loss)

        print(f"Test Acc: {test_acc:.2f}% | Test Loss: {avg_test_loss:.4f}")

        # Update overall training progress bar
        training_pbar.set_postfix(
            epoch=f"{epoch + 1}/{args.epochs}",
            train_acc=f"{train_acc:.2f}%",
            test_acc=f"{test_acc:.2f}%",
            test_loss=f"{avg_test_loss:.4f}",
        )

        # Save checkpoint after each epoch
        save_checkpoint(
            model,
            optimizer,
            epoch,
            epoch_losses,
            epoch_accuracies,
            test_accuracies,
            test_losses,
            args,
            dataset_basename,
            run_dir_path,
            model_save_path,
            DEVICE,
            train_ds.total_dim,
            10,  # num_classes
            run_dir_name,
            train_ds,
        )

    # Save final model
    print(f"Saving trained model to {model_save_path}")
    torch.save(model.state_dict(), model_save_path)

    # Save training configuration
    config = {
        "model_type": "FusionSNN",
        "dataset_dirs": {
            "dir_a": args.dir_a,
            "dir_b": args.dir_b,
            "dir_c": args.dir_c,
            "dir_d": args.dir_d,
        },
        "dataset_basename": dataset_basename,
        "run_dir_name": run_dir_name,
        "run_dir_path": run_dir_path,
        "model_save_path": model_save_path,
        "output_dir": "fusion_models",
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "device": str(DEVICE),
        "input_size": train_ds.total_dim,
        "hidden_size": 512,
        "output_size": 10,
        "optimizer": "Adam",
        "optimizer_betas": [0.9, 0.999],
        "loss_function": "ce_rate_loss",
        "neuron_type": "Leaky",
        "beta": 0.9,
        "training_timestamp": datetime.datetime.now().isoformat(),
        "final_train_loss": epoch_losses[-1] if epoch_losses else None,
        "final_train_accuracy": epoch_accuracies[-1] if epoch_accuracies else None,
        "final_test_accuracy": test_accuracies[-1] if test_accuracies else None,
        "final_test_loss": test_losses[-1] if test_losses else None,
        "test_accuracies": test_accuracies,
        "test_losses": test_losses,
        "epoch_losses": epoch_losses,
        "epoch_accuracies": epoch_accuracies,
        "total_train_samples": len(train_ds),
        "total_test_samples": len(test_ds),
        "feature_types": ["firings", "avg_S"],
        "num_features": 2,
        "dataset_metadata": {
            "feature_dimensions": {
                "dim_a": train_ds.dim_a,
                "dim_b": train_ds.dim_b,
                "dim_c": train_ds.dim_c,
                "dim_d": train_ds.dim_d,
                "total_dim": train_ds.total_dim,
            }
        },
    }

    config_save_path = model_save_path.replace(".pth", "_config.json")
    with open(config_save_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved training configuration to {config_save_path}")

    # Save training history plots
    if epoch_losses and test_accuracies:
        plt.figure(figsize=(16, 8))

        plt.subplot(2, 2, 1)
        plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, "b-", linewidth=2)
        plt.title("Training Loss Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 2)
        plt.plot(
            range(1, len(epoch_accuracies) + 1), epoch_accuracies, "r-", linewidth=2
        )
        plt.title("Training Accuracy Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 3)
        plt.plot(
            range(1, len(test_accuracies) + 1),
            test_accuracies,
            "g-",
            linewidth=2,
            marker="o",
        )
        plt.title("Test Accuracy Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 4)
        plt.plot(
            range(1, len(test_losses) + 1), test_losses, "m-", linewidth=2, marker="o"
        )
        plt.title("Test Loss Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        graph_save_path = model_save_path.replace(".pth", "_loss_graph.png")
        plt.savefig(graph_save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved loss graph to {graph_save_path}")

    print("\nFinal Results:")
    print(
        f"  Final Train Accuracy: {epoch_accuracies[-1]:.2f}%"
        if epoch_accuracies
        else "N/A"
    )
    print(
        f"  Final Test Accuracy: {test_accuracies[-1]:.2f}%"
        if test_accuracies
        else "N/A"
    )
    print(
        f"  Best Test Accuracy: {max(test_accuracies):.2f}%"
        if test_accuracies
        else "N/A"
    )
    print(f"  Best Test Loss: {min(test_losses):.4f}" if test_losses else "N/A")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, required=True)
    parser.add_argument("--dir_a", type=str, required=True)
    parser.add_argument("--dir_b", type=str, required=True)
    parser.add_argument("--dir_c", type=str, required=True)
    parser.add_argument("--dir_d", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    train_fusion(args)
