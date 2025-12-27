import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import json

import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction="mean"):
        """
        Args:
            alpha (float): Weighting factor for the rare class (default: 1.0).
            gamma (float): Focusing parameter (default: 2.0).
                           Higher gamma = harder focus on difficult examples.
            reduction (str): 'mean', 'sum', or 'none'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 1. Calculate standard Cross Entropy Loss (element-wise)
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")

        # 2. Calculate the probability of the true class (pt)
        # ce_loss = -log(pt)  ->  pt = exp(-ce_loss)
        pt = torch.exp(-ce_loss)

        # 3. Calculate the Focal Term: (1 - pt)^gamma
        # If pt is high (easy example), this term goes to 0.
        # If pt is low (hard example), this term stays high.
        focal_term = (1 - pt) ** self.gamma

        # 4. Combine
        loss = self.alpha * focal_term * ce_loss

        # 5. Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


# ==========================================
# 1. The Fusion Dataset Class
# ==========================================
class FusionDataset(Dataset):
    """
    Loads 4 separate dataset files (from 4 different networks)
    and concatenates their features into a single vector.
    """

    def __init__(self, dir_a, dir_b, dir_c, dir_d, split="train"):
        self.split = split

        # Load Data A (Texture / 125N)
        self.data_a = torch.load(os.path.join(dir_a, f"{split}_data.pt"))
        self.labels_a = torch.load(os.path.join(dir_a, f"{split}_labels.pt"))

        # Load Data B (Geometry / 659N)
        self.data_b = torch.load(os.path.join(dir_b, f"{split}_data.pt"))
        self.labels_b = torch.load(os.path.join(dir_b, f"{split}_labels.pt"))

        # Load Data C (Color / 325N)
        self.data_c = torch.load(os.path.join(dir_c, f"{split}_data.pt"))
        self.labels_c = torch.load(os.path.join(dir_c, f"{split}_labels.pt"))

        # Load Data D (4th Network)
        self.data_d = torch.load(os.path.join(dir_d, f"{split}_data.pt"))
        self.labels_d = torch.load(os.path.join(dir_d, f"{split}_labels.pt"))

        # Safety Check: Ensure datasets are aligned
        assert (
            len(self.labels_a)
            == len(self.labels_b)
            == len(self.labels_c)
            == len(self.labels_d)
        ), "Dataset sizes mismatch!"

        # Check label alignment between all pairs (first 50 labels for better confidence)
        mismatches = []
        if not torch.equal(self.labels_a[:50], self.labels_b[:50]):
            mismatches.append("A vs B")
        if not torch.equal(self.labels_a[:50], self.labels_c[:50]):
            mismatches.append("A vs C")
        if not torch.equal(self.labels_a[:50], self.labels_d[:50]):
            mismatches.append("A vs D")
        if not torch.equal(self.labels_b[:50], self.labels_c[:50]):
            mismatches.append("B vs C")
        if not torch.equal(self.labels_b[:50], self.labels_d[:50]):
            mismatches.append("B vs D")
        if not torch.equal(self.labels_c[:50], self.labels_d[:50]):
            mismatches.append("C vs D")

        if mismatches:
            print(f"WARNING: Label mismatches found in: {', '.join(mismatches)}")
            print("First 10 labels comparison:")
            print(f"  Dataset A: {self.labels_a[:10].tolist()}")
            print(f"  Dataset B: {self.labels_b[:10].tolist()}")
            print(f"  Dataset C: {self.labels_c[:10].tolist()}")
            print(f"  Dataset D: {self.labels_d[:10].tolist()}")
            print("This indicates different train/test splits or data ordering.")
            print(
                "For fusion to work properly, all networks must be trained on the SAME data split."
            )
        else:
            print("✓ All dataset labels are aligned - good for fusion!")

        # Calculate Dimensions (total elements when flattened)
        self.dim_a = self.data_a[0].numel()
        self.dim_b = self.data_b[0].numel()
        self.dim_c = self.data_c[0].numel()
        self.dim_d = self.data_d[0].numel()

        self.total_dim = self.dim_a + self.dim_b + self.dim_c + self.dim_d
        print(
            f"[{split.upper()}] Feature Dims: A={self.dim_a} + B={self.dim_b} + C={self.dim_c} + D={self.dim_d} = {self.total_dim}"
        )

    def __len__(self):
        return len(self.labels_a)

    def __getitem__(self, idx):
        # Flatten and Concatenate
        # Inputs might be (Neurons, Ticks) or flattened. We assume flattened feature vectors here.
        # If your prepare_activity_data outputs [Neurons, Features], we flatten to [Neurons*Features]

        vec_a = self.data_a[idx].float().view(-1)
        vec_b = self.data_b[idx].float().view(-1)
        vec_c = self.data_c[idx].float().view(-1)
        vec_d = self.data_d[idx].float().view(-1)

        fusion_vector = torch.cat([vec_a, vec_b, vec_c, vec_d], dim=0)
        label = self.labels_a[idx]

        return fusion_vector, label


# ==========================================
# 2. The Fusion Classifier (PFC)
# ==========================================
class FusionClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=10):
        super().__init__()

        # A robust MLP to integrate the signals
        self.net = nn.Sequential(
            # 1. Input Normalization (Crucial for Rod Cell 0.33v vs Cone 0.6v)
            nn.BatchNorm1d(input_dim),
            # 2. Layer 1: High Capacity (512 Neurons)
            # Needed to separate "Fuzzy Deer" from "Fuzzy Dog"
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),  # High dropout to prevent arrogance
            # 3. Layer 2: Processing (256 Neurons)
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            # 4. Output: The Logits
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ==========================================
# 3. Training Loop
# ==========================================
def train_fusion(args):
    # --- CONFIGURATION ---
    # IMPORTANT: These datasets must be generated from the SAME data split for labels to align!
    # For proper fusion, generate datasets from the same data split.

    DIR_A = args.dir_a
    DIR_B = args.dir_b
    DIR_C = args.dir_c
    DIR_D = args.dir_d

    BATCH_SIZE = args.batch_size
    LR = args.learning_rate
    EPOCHS = args.epochs
    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # Create model save directory
    model_dir = f"fusion_models/{args.prefix}_bs{args.batch_size}_lr{args.learning_rate}_epochs{args.epochs}"
    os.makedirs(model_dir, exist_ok=True)

    # Prepare metadata
    metadata = {
        "experiment_name": args.prefix,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "device": DEVICE,
        "data_dirs": {
            "dir_a": args.dir_a,
            "dir_b": args.dir_b,
            "dir_c": args.dir_c,
            "dir_d": args.dir_d,
        },
        "model_dir": model_dir,
        "training_start_time": None,  # Will be set later
        "best_accuracy": None,
        "best_loss": None,
        "total_training_time": None,
        "feature_dimensions": None,  # Will be set after dataset loading
    }

    print(f"Initializing Fusion Training on {DEVICE}...")
    print(f"Models will be saved to: {model_dir}")

    # Load Datasets
    train_ds = FusionDataset(DIR_A, DIR_B, DIR_C, DIR_D, split="train")
    test_ds = FusionDataset(DIR_A, DIR_B, DIR_C, DIR_D, split="test")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Update metadata with feature dimensions
    metadata["feature_dimensions"] = {
        "dim_a": train_ds.dim_a,
        "dim_b": train_ds.dim_b,
        "dim_c": train_ds.dim_c,
        "dim_d": train_ds.dim_d,
        "total_dim": train_ds.total_dim,
    }

    # Initialize Model
    model = FusionClassifier(input_dim=train_ds.total_dim).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    history = {"acc": [], "loss": []}

    print(f"\nStarting Training (Input Size: {train_ds.total_dim})")
    print("-" * 60)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for data, target in pbar:
            data, target = data.to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            output = model(data)
            # 1. The Accuracy Loss (Standard)
            # Note: Use standard CrossEntropyLoss for now. Avoid FocalLoss unless necessary.
            main_loss = criterion(output, target)
            # 2. THE VOLTAGE REGULATOR (Activity Regularization)
            # This forces logits to stay small (e.g., -10 to +10).
            # If logits explode to 300, this term becomes massive (300^2 = 90,000),
            # forcing the optimizer to lower the weights immediately.
            lambda_reg = 0.01  # Strength of the regulator
            logit_loss = torch.mean(output**2)

            # 3. Combine
            total_loss = main_loss + (lambda_reg * logit_loss)
            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += len(target)

            pbar.set_postfix(loss=total_loss.item())

        # Evaluation
        model.eval()
        test_correct = 0
        test_total = 0
        test_loss = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                test_correct += pred.eq(target.view_as(pred)).sum().item()
                test_total += len(target)

        acc = 100.0 * test_correct / test_total
        avg_test_loss = test_loss / len(test_loader)

        history["acc"].append(acc)
        history["loss"].append(avg_test_loss)

        scheduler.step()

        print(
            f"Epoch {epoch + 1}: Test Acc: {acc:.2f}% | Test Loss: {avg_test_loss:.4f}"
        )

        # Save best (lowest loss)
        if avg_test_loss <= min(history["loss"]):
            model_path = os.path.join(model_dir, "best_fusion_model.pth")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_test_loss,
                    "accuracy": acc,
                    "history": history,
                },
                model_path,
            )
            print(
                f">>> New Best! Saved to {model_path}. (Loss: {avg_test_loss:.4f}, Acc: {acc:.2f}%)"
            )

    best_acc = max(history["acc"])
    best_loss = min(history["loss"])

    # Update and save metadata
    metadata["best_accuracy"] = best_acc
    metadata["best_loss"] = best_loss

    metadata_path = os.path.join(model_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nFinal Results:")
    print(f"  Best Accuracy: {best_acc:.2f}%")
    print(f"  Best Loss: {best_loss:.4f}")
    print(f"  Model saved to: {model_dir}/best_fusion_model.pth")
    print(f"  Metadata saved to: {metadata_path}")

    # Plot training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history["acc"])
    plt.title("Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")

    plt.subplot(1, 2, 2)
    plt.plot(history["loss"])
    plt.title("Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.tight_layout()
    plot_path = os.path.join(model_dir, "training_history.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"  Training plot saved to: {plot_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train fusion classifier with 4 neural networks"
    )

    # Experiment naming
    parser.add_argument(
        "--prefix",
        type=str,
        required=True,
        help="Experiment name prefix for model saving",
    )

    # Network data directories
    parser.add_argument(
        "--dir_a",
        type=str,
        required=True,
        help="Path to first network's prepared data directory",
    )
    parser.add_argument(
        "--dir_b",
        type=str,
        required=True,
        help="Path to second network's prepared data directory",
    )
    parser.add_argument(
        "--dir_c",
        type=str,
        required=True,
        help="Path to third network's prepared data directory",
    )
    parser.add_argument(
        "--dir_d",
        type=str,
        required=True,
        help="Path to fourth network's prepared data directory",
    )

    # Training parameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training (default: 64)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of training epochs (default: 30)"
    )

    args = parser.parse_args()
    train_fusion(args)
