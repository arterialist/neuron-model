"""
Classifier trainer step for the pipeline.

Trains SNN classifiers on prepared activity data.
Wraps functionality from snn_classification_realtime/train_snn_classifier.py.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import snntorch as snn
except ImportError:
    snn = None

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline.config import TrainingConfig
from pipeline.steps.base import (
    PipelineStep,
    StepContext,
    StepResult,
    StepStatus,
    Artifact,
    StepRegistry,
)


class ActivityDataset(Dataset):
    """Dataset for activity time-series data."""

    def __init__(self, data: List[torch.Tensor], labels: torch.Tensor):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def collate_fn(batch):
    """Collate function to handle variable length sequences."""
    data, labels = zip(*batch)
    max_len = max(d.shape[0] for d in data)

    padded = []
    for d in data:
        if d.shape[0] < max_len:
            pad = torch.zeros(max_len - d.shape[0], d.shape[1])
            d = torch.cat([d, pad], dim=0)
        padded.append(d)

    return torch.stack(padded), torch.tensor(labels)


class SNNClassifier(nn.Module):
    """Spiking Neural Network classifier."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        beta = 0.9

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=beta) if snn else None
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.lif2 = snn.Leaky(beta=beta) if snn else None
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.lif3 = snn.Leaky(beta=beta, output=True) if snn else None

    def forward(self, x, mem1=None, mem2=None, mem3=None):
        if mem1 is None:
            mem1 = (
                self.lif1.init_leaky()
                if self.lif1
                else torch.zeros(x.shape[0], self.fc1.out_features)
            )
        if mem2 is None:
            mem2 = (
                self.lif2.init_leaky()
                if self.lif2
                else torch.zeros(x.shape[0], self.fc2.out_features)
            )
        if mem3 is None:
            mem3 = (
                self.lif3.init_leaky()
                if self.lif3
                else torch.zeros(x.shape[0], self.fc3.out_features)
            )

        cur1 = self.fc1(x)
        if self.lif1:
            spk1, mem1 = self.lif1(cur1, mem1)
        else:
            spk1 = torch.relu(cur1)

        cur2 = self.fc2(spk1)
        if self.lif2:
            spk2, mem2 = self.lif2(cur2, mem2)
        else:
            spk2 = torch.relu(cur2)

        cur3 = self.fc3(spk2)
        if self.lif3:
            spk3, mem3 = self.lif3(cur3, mem3)
        else:
            spk3 = cur3

        return spk3, mem1, mem2, mem3


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for data, labels in train_loader:
        data, labels = data.to(device), labels.to(device)
        batch_size, T, features = data.shape

        optimizer.zero_grad()

        # Process sequence
        mem1, mem2, mem3 = None, None, None
        spk_rec = []

        for t in range(T):
            spk, mem1, mem2, mem3 = model(data[:, t], mem1, mem2, mem3)
            spk_rec.append(spk)

        # Sum spikes over time
        spk_sum = torch.stack(spk_rec).sum(dim=0)

        loss = criterion(spk_sum, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def test_model(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Test the model and return accuracy and loss."""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            batch_size, T, features = data.shape

            mem1, mem2, mem3 = None, None, None
            spk_rec = []

            for t in range(T):
                spk, mem1, mem2, mem3 = model(data[:, t], mem1, mem2, mem3)
                spk_rec.append(spk)

            spk_sum = torch.stack(spk_rec).sum(dim=0)

            loss = criterion(spk_sum, labels)
            total_loss += loss.item()

            _, predicted = spk_sum.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total if total > 0 else 0.0
    avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else 0.0

    return accuracy, avg_loss


@StepRegistry.register
class ClassifierTrainerStep(PipelineStep):
    """Pipeline step for training SNN classifiers."""

    @property
    def name(self) -> str:
        return "training"

    @property
    def display_name(self) -> str:
        return "Classifier Training"

    def run(self, context: StepContext) -> StepResult:
        start_time = datetime.now()
        logs: list[str] = []

        try:
            config: TrainingConfig = context.config
            log = context.logger or logging.getLogger(__name__)

            if snn is None:
                raise ImportError("snntorch is required for classifier training")

            # Get prepared data from previous step
            prep_artifacts = context.previous_artifacts.get("data_preparation", [])
            if not prep_artifacts:
                raise ValueError("Data preparation artifacts not found")

            # Find the data directory
            prep_dir = prep_artifacts[0].path.parent

            # Create output directory
            step_dir = context.output_dir / self.name
            step_dir.mkdir(parents=True, exist_ok=True)

            # Load data
            log.info(f"Loading prepared data from {prep_dir}")
            logs.append(f"Loading prepared data from {prep_dir}")

            train_data = torch.load(prep_dir / "train_data.pt")
            train_labels = torch.load(prep_dir / "train_labels.pt")
            test_data = torch.load(prep_dir / "test_data.pt")
            test_labels = torch.load(prep_dir / "test_labels.pt")

            log.info(f"Loaded: train={len(train_data)}, test={len(test_data)}")
            logs.append(
                f"Loaded {len(train_data)} training and {len(test_data)} test samples"
            )

            with open(prep_dir / "metadata.json", "r") as f:
                prep_metadata = json.load(f)

            # Create datasets and loaders
            train_dataset = ActivityDataset(train_data, train_labels)
            test_dataset = ActivityDataset(test_data, test_labels)

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

            # Determine model dimensions
            input_size = prep_metadata["num_features"]

            # Combine all labels to find the true range of classes
            all_labels_combined = torch.cat([train_labels, test_labels])
            num_classes = int(all_labels_combined.max().item()) + 1

            hidden_size = config.hidden_size

            log.info(
                f"Creating model: input={input_size}, hidden={hidden_size}, output={num_classes}"
            )
            logs.append(
                f"Model: input={input_size}, hidden={hidden_size}, output={num_classes}"
            )

            # Create model
            device = torch.device(config.device)
            model = SNNClassifier(input_size, hidden_size, num_classes).to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

            # Training loop
            train_losses = []
            test_accuracies = []
            test_losses = []
            best_accuracy = 0.0

            log.info(f"Training for {config.epochs} epochs")
            logs.append(f"Training for {config.epochs} epochs")

            for epoch in tqdm(range(config.epochs), desc="Training", disable=True):
                train_loss = train_epoch(
                    model, train_loader, criterion, optimizer, device
                )
                train_losses.append(train_loss)

                if (epoch + 1) % config.test_every == 0:
                    test_acc, test_loss = test_model(
                        model, test_loader, criterion, device
                    )
                    test_accuracies.append(test_acc)
                    test_losses.append(test_loss)

                    # Save best model if it's the first test or accuracy improved
                    if test_acc > best_accuracy or epoch == (config.test_every - 1):
                        best_accuracy = test_acc
                        # Save best model
                        torch.save(model.state_dict(), step_dir / "best_model.pth")

                    log.info(
                        f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, test_acc={test_acc:.2f}%"
                    )
                    logs.append(
                        f"Epoch {epoch + 1}/{config.epochs}: Loss={train_loss:.4f}, Acc={test_acc:.2f}%"
                    )

            logs.append(f"Best test accuracy: {best_accuracy:.2f}%")

            # Save final model
            torch.save(model.state_dict(), step_dir / "final_model.pth")

            # Save training history
            history = {
                "train_losses": train_losses,
                "test_accuracies": test_accuracies,
                "test_losses": test_losses,
                "best_accuracy": best_accuracy,
                "config": {
                    "epochs": config.epochs,
                    "batch_size": config.batch_size,
                    "learning_rate": config.learning_rate,
                    "hidden_size": hidden_size,
                    "input_size": input_size,
                    "num_classes": num_classes,
                },
            }

            with open(step_dir / "training_history.json", "w") as f:
                json.dump(history, f, indent=2)

            # Save model config for evaluation
            model_config = {
                "input_size": input_size,
                "hidden_size": hidden_size,
                "num_classes": num_classes,
                "feature_types": prep_metadata.get("feature_types", []),
                "scaler_state": prep_metadata.get("scaler_state", {}),
            }

            with open(step_dir / "model_config.json", "w") as f:
                json.dump(model_config, f, indent=2)

            # Create artifacts
            artifacts = [
                Artifact(
                    name="best_model.pth",
                    path=step_dir / "best_model.pth",
                    artifact_type="model",
                    size_bytes=(step_dir / "best_model.pth").stat().st_size,
                ),
                Artifact(
                    name="final_model.pth",
                    path=step_dir / "final_model.pth",
                    artifact_type="model",
                    size_bytes=(step_dir / "final_model.pth").stat().st_size,
                ),
                Artifact(
                    name="training_history.json",
                    path=step_dir / "training_history.json",
                    artifact_type="metadata",
                    size_bytes=(step_dir / "training_history.json").stat().st_size,
                ),
                Artifact(
                    name="model_config.json",
                    path=step_dir / "model_config.json",
                    artifact_type="metadata",
                    size_bytes=(step_dir / "model_config.json").stat().st_size,
                ),
            ]

            return StepResult(
                status=StepStatus.COMPLETED,
                artifacts=artifacts,
                metrics={
                    "best_accuracy": best_accuracy,
                    "final_train_loss": train_losses[-1] if train_losses else 0.0,
                    "epochs_trained": config.epochs,
                },
                start_time=start_time,
                end_time=datetime.now(),
                logs=logs,
            )

        except Exception as e:
            import traceback

            return StepResult(
                status=StepStatus.FAILED,
                error_message=str(e),
                start_time=start_time,
                end_time=datetime.now(),
                logs=logs + [f"ERROR: {e}", traceback.format_exc()],
            )
