"""Training loss and accuracy plotting."""

from typing import Any

import matplotlib.pyplot as plt


def save_loss_graph(
    epoch_losses: list[float],
    epoch_accuracies: list[float],
    test_accuracies: list[float],
    test_losses: list[float | None],
    save_path: str,
    test_every: int,
    title_suffix: str = "",
) -> None:
    """Save training/test loss and accuracy plots."""
    if test_accuracies:
        plt.figure(figsize=(24, 4))
        plt.subplot(1, 4, 1)
        plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, "b-", linewidth=2)
        plt.title(f"Training Loss Over Epochs{title_suffix}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 4, 2)
        plt.plot(
            range(1, len(epoch_accuracies) + 1), epoch_accuracies, "r-", linewidth=2
        )
        plt.title(f"Training Accuracy Over Epochs{title_suffix}")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 4, 3)
        test_epochs = [i * test_every for i in range(1, len(test_accuracies) + 1)]
        plt.plot(test_epochs, test_accuracies, "g-", linewidth=2, marker="o")
        plt.title(f"Test Accuracy Over Epochs{title_suffix}")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 4, 4)
        if test_losses:
            plt.plot(test_epochs, test_losses, "m-", linewidth=2, marker="o")
        else:
            plt.plot([], [])
        plt.title(f"Test Loss Over Epochs{title_suffix}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.3)
    else:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, "b-", linewidth=2)
        plt.title(f"Training Loss Over Epochs{title_suffix}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(
            range(1, len(epoch_accuracies) + 1), epoch_accuracies, "r-", linewidth=2
        )
        plt.title(f"Training Accuracy Over Epochs{title_suffix}")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
