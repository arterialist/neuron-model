"""Model evaluation utilities."""

from typing import Any

import torch


def test_model(
    net: Any,
    test_loader: Any,
    device: torch.device,
    criterion: torch.nn.Module | None = None,
    epoch: int | None = None,
) -> tuple[float, float | None]:
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

            spk_rec = torch.stack(spk_rec, dim=0)

            if criterion is not None:
                loss = criterion(spk_rec.sum(0), labels)
                total_loss += loss.item()
                num_batches += 1

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
