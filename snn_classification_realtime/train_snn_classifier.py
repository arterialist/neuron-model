"""Train an SNN classifier on network activity data.

Entry point for the SNN classifier training CLI.
All logic has been refactored into the snn_trainer package.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from snn_classification_realtime.snn_trainer import SNNClassifier, ActivityDataset, collate_fn
from snn_classification_realtime.snn_trainer.config import TrainConfig
from snn_classification_realtime.snn_trainer.run import run_train

__all__ = ["main", "SNNClassifier", "ActivityDataset", "collate_fn"]


def main() -> None:
    """Entry point for the train_snn_classifier CLI."""
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
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size. (default: 32)",
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

    config = TrainConfig(
        dataset_dir=args.dataset_dir,
        model_save_path=args.model_save_path,
        load_model_path=args.load_model_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        test_every=args.test_every,
        device=args.device,
    )

    run_train(config)


if __name__ == "__main__":
    main()
