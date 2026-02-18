"""Configuration for activity data preparation."""

from dataclasses import dataclass


@dataclass
class PrepareConfig:
    """Configuration for prepare_activity_data workflow."""

    input_file: str
    output_dir: str
    feature_types: list[str]
    train_split: float
    use_streaming: bool
    scaler: str
    scale_eps: float
    max_ticks: int | None
    max_samples: int | None
    legacy_json: bool

    @property
    def structured_output_dir(self) -> str:
        """Output directory with input basename and feature suffix."""
        import os
        input_basename = os.path.splitext(os.path.basename(self.input_file))[0]
        feature_suffix = "_".join(self.feature_types)
        return os.path.join(self.output_dir, f"{input_basename}_{feature_suffix}")
