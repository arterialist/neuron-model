"""
Tests for the config module.
"""

import pytest
import yaml

from pipeline.config import (
    PipelineConfig,
    NetworkConfig,
    ActivityRecordingConfig,
    DataPreparationConfig,
    TrainingConfig,
    EvaluationConfig,
    load_config_from_string,
    DatasetType,
    LayerType,
)


class TestConfigParsing:
    """Tests for configuration parsing."""

    def test_load_minimal_config(self, sample_config_yaml):
        """Test loading a minimal valid configuration."""
        config = load_config_from_string(sample_config_yaml)

        assert config.job_name == "test_job"
        assert config.network.source == "build"
        assert config.activity_recording.dataset == DatasetType.MNIST

    def test_network_file_config(self):
        """Test network config with file source."""
        yaml_str = """
job_name: "test"
network:
  source: "file"
  path: "some/network.json"
"""
        config = load_config_from_string(yaml_str)
        assert config.network.source == "file"
        assert config.network.path == "some/network.json"

    def test_network_build_config(self):
        """Test network config with build source."""
        yaml_str = """
job_name: "test"
network:
  source: "build"
  build_config:
    dataset: "cifar10"
    layers:
      - type: "conv"
        kernel_size: 5
        stride: 2
        connectivity: 0.8
      - type: "dense"
        size: 64
"""
        config = load_config_from_string(yaml_str)
        assert config.network.source == "build"
        assert config.network.build_config.dataset == DatasetType.CIFAR10
        assert len(config.network.build_config.layers) == 2
        assert config.network.build_config.layers[0].type == LayerType.CONV

    def test_invalid_network_config_missing_path(self):
        """Test that file source requires path."""
        yaml_str = """
job_name: "test"
network:
  source: "file"
"""
        with pytest.raises(ValueError, match="path is required"):
            load_config_from_string(yaml_str)

    def test_invalid_network_config_missing_build_config(self):
        """Test that build source requires build_config."""
        yaml_str = """
job_name: "test"
network:
  source: "build"
"""
        with pytest.raises(ValueError, match="build_config is required"):
            load_config_from_string(yaml_str)

    def test_default_values(self):
        """Test that default values are applied."""
        yaml_str = """
job_name: "test"
network:
  source: "file"
  path: "net.json"
"""
        config = load_config_from_string(yaml_str)

        # Check defaults
        assert config.activity_recording.ticks_per_image == 20
        assert config.activity_recording.images_per_label == 500
        assert config.training.epochs == 20
        assert config.training.batch_size == 32
        assert config.evaluation.samples == 1000

    def test_feature_types_validation(self):
        """Test that invalid feature types are rejected."""
        yaml_str = """
job_name: "test"
network:
  source: "file"
  path: "net.json"
data_preparation:
  feature_types:
    - "invalid_feature"
"""
        with pytest.raises(ValueError, match="Invalid feature type"):
            load_config_from_string(yaml_str)

    def test_skip_steps_validation(self):
        """Test that invalid step names are rejected."""
        yaml_str = """
job_name: "test"
network:
  source: "file"
  path: "net.json"
skip_steps:
  - "invalid_step"
"""
        with pytest.raises(ValueError, match="Invalid step to skip"):
            load_config_from_string(yaml_str)

    def test_webhook_config(self):
        """Test webhook configuration parsing."""
        yaml_str = """
job_name: "test"
network:
  source: "file"
  path: "net.json"
webhooks:
  - url: "https://example.com/hook"
    events:
      - "job_started"
      - "job_completed"
    headers:
      Authorization: "Bearer token"
"""
        config = load_config_from_string(yaml_str)

        assert len(config.webhooks) == 1
        assert config.webhooks[0].url == "https://example.com/hook"
        assert len(config.webhooks[0].events) == 2

    def test_empty_config_raises_error(self):
        """Test that empty config raises error."""
        with pytest.raises(ValueError, match="Empty configuration"):
            load_config_from_string("")


class TestConfigValidation:
    """Tests for configuration validation."""

    def test_connectivity_bounds(self):
        """Test that connectivity is bounded 0-1."""
        yaml_str = """
job_name: "test"
network:
  source: "build"
  build_config:
    dataset: "mnist"
    layers:
      - type: "dense"
        connectivity: 1.5
"""
        with pytest.raises(ValueError):
            load_config_from_string(yaml_str)

    def test_train_split_bounds(self):
        """Test that train_split is bounded."""
        yaml_str = """
job_name: "test"
network:
  source: "file"
  path: "net.json"
data_preparation:
  train_split: 1.5
"""
        with pytest.raises(ValueError):
            load_config_from_string(yaml_str)

    def test_positive_epochs(self):
        """Test that epochs must be positive."""
        yaml_str = """
job_name: "test"
network:
  source: "file"
  path: "net.json"
training:
  epochs: 0
"""
        with pytest.raises(ValueError):
            load_config_from_string(yaml_str)
