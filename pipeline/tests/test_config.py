import pytest
import yaml
from pipeline.config import JobConfig

def test_job_config_loading():
    yaml_content = """
name: test_job
network:
  layers:
    - layer_type: conv
      filters: 0
      kernel_size: 9
      stride: 3
  inhibitory_signals: false
simulation:
  ticks_per_image: 10
  images_per_label: 2
  dataset_name_base: "mnist_test"
preparation:
  feature_types: ["firings"]
training:
  epochs: 1
evaluation:
  eval_samples: 5
"""
    cfg_dict = yaml.safe_load(yaml_content)
    config = JobConfig(**cfg_dict)
    assert config.name == "test_job"
    assert config.network.layers[0].layer_type == "conv"
    assert config.simulation.ticks_per_image == 10
