import os
import pytest
import tempfile
import shutil
import json
import torch
import numpy as np
from unittest.mock import MagicMock, patch

from pipeline.config import NetworkConfig, LayerConfig, SimulationConfig, PreparationConfig, TrainingConfig, EvaluationConfig
from pipeline.step1_build_network import build_network
from pipeline.step3_prepare_data import process_data
from pipeline.step4_train_model import train_model

# Step 2 and 5 involve complex simulation/evaluation, we might mock them or run minimal versions
# For now, let's test the data flow structure

@pytest.fixture
def temp_dir():
    dir_path = tempfile.mkdtemp()
    yield dir_path
    shutil.rmtree(dir_path)

def test_step1_build_network(temp_dir):
    config = NetworkConfig(
        layers=[
            LayerConfig(layer_type="conv", filters=1, kernel_size=3, stride=1, connectivity=1.0)
        ],
        inhibitory_signals=False
    )
    build_network(config, temp_dir)
    assert os.path.exists(os.path.join(temp_dir, "network.json"))

    with open(os.path.join(temp_dir, "network.json"), 'r') as f:
        data = json.load(f)
        assert "neurons" in data

def test_step3_prepare_data(temp_dir):
    # Mock HDF5 input
    import h5py
    input_dir = os.path.join(temp_dir, "input")
    os.makedirs(input_dir)
    h5_path = os.path.join(input_dir, "activity_dataset.h5")

    # Create a dummy HDF5 file
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("u", data=np.random.rand(10, 20, 5)) # 10 samples, 20 ticks, 5 neurons
        f.create_dataset("t_ref", data=np.random.rand(10, 20, 5))
        f.create_dataset("fr", data=np.random.rand(10, 20, 5))

        # Spikes is vlen int32
        dt_vlen = h5py.special_dtype(vlen=np.dtype("int32"))
        # Create empty spikes for each sample
        spikes_data = np.array([np.array([], dtype=np.int32) for _ in range(10)], dtype=object)
        f.create_dataset("spikes", data=spikes_data, dtype=dt_vlen)

        f.create_dataset("labels", data=np.array([0]*10))
        f.create_dataset("num_samples", data=np.array([10]))
        dt = h5py.special_dtype(vlen=str)
        f.create_dataset("neuron_ids", data=np.array(["n1", "n2", "n3", "n4", "n5"], dtype=object), dtype=dt)
        f.create_dataset("layer_structure", data=np.array([5]))

    config = PreparationConfig(feature_types=["avg_S"], train_split=0.5)
    output_dir = os.path.join(temp_dir, "output")

    # Run step 3
    process_data(config, input_dir, output_dir)

    assert os.path.exists(os.path.join(output_dir, "train_data.pt"))
    assert os.path.exists(os.path.join(output_dir, "test_data.pt"))
    assert os.path.exists(os.path.join(output_dir, "dataset_metadata.json"))

def test_step4_train_model(temp_dir):
    # Create dummy .pt files
    input_dir = os.path.join(temp_dir, "data")
    os.makedirs(input_dir)

    # 5 samples, 1 feature, 20 ticks, 5 neurons -> input_size = 5 (if flat features)
    # The pipeline flattens features: num_neurons * num_features
    # SNNClassifier expects input_size

    X = torch.randn(5, 20, 10) # 5 samples, 20 ticks, 10 inputs
    y = torch.tensor([0, 1, 0, 1, 0])

    torch.save(X, os.path.join(input_dir, "train_data.pt"))
    torch.save(y, os.path.join(input_dir, "train_labels.pt"))
    torch.save(X, os.path.join(input_dir, "test_data.pt"))
    torch.save(y, os.path.join(input_dir, "test_labels.pt"))

    with open(os.path.join(input_dir, "dataset_metadata.json"), "w") as f:
        json.dump({"feature_types": ["avg_S"], "num_features": 1}, f)

    config = TrainingConfig(epochs=1, batch_size=2)
    output_dir = os.path.join(temp_dir, "model")

    # Mock snntorch if needed, but it should be installed
    train_model(config, input_dir, output_dir)

    assert os.path.exists(os.path.join(output_dir, "model.pth"))
    assert os.path.exists(os.path.join(output_dir, "training_results.json"))
