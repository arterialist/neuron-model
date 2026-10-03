from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
import pytest

from pipeline.utils.activity_data import (
    HDF5TensorRecorder,
    LazyActivityDataset,
    load_activity_dataset,
)


def recorded_file(directory, neuron_id=1):
    network = SimpleNamespace(neurons={neuron_id: SimpleNamespace(metadata={})})
    recorder = HDF5TensorRecorder(str(directory), network)
    values = np.array([[1.0], [2.0]], dtype=np.float32)
    recorder.save_sample_from_buffers(
        0, 6, values, values + 1, values + 2, np.array([[1, 0]], dtype=np.int32)
    )
    recorder.close()
    return directory / "activity_dataset.h5"


def test_load_uses_explicit_hdf5_filename(temp_dir):
    path = recorded_file(temp_dir)
    custom_path = temp_dir / "recording.h5"
    path.rename(custom_path)
    records, layers = load_activity_dataset(str(custom_path))
    assert records[0]["label"] == 6
    np.testing.assert_array_equal(records[0]["u"], [[1.0], [2.0]])
    assert layers == [1]


def test_noninteger_neuron_ids_are_decoded_strings(temp_dir):
    path = recorded_file(temp_dir, neuron_id="cell-uuid")
    dataset = LazyActivityDataset(path)
    try:
        assert dataset.neuron_ids == ["cell-uuid"]
    finally:
        dataset.close()


def test_failed_metadata_loading_closes_hdf5_handle(temp_dir):
    path = temp_dir / "invalid.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("labels", data=[1])
    opened = []
    file_factory = h5py.File

    def open_file(*args, **kwargs):
        handle = file_factory(*args, **kwargs)
        opened.append(handle)
        return handle

    with patch("pipeline.utils.activity_data.h5py.File", side_effect=open_file):
        with pytest.raises(FileNotFoundError, match="Neuron metadata"):
            LazyActivityDataset(path)
    assert len(opened) == 1
    assert not opened[0].id.valid


def test_bulk_loading_closes_dataset_after_sample_error(temp_dir):
    path = recorded_file(temp_dir)
    dataset = LazyActivityDataset(path)
    handle = dataset.h5_file
    with (
        patch("pipeline.utils.activity_data.LazyActivityDataset", return_value=dataset),
        patch.object(LazyActivityDataset, "__getitem__", side_effect=ValueError("bad sample")),
    ):
        with pytest.raises(ValueError, match="bad sample"):
            load_activity_dataset(str(path))
    assert not handle.id.valid
