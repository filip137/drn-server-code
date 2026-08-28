from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from experiments.artifacts import sha256_file
from experiments.mnist_relu_drn.ibm_om_baseline_selection_runtime import (
    BASELINE_RESIDUAL_SCHEMA,
    BASELINE_RESIDUAL_SCHEMA_VERSION,
    _dataset_provenance,
    _rail_difference,
    _save_baseline_rail_residual,
)
from experiments.mnist_relu_drn.ibm_om_bounded_codebook_scheme_screen import (
    _tensor_sha256,
)
from experiments.mnist_shared import FlattenedDataset


class _RawTensorDataset(Dataset):
    def __init__(self, data: torch.Tensor, targets: torch.Tensor) -> None:
        self.data = data
        self.targets = targets

    def __len__(self) -> int:
        return int(self.targets.numel())

    def __getitem__(self, index: int):
        return self.data[index].to(torch.float32).unsqueeze(0), self.targets[index]


def test_dataset_provenance_binds_calibration_images_and_test_split() -> None:
    train = _RawTensorDataset(
        torch.arange(8 * 4, dtype=torch.uint8).reshape(8, 2, 2),
        torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int64),
    )
    test = _RawTensorDataset(
        torch.arange(4 * 4, dtype=torch.uint8).reshape(4, 2, 2),
        torch.tensor([3, 2, 1, 0], dtype=torch.int64),
    )
    indices = [1, 4, 7]
    loaders = SimpleNamespace(
        test=SimpleNamespace(dataset=FlattenedDataset(test)),
        calibration=SimpleNamespace(
            dataset=Subset(FlattenedDataset(train), indices)
        ),
    )
    labels = train.targets[torch.tensor(indices)]

    first = _dataset_provenance(loaders, labels)
    train.data[4, 0, 0] += 1
    second = _dataset_provenance(loaders, labels)

    assert first["test_examples"] == 4
    assert first["training_examples"] == 8
    assert first["calibration_examples"] == 3
    assert first["calibration_raw_images_sha256"] != second[
        "calibration_raw_images_sha256"
    ]
    assert first["test_data_sha256"] == second["test_data_sha256"]


def test_baseline_residual_artifact_is_pickle_free_and_hashed(
    tmp_path: Path,
) -> None:
    residuals = (
        torch.tensor([[1.0, -2.0], [0.5, 0.0]], dtype=torch.float32),
        torch.tensor([[0.25], [-0.75]], dtype=torch.float32),
    )
    report = {
        "definition": "paired_rail_voltage_difference_with_all_offsets_d_equal_zero",
        "examples": 2,
        "layers": [],
        "residual_hashes": [_tensor_sha256(value) for value in residuals],
    }
    path = tmp_path / "baseline_rail_residual.npz"

    record = _save_baseline_rail_residual(
        path,
        report=report,
        residuals=residuals,
    )

    with np.load(path, allow_pickle=False) as arrays:
        assert arrays["schema"].item() == BASELINE_RESIDUAL_SCHEMA
        assert arrays["schema_version"].item() == BASELINE_RESIDUAL_SCHEMA_VERSION
        assert arrays["examples"].item() == 2
        np.testing.assert_array_equal(
            arrays["layer_0_rail_voltage_residual"], residuals[0].numpy()
        )
        np.testing.assert_array_equal(
            arrays["layer_1_rail_voltage_residual"], residuals[1].numpy()
        )
    receipt = json.loads(Path(record["receipt"]).read_text(encoding="utf-8"))
    assert receipt["artifact_sha256"] == sha256_file(path)
    assert set(receipt["tensor_hashes"]) == {
        "layer_0_rail_voltage_residual",
        "layer_1_rail_voltage_residual",
    }


def test_rail_difference_uses_declared_layout() -> None:
    halves = torch.tensor([[1.0, 2.0, 0.25, 0.5]])
    paired = torch.tensor([[1.0, 0.25, 2.0, 0.5]])

    torch.testing.assert_close(
        _rail_difference(halves, layout="halves"),
        torch.tensor([[0.75, 1.5]]),
    )
    torch.testing.assert_close(
        _rail_difference(paired, layout="paired"),
        torch.tensor([[0.75, 1.5]]),
    )
