from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.artifacts import sha256_file
from experiments.mnist_relu_drn.ibm_om_baseline_spacing_pv import (
    RawXConductanceEmbedding,
)
from experiments.mnist_relu_drn.ibm_om_baseline_spacing_pv_no_clip_runtime import (
    _apparent_raw_diagnostic,
    _native_assignment_from_joint_artifact,
    _persistent_affine_handoff,
    _save_predictions,
)


def _joint_artifact(path: Path) -> None:
    arrays = {
        "schema": np.asarray(
            "ebl.mnist_relu_drn.ibm_om_baseline_joint_assignment"
        ),
        "schema_version": np.asarray(1, dtype=np.int64),
        "assignment_seed": np.asarray(17, dtype=np.int64),
        "hardware_instance_id": np.asarray("hardware"),
        "layer_0_reset_mean_raw_a": np.asarray(
            [[-2.5, -2.0], [2.0, 2.0]], dtype=np.float32
        ),
        "layer_0_min_bound": np.asarray(
            [[-3.0, -1.0], [-0.5, 1.0]], dtype=np.float32
        ),
        "layer_0_max_bound": np.asarray(
            [[-1.0, 1.0], [1.5, 3.0]], dtype=np.float32
        ),
        "layer_1_reset_mean_raw_a": np.asarray([[-1.5]], dtype=np.float32),
        "layer_1_min_bound": np.asarray([[-2.0]], dtype=np.float32),
        "layer_1_max_bound": np.asarray([[0.0]], dtype=np.float32),
    }
    np.savez(path, **arrays)


def test_native_assignment_retains_negative_and_above_one_raw_x(
    tmp_path: Path,
) -> None:
    path = tmp_path / "joint.npz"
    _joint_artifact(path)
    native = _native_assignment_from_joint_artifact(
        {
            "assignment_seed": 17,
            "hardware_instance_id": "hardware",
            "joint_assignment": {"path": str(path), "sha256": sha256_file(path)},
        }
    )

    lower = native["cell_lower_raw_x"][0]
    upper = native["cell_upper_raw_x"][0]
    reset = native["reset_baseline_raw_x"][0]
    assert lower[0, 0].item() == -1.0
    assert upper[1, 1].item() == 2.0
    assert reset[0, 0].item() == -0.75
    assert reset[0, 1].item() == 0.0
    assert reset[1, 0].item() == 1.25
    assert native["report"]["public_0_1_intersection_applied"] is False
    assert native["report"]["native_support_bounded_reset_count"] == 2


def _outcome(*, persistent: torch.Tensor, apparent: torch.Tensor):
    return SimpleNamespace(
        persistent_endpoint_unit=persistent,
        apparent_endpoint_unit=apparent,
        raw_lower_unit=torch.tensor([-1.5, -0.5], dtype=torch.float32),
        raw_upper_unit=torch.tensor([0.0, 1.5], dtype=torch.float32),
        coordinate="x=(native_raw_active_a+1)/2",
    )


def test_persistent_handoff_is_affine_without_public_projection() -> None:
    outcome = _outcome(
        persistent=torch.tensor([-1.0, 0.5], dtype=torch.float32),
        apparent=torch.tensor([-2.5, 2.0], dtype=torch.float32),
    )
    embedding = RawXConductanceEmbedding(
        raw_x_origin=-2.0,
        conductance_per_raw_x=1.0,
        conductance_ceiling=4.0,
    )
    population = SimpleNamespace(size=2, binding_shapes=((2,),))

    layers, report = _persistent_affine_handoff(
        outcome, population, embedding
    )

    assert torch.equal(layers[0], torch.tensor([1.0, 2.5]))
    assert report["projected"] == 0
    assert report["below_native_support"] == 0
    assert report["above_native_support"] == 0


def test_persistent_handoff_fails_instead_of_projecting_native_support() -> None:
    outcome = _outcome(
        persistent=torch.tensor([-1.6, 0.5], dtype=torch.float32),
        apparent=torch.tensor([-1.0, 0.5], dtype=torch.float32),
    )
    embedding = RawXConductanceEmbedding(
        raw_x_origin=-2.0,
        conductance_per_raw_x=1.0,
        conductance_ceiling=4.0,
    )

    with pytest.raises(RuntimeError, match="left native support"):
        _persistent_affine_handoff(
            outcome,
            SimpleNamespace(size=2, binding_shapes=((2,),)),
            embedding,
        )


def test_apparent_endpoint_is_diagnostic_even_outside_affine_range() -> None:
    outcome = _outcome(
        persistent=torch.tensor([-1.0, 0.5], dtype=torch.float32),
        apparent=torch.tensor([-2.5, 2.5], dtype=torch.float32),
    )
    embedding = RawXConductanceEmbedding(
        raw_x_origin=-2.0,
        conductance_per_raw_x=1.0,
        conductance_ceiling=4.0,
    )

    report = _apparent_raw_diagnostic(outcome, embedding)

    assert report["applied_to_drn"] is False
    assert report["at_or_below_affine_origin"] == 1
    assert report["above_affine_ceiling"] == 1


def test_prediction_artifact_has_no_apparent_network_prediction(
    tmp_path: Path,
) -> None:
    path = tmp_path / "predictions.npz"
    labels = torch.tensor([0, 1, 1], dtype=torch.int64)
    _save_predictions(
        path,
        labels=labels,
        teacher=labels.clone(),
        continuous=labels.clone(),
        ideal=torch.tensor([0, 0, 1]),
        persistent=(torch.tensor([0, 1, 1]),),
        endpoint_seeds=(7,),
    )

    with np.load(path, allow_pickle=False) as payload:
        assert "pv_persistent_prediction_seed_7" in payload.files
        assert not any("apparent" in name for name in payload.files)
