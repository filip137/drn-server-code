from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.mnist_relu_drn.ibm_om_winsorized_pv_ensemble_qat import (
    PersistentEndpointCodebookEnsemble,
)
from experiments.mnist_relu_drn.ibm_om_winsorized_pv_ensemble_qat_runtime import (
    ENSEMBLE_SCHEMA,
    _development_key,
    _ensemble_report,
    _objective_from_losses,
    _sample_indices,
    _save_ensemble,
)


def _ensemble() -> PersistentEndpointCodebookEnsemble:
    maximum = torch.tensor([0, 2, 1], dtype=torch.int64)
    values = torch.full((4, 3, 3), torch.nan, dtype=torch.float32)
    for sample in range(4):
        for cell, limit in enumerate(maximum.tolist()):
            for level in range(limit + 1):
                values[sample, cell, level] = 0.01 * (
                    1 + sample + cell + level
                )
    return PersistentEndpointCodebookEnsemble(
        assignment_seed=87002,
        population_fingerprint="runtime-test-population",
        binding_shapes=((1, 1), (1, 2)),
        endpoint_seeds=(89521, 89522, 89523, 89524),
        spacing_raw_x=0.04745,
        tolerance_raw_x=0.023725,
        maximum_program_pulses=128,
        baseline_raw_x=torch.tensor([0.1, 0.2, 0.3]),
        maximum_index=maximum,
        persistent_raw_x=values,
    )


def test_sample_schedule_is_balanced_and_arm_exact() -> None:
    assert _sample_indices("deterministic", 123) == ()
    assert [_sample_indices("mean2", index) for index in range(4)] == [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
    ]
    assert _sample_indices("tail4", 91) == (0, 1, 2, 3)
    with pytest.raises(ValueError, match="Unsupported"):
        _sample_indices("noise", 0)
    with pytest.raises(ValueError, match="nonnegative"):
        _sample_indices("mean2", -1)


def test_development_key_prioritizes_mean_then_minimum_then_kl() -> None:
    base = {
        "persistent_mean_accuracy": 0.80,
        "persistent_minimum_accuracy": 0.70,
        "persistent_mean_kl_teacher_student": 0.2,
    }
    assert _development_key({**base, "persistent_mean_accuracy": 0.81}, 2) > (
        _development_key({**base, "persistent_minimum_accuracy": 0.99}, 1)
    )
    assert _development_key({**base, "persistent_minimum_accuracy": 0.71}, 2) > (
        _development_key(base, 1)
    )
    assert _development_key(
        {**base, "persistent_mean_kl_teacher_student": 0.1}, 2
    ) > _development_key(base, 1)
    assert _development_key(base, 1) > _development_key(base, 2)


def test_objective_uses_the_declared_dataclass_weights() -> None:
    assert _objective_from_losses(
        "deterministic", ideal_loss=2.0, persistent_losses=()
    ) == pytest.approx(2.0)
    assert _objective_from_losses(
        "mean2", ideal_loss=2.0, persistent_losses=(4.0, 8.0)
    ) == pytest.approx(5.0)
    # Tail-four places its extra 0.25 weight on the stable first maximum.
    assert _objective_from_losses(
        "tail4", ideal_loss=2.0, persistent_losses=(1.0, 8.0, 8.0, 3.0)
    ) == pytest.approx(5.0)


def test_saved_ensemble_preserves_nans_and_reported_no_projection(
    tmp_path: Path,
) -> None:
    ensemble = _ensemble()
    path = _save_ensemble(tmp_path / "ensemble.npz", ensemble)
    report = _ensemble_report(ensemble)

    with np.load(path, allow_pickle=False) as payload:
        assert payload["schema"].item() == ENSEMBLE_SCHEMA
        assert payload["assignment_seed"].item() == 87002
        observed = torch.from_numpy(payload["persistent_raw_x"])
    assert torch.equal(torch.isnan(observed), torch.isnan(ensemble.persistent_raw_x))
    assert torch.equal(
        torch.nan_to_num(observed), torch.nan_to_num(ensemble.persistent_raw_x)
    )
    assert report["apparent_endpoint_applied_to_drn"] is False
    assert report["endpoint_projection_count"] == 0
    assert report["full_conductance_formula"] == "G=2*x_persistent"
