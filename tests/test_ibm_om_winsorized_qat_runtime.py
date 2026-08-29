import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.artifacts import sha256_file
from experiments.mnist_relu_drn.ibm_om_baseline_selection import (
    hardware_instance_fingerprint,
    scatter_quads,
)
from experiments.mnist_relu_drn.ibm_om_bounded_codebook_scheme_screen import (
    _tensor_sha256,
)
from experiments.mnist_relu_drn.ibm_om_winsorized_qat import (
    WinsorizedQatLayerTemplate,
)
from experiments.mnist_relu_drn.ibm_om_winsorized_qat_runtime import (
    CHECKPOINT_SCHEMA,
    CHECKPOINT_SCHEMA_VERSION,
    _load_epoch_10_logical_master,
    _load_winsorized_population,
)
from training.ibm_reram_hwa import (
    IbmReramArrayPopulation,
    load_om_array_population,
    save_om_array_population,
)


def _write_winsorized_population(
    directory: Path,
) -> tuple[Path, Path, IbmReramArrayPopulation, str]:
    source_hardware_id = "source-joint-hardware"
    tensors = {
        "min_bound": torch.tensor([-1.0, -0.5, 0.0], dtype=torch.float32),
        "max_bound": torch.tensor([0.5, 1.0, 0.75], dtype=torch.float32),
        "dwmin_up": torch.full((3,), 0.1, dtype=torch.float32),
        "dwmin_down": torch.full((3,), 0.1, dtype=torch.float32),
        "reference": torch.zeros(3, dtype=torch.float32),
        "corrupt": torch.zeros(3, dtype=torch.bool),
        "published_corrupt": torch.zeros(3, dtype=torch.bool),
    }
    fingerprint = hardware_instance_fingerprint(
        {
            "source_joint_hardware_instance_id": source_hardware_id,
            "intervention": "nominal_bound_winsorization",
            "raw_a_minimum": -1.0,
            "raw_a_maximum": 1.0,
            "identity_resampled": False,
        },
        tensors,
    )
    population = IbmReramArrayPopulation(
        assignment_seed=86001,
        corruption_policy="counterfactual_repaired",
        binding_keys=("base.dense_weight.0", "base.dense_weight.1"),
        binding_shapes=((1, 2), (1, 1)),
        binding_sampling_seeds=(101, 102),
        donor_sampling_seeds=(201, 202),
        nominal_dw_min=0.0949,
        dw_min_std=0.3,
        write_noise_std=0.1,
        max_bound=tensors["max_bound"],
        min_bound=tensors["min_bound"],
        dwmin_up=tensors["dwmin_up"],
        dwmin_down=tensors["dwmin_down"],
        reference=tensors["reference"],
        corrupt=tensors["corrupt"],
        published_corrupt=tensors["published_corrupt"],
        fingerprint=fingerprint,
        aihwkit_version="1.1.0",
    )
    path = directory / "winsorized_population.npz"
    save_om_array_population(path, population)
    receipt_path = path.with_suffix(".receipt.json")
    receipt_path.write_text(
        json.dumps(
            {
                "schema": (
                    "ebl.mnist_relu_drn.ibm_om_winsorized_population_receipt"
                ),
                "schema_version": 1,
                "artifact": path.name,
                "artifact_sha256": sha256_file(path),
                "assignment_seed": population.assignment_seed,
                "population_fingerprint": fingerprint,
                "source_population_fingerprint": source_hardware_id,
                "intervention": {
                    "policy": "nominal_bound_winsorization",
                    "source_population_fingerprint": source_hardware_id,
                    "winsorized_population_fingerprint": fingerprint,
                    "raw_a_interval": [-1.0, 1.0],
                    "identity_resampled": False,
                },
            }
        ),
        encoding="utf-8",
    )
    return path, receipt_path, population, source_hardware_id


def test_study_local_loader_accepts_custom_fingerprint_without_weakening_generic(
    tmp_path: Path,
) -> None:
    path, receipt_path, expected, source_hardware_id = (
        _write_winsorized_population(tmp_path)
    )
    # The generic contract stays fail-closed for this study-local artifact.
    with pytest.raises(ValueError):
        load_om_array_population(path)

    actual = _load_winsorized_population(
        path,
        receipt_path,
        expected_sha256=sha256_file(path),
        expected_fingerprint=expected.fingerprint,
        expected_assignment_seed=expected.assignment_seed,
        source_joint_hardware_instance_id=source_hardware_id,
    )
    assert actual.fingerprint == expected.fingerprint
    assert actual.assignment_seed == 86001
    assert torch.equal(actual.min_bound, expected.min_bound)
    assert torch.equal(actual.max_bound, expected.max_bound)


def test_study_local_loader_rejects_receipt_rebinding(tmp_path: Path) -> None:
    path, receipt_path, expected, source_hardware_id = (
        _write_winsorized_population(tmp_path)
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["source_population_fingerprint"] = "different-source"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="receipt does not bind"):
        _load_winsorized_population(
            path,
            receipt_path,
            expected_sha256=sha256_file(path),
            expected_fingerprint=expected.fingerprint,
            expected_assignment_seed=expected.assignment_seed,
            source_joint_hardware_instance_id=source_hardware_id,
        )


def _template(layer_index: int, layout: str) -> WinsorizedQatLayerTemplate:
    baseline_quad = torch.tensor([[[0.25, 0.375, 0.25, 0.375]]])
    upper_quad = torch.tensor([[[0.75, 0.625, 0.5, 0.875]]])
    return WinsorizedQatLayerTemplate(
        layer_index=layer_index,
        layout=layout,
        baseline_full_g=2.0
        * scatter_quads(baseline_quad, shape=(2, 2), layout=layout),
        baseline_raw_x=scatter_quads(
            baseline_quad, shape=(2, 2), layout=layout
        ),
        cell_upper_raw_x=scatter_quads(
            upper_quad, shape=(2, 2), layout=layout
        ),
        positive_headroom_raw_x=torch.tensor([[0.5]]),
        negative_headroom_raw_x=torch.tensor([[0.25]]),
        positive_capacity=torch.tensor([[4]], dtype=torch.int64),
        negative_capacity=torch.tensor([[2]], dtype=torch.int64),
        initial_normalized_weight=torch.tensor([[0.25]]),
        level_spacing_raw_x=0.125,
    )


def _checkpoint_payload(masters: tuple[torch.Tensor, ...]) -> dict:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "epoch": 10,
        "evidence_tier": "exploratory_noncanonical",
        "teacher_sha256": "teacher-sha",
        "spacing_delta_x_multiplier": 1,
        "fixed_logit_gain": 14.125,
        "source_absmax": [1.0, 2.0],
        "logical_master": masters,
        "logical_master_sha256": tuple(_tensor_sha256(value) for value in masters),
        "optimizer_state": {"state": {}, "param_groups": []},
        "validation": {"student_accuracy": 0.97, "student_correct": 4850},
    }


def test_epoch_10_recovery_checkpoint_is_strictly_validated(tmp_path: Path) -> None:
    templates = (_template(0, "halves"), _template(1, "paired"))
    masters = (
        torch.tensor([[0.5]], dtype=torch.float32),
        torch.tensor([[-0.25]], dtype=torch.float32),
    )
    path = tmp_path / "epoch_10.pt"
    torch.save(_checkpoint_payload(masters), path)
    protocol = SimpleNamespace(
        mapping=SimpleNamespace(
            spacing_delta_x_multiplier=1,
            fixed_logit_gain=14.125,
        )
    )

    loaded, payload = _load_epoch_10_logical_master(
        path,
        templates=templates,
        protocol=protocol,
        teacher_sha256="teacher-sha",
        source_absmax=(1.0, 2.0),
        device=torch.device("cpu"),
    )
    assert payload["epoch"] == 10
    assert all(torch.equal(left, right) for left, right in zip(loaded, masters))

    bad = _checkpoint_payload(masters)
    bad["epoch"] = 9
    torch.save(bad, path)
    with pytest.raises(ValueError, match="protocol mismatch"):
        _load_epoch_10_logical_master(
            path,
            templates=templates,
            protocol=protocol,
            teacher_sha256="teacher-sha",
            source_absmax=(1.0, 2.0),
            device=torch.device("cpu"),
        )


def test_epoch_10_recovery_rejects_master_hash_mismatch(tmp_path: Path) -> None:
    templates = (_template(0, "halves"), _template(1, "paired"))
    masters = (torch.tensor([[0.5]]), torch.tensor([[-0.25]]))
    payload = _checkpoint_payload(masters)
    payload["logical_master_sha256"] = ("0" * 64, "1" * 64)
    path = tmp_path / "epoch_10.pt"
    torch.save(payload, path)
    protocol = SimpleNamespace(
        mapping=SimpleNamespace(
            spacing_delta_x_multiplier=1,
            fixed_logit_gain=14.125,
        )
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _load_epoch_10_logical_master(
            path,
            templates=templates,
            protocol=protocol,
            teacher_sha256="teacher-sha",
            source_absmax=(1.0, 2.0),
            device=torch.device("cpu"),
        )
