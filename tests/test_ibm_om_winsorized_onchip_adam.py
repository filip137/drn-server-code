from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from experiments.mnist_relu_drn.ibm_om_baseline_selection import quad_stack
from experiments.mnist_relu_drn.ibm_om_bounded_codebook_scheme_screen import (
    _tensor_sha256,
)
from experiments.mnist_relu_drn.ibm_om_winsorized_onchip_adam import (
    CLAIM_LABEL,
    EVIDENCE_TIER,
    P0_SCHEMA,
    STATE_AUTHORITY,
    PersistentPulseAdam,
    clone_p0_bundle,
    load_p0_bundle,
    save_p0_bundle,
)
from experiments.mnist_relu_drn.ibm_om_winsorized_onchip_adam_runtime import (
    MULTI_QAT_CHECKPOINT_SCHEMA,
    _load_qat_logical_master,
)
from training.ibm_reram_program_verify import (
    IbmReramPopulation,
    IbmReramRawActivePlant,
)


def _population(size: int) -> IbmReramPopulation:
    vector = lambda value: torch.full((size,), value, dtype=torch.float32)
    return IbmReramPopulation(
        preset="reram_array_om",
        aihwkit_version="1.1.0",
        nominal_dw_min=0.1,
        dw_min_std=0.0,
        write_noise_std=0.0,
        mult_noise=False,
        construction_seeds=torch.arange(1, size + 1, dtype=torch.int64),
        max_bound=vector(1.0),
        min_bound=vector(-1.0),
        dwmin_up=vector(0.1),
        dwmin_down=vector(0.1),
        reference=vector(0.0),
        corrupt=torch.zeros(size, dtype=torch.bool),
        preset_parameters={},
    )


def _plant() -> IbmReramRawActivePlant:
    population = _population(8)
    return IbmReramRawActivePlant(
        population, seeds=tuple(range(11, 19)), device="cpu"
    )


def _p0_payload() -> dict:
    size = 8
    return {
        "schema": P0_SCHEMA,
        "schema_version": 1,
        "evidence_tier": EVIDENCE_TIER,
        "claim_label": CLAIM_LABEL,
        "state_authority": STATE_AUTHORITY,
        "assignment_seed": 87003,
        "endpoint_seed": 89301,
        "spacing_delta_x_multiplier": 2,
        "qat_checkpoint_sha256": "0" * 64,
        "teacher_sha256": "1" * 64,
        "population_sha256": "2" * 64,
        "population_fingerprint": "3" * 64,
        "mapping_sha256": "4" * 64,
        "joint_assignment_sha256": "5" * 64,
        "commissioning_sha256": "6" * 64,
        "binding_keys": ["w0", "w1"],
        "binding_shapes": [[2, 2], [2, 2]],
        "maximum_program_pulses": 128,
        "recovery_pulse_cap": 64,
        "maximum_random_draws": 385,
        "pulse_selection_seed": 90403,
        "target_raw_x": torch.full((size,), 0.5, dtype=torch.float32),
        "requested_index": torch.zeros(size, dtype=torch.int64),
        "logical_master_sha256": ["7" * 64, "8" * 64],
        "programming": {},
        "continuation_state": {
            "schema_version": 2,
            "preset": "reram_array_om",
            "construction_seeds": torch.arange(1, size + 1, dtype=torch.int64),
            "persistent": torch.zeros(size, dtype=torch.float32),
            "apparent": torch.zeros(size, dtype=torch.float32),
            "rng_backend": "per_trajectory_buffered_torch_cpu",
            "seeds": list(range(11, 19)),
            "maximum_random_draws": 385,
            "draw_indices": torch.zeros(size, dtype=torch.int64),
            "state_coordinate": "native_raw_active_a",
        },
        "validation": {},
        "validation_prediction_sha256": "9" * 64,
        "no_remap_after_p0": True,
        "apparent_endpoint_applied_to_drn": False,
    }


def test_p0_clone_and_round_trip_own_tensor_storage(tmp_path) -> None:
    original = _p0_payload()
    clone = clone_p0_bundle(original)
    clone["continuation_state"]["persistent"][0] = 0.25
    assert original["continuation_state"]["persistent"][0].item() == 0.0

    path = save_p0_bundle(original, tmp_path / "p0.pt")
    loaded = load_p0_bundle(path)
    assert torch.equal(
        loaded["continuation_state"]["persistent"],
        original["continuation_state"]["persistent"],
    )


def test_contrast_adam_realizes_four_coordinated_rails_or_none() -> None:
    plant = _plant()
    physical_port = plant.controller_port()

    class RecordingPort:
        size = physical_port.size

        def __init__(self) -> None:
            self.requests = []

        def verify(self):
            return physical_port.verify()

        def apply_identical_pulses(self, directions, counts):
            self.requests.append((directions.clone(), counts.clone()))
            physical_port.apply_identical_pulses(directions, counts)

    port = RecordingPort()
    optimizer = PersistentPulseAdam(
        port,
        device="cpu",
        binding_shapes=((2, 2), (2, 2)),
        technique="contrast_constrained",
        learning_rate_raw_x=1.0,
        nominal_delta_x=0.1,
        pulse_cap=1,
        pulse_selection_seed=17,
    )
    gradient = torch.tensor(((1.0, -1.0), (-1.0, 1.0)))
    before = plant.persistent.clone()
    step = optimizer.step((gradient, gradient))
    delta = plant.persistent - before
    assert step.pulsed_cells == 8
    requested_direction, requested_count = port.requests[-1]
    for matrix, layout in (
        (requested_direction[:4].reshape(2, 2), "halves"),
        (requested_direction[4:].reshape(2, 2), "paired"),
    ):
        quad = quad_stack(matrix, layout=layout).reshape(-1)
        assert torch.count_nonzero(quad).item() == 4
        assert quad.sum().item() == 0
    assert torch.equal(requested_count, torch.ones_like(requested_count))
    for matrix in (delta[:4].reshape(2, 2), delta[4:].reshape(2, 2)):
        quad = quad_stack(matrix, layout="halves").reshape(-1)
        assert torch.count_nonzero(quad).item() == 4
        assert torch.allclose(quad, -quad[0] * torch.tensor((-1.0, 1.0, 1.0, -1.0)))
        assert quad.sum().item() == pytest.approx(0.0, abs=1e-7)

    # One rail at cap blocks its entire quad; no partial physical update occurs.
    before = plant.persistent.clone()
    step = optimizer.step((gradient, gradient))
    assert step.pulsed_cells == 0
    assert step.capped_cells == 8
    assert torch.equal(plant.persistent, before)


def test_multi_assignment_epoch10_checkpoint_contract_is_accepted(tmp_path) -> None:
    teacher = torch.nn.Module()
    teacher.p0 = torch.nn.Parameter(torch.tensor(((1.0, -0.5),)))
    teacher.p1 = torch.nn.Parameter(torch.tensor(((0.25,), (-0.75,))))
    masters = (
        torch.tensor(((0.2, -0.3),), dtype=torch.float32),
        torch.tensor(((0.4,), (-0.1,)), dtype=torch.float32),
    )
    hashes = tuple(_tensor_sha256(value) for value in masters)
    contract = SimpleNamespace(
        qat_training_assignment_seeds=(86001, 87001),
        qat_training_population_fingerprints=("a" * 64, "b" * 64),
        qat_development_assignment_seed=87002,
        qat_development_population_fingerprint="c" * 64,
        qat_assignment_cycle="global_minibatch_ordinal_modulo_two_start_86001",
        qat_expected_global_minibatch_ordinal=34380,
    )
    payload = {
        "schema": MULTI_QAT_CHECKPOINT_SCHEMA,
        "schema_version": 1,
        "epoch": 10,
        "evidence_tier": EVIDENCE_TIER,
        "teacher_sha256": "d" * 64,
        "spacing_delta_x_multiplier": 2,
        "fixed_logit_gain": 14.12537544622754,
        "source_absmax": [1.0, 0.75],
        "training_assignment_seeds": [86001, 87001],
        "training_population_fingerprints": {
            "86001": "a" * 64,
            "87001": "b" * 64,
        },
        "development_assignment_seed": 87002,
        "development_population_fingerprint": "c" * 64,
        "assignment_cycle": "global_minibatch_ordinal_modulo_two_start_86001",
        "global_minibatch_ordinal": 34380,
        "logical_master": masters,
        "logical_master_sha256": hashes,
        "optimizer_state": {},
        "development_validation": {},
    }
    path = tmp_path / "multi.pt"
    torch.save(payload, path)
    loaded, loaded_hashes = _load_qat_logical_master(
        path,
        templates=(
            SimpleNamespace(logical_shape=(1, 2)),
            SimpleNamespace(logical_shape=(2, 1)),
        ),
        spacing_delta_x_multiplier=2,
        fixed_logit_gain=14.12537544622754,
        teacher_sha256="d" * 64,
        teacher=teacher,
        source_contract=contract,
        device=torch.device("cpu"),
    )
    assert loaded_hashes == hashes
    assert all(torch.equal(left, right) for left, right in zip(loaded, masters))

    payload["schema"] = "ebl.mnist_relu_drn.ibm_om_winsorized_qat_logical_master"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="Multi-assignment QAT checkpoint"):
        _load_qat_logical_master(
            path,
            templates=(
                SimpleNamespace(logical_shape=(1, 2)),
                SimpleNamespace(logical_shape=(2, 1)),
            ),
            spacing_delta_x_multiplier=2,
            fixed_logit_gain=14.12537544622754,
            teacher_sha256="d" * 64,
            teacher=teacher,
            source_contract=contract,
            device=torch.device("cpu"),
        )
