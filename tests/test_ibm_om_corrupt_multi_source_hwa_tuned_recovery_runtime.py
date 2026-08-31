from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.definitions import resolve_experiment_config
from experiments.mnist_relu_drn import (
    ibm_om_corrupt_multi_source_hwa_tuned_recovery_runtime as runtime,
)
from experiments.schema import RunMode


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "examples/mnist_relu_drn/"
    "ibm_om_corrupt_multi_source_hwa_tuned_recovery/standard_ladder.json"
)


def test_runtime_resolves_exact_frozen_contract_and_required_run_helpers() -> None:
    _definition, spec = resolve_experiment_config(CONFIG_PATH, RunMode.TRAIN)
    contract = runtime._strict_protocol_values(spec.protocol)

    assert contract["train_assignments"] == (94001, 94002)
    assert contract["train_endpoints"] == {
        94001: (94101, 94102, 94103, 94104),
        94002: (94111, 94112, 94113, 94114),
    }
    assert contract["development_assignment"] == 94003
    assert contract["target_assignment"] == 94004
    assert contract["p0_endpoint"] == 94301
    assert contract["recovery_pulse_seed"] == 94401
    assert contract["recovery_data_order_seed"] == 94402

    # These predecessor helpers are reached only late in the real run.  Keep
    # the run_train global surface covered so a missing lazy import fails here,
    # rather than after the four-arm CUDA screen finishes.
    for name in ("_save_endpoint_bank", "_population_report", "_paired_target_hwa_gate"):
        assert callable(runtime.run_train.__globals__[name])


def test_arm_fields_use_exact_source_schedule_and_w2_multiplier() -> None:
    _definition, spec = resolve_experiment_config(CONFIG_PATH, RunMode.TRAIN)
    arms = tuple(spec.protocol.training.arms)

    assert tuple(runtime._arm_source_seeds(arm) for arm in arms) == (
        (94001,),
        (94001,),
        (94001, 94002),
        (94001, 94002),
    )
    assert tuple(runtime._arm_w2_factor(arm) for arm in arms) == (1, 30, 1, 30)


def test_hwa_selection_orders_mean_min_kl_epoch_then_simplicity() -> None:
    def candidate(
        arm_id: str,
        epoch: int,
        *,
        mean: float = 0.8,
        minimum: float = 0.7,
        kl: float = 0.2,
    ) -> dict[str, object]:
        return {
            "arm_id": arm_id,
            "epoch": epoch,
            "development": {
                "persistent_mean_accuracy": mean,
                "persistent_minimum_accuracy": minimum,
                "persistent_mean_kl_teacher_student": kl,
            },
        }

    candidates = (
        candidate("single_w2_x1", 2),
        candidate("single_w2_x30", 1),
        candidate("multi_w2_x1", 1, kl=0.19),
        candidate("multi_w2_x30", 6, mean=0.81, minimum=0.60, kl=0.40),
    )
    assert runtime._select_hwa_candidate(candidates)["arm_id"] == "multi_w2_x30"

    tied_mean = (
        candidate("single_w2_x1", 2),
        candidate("single_w2_x30", 1),
        candidate("multi_w2_x1", 1, kl=0.19),
    )
    assert runtime._select_hwa_candidate(tied_mean)["arm_id"] == "multi_w2_x1"

    exact_tie = (
        candidate("single_w2_x1", 1),
        candidate("single_w2_x30", 1),
    )
    assert runtime._select_hwa_candidate(exact_tie)["arm_id"] == "single_w2_x1"


def test_recovery_selection_and_output_kl_keep_both_physical_gradients() -> None:
    reports = (
        {
            "epoch": 1,
            "validation": {"student_accuracy": 0.8, "kl_teacher_student": 0.3},
        },
        {
            "epoch": 2,
            "validation": {"student_accuracy": 0.8, "kl_teacher_student": 0.2},
        },
        {
            "epoch": 3,
            "validation": {"student_accuracy": 0.8, "kl_teacher_student": 0.2},
        },
    )
    assert max(reports, key=runtime._recovery_selection_key)["epoch"] == 2

    hidden = torch.tensor([1.0, -2.0])
    output = torch.tensor([3.0])
    returned_hidden, returned_output = runtime._output_kl_gradients((hidden, output))
    assert returned_hidden is hidden
    assert returned_output is output
    with pytest.raises(FloatingPointError):
        runtime._output_kl_gradients((hidden, torch.tensor([float("nan")])))


def test_p0_validation_parity_is_exact_and_fail_closed() -> None:
    saved = {
        "validation": {
            "student_correct": 123,
            "kl_teacher_student": 0.25,
        },
        "validation_prediction_sha256": "a" * 64,
    }
    observed = {"student_correct": 123, "kl_teacher_student": 0.25}
    assert runtime._assert_p0_validation_parity(saved, observed, "a" * 64) == {
        "student_correct_exact": True,
        "kl_teacher_student_exact": True,
        "prediction_sha256_exact": True,
        "checked_before_first_update": True,
    }

    with pytest.raises(RuntimeError, match="differs before"):
        runtime._assert_p0_validation_parity(
            saved,
            {**observed, "kl_teacher_student": 0.2500000001},
            "a" * 64,
        )
    with pytest.raises(RuntimeError, match="differs before"):
        runtime._assert_p0_validation_parity(saved, observed, "b" * 64)


def test_recovery_generator_state_roundtrip_is_fail_closed() -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(94402)
    record = runtime._generator_state_record(generator.get_state())
    restored = runtime._assert_generator_state_record(record)
    assert torch.equal(restored, generator.get_state())

    corrupted = dict(record)
    corrupted["sha256"] = "0" * 64
    with pytest.raises(RuntimeError):
        runtime._assert_generator_state_record(corrupted)


def test_programmability_excludes_corrupt_supported_entries() -> None:
    bank = SimpleNamespace(
        maximum_index=torch.tensor([1, 0, 2], dtype=torch.int64),
        num_levels=3,
        endpoint_seeds=(1, 2, 3, 4),
    )
    population = SimpleNamespace(
        corrupt=torch.tensor([False, True, False], dtype=torch.bool)
    )
    report = runtime._corrected_bank_programmability(bank, population)

    assert report["per_endpoint_repaired_supported_entries"] == 6
    assert report["per_endpoint_controller_programmable_healthy_entries"] == 5
    assert report["per_endpoint_immutable_corrupt_supported_entries"] == 1
    assert report["all_endpoint_controller_programmable_healthy_entries"] == 20


def test_source_and_development_masks_are_pairwise_distinct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def pair(seed: int, mask: tuple[bool, ...]) -> SimpleNamespace:
        corrupt = torch.tensor(mask, dtype=torch.bool)
        published = SimpleNamespace(corrupt=corrupt, fingerprint=f"p-{seed}")
        repaired = SimpleNamespace(
            corrupt=torch.zeros_like(corrupt),
            published_corrupt=corrupt.clone(),
            fingerprint=f"r-{seed}",
        )
        return SimpleNamespace(published=published, repaired=repaired)

    monkeypatch.setattr(
        runtime,
        "_assert_distinct_array_pair",
        lambda _left, _right: {
            "published_fingerprints_differ": True,
            "repaired_fingerprints_differ": True,
        },
    )
    pairs = {
        94001: pair(94001, (True, False, False)),
        94002: pair(94002, (False, True, False)),
        94003: pair(94003, (False, False, True)),
    }
    report = runtime._assert_pairwise_source_development_distinctness(
        pairs, (94001, 94002, 94003)
    )
    assert report["pair_count"] == 3
    assert report["all_published_corrupt_masks_pairwise_distinct"] is True

    pairs[94003] = pair(94003, (False, True, False))
    with pytest.raises(RuntimeError, match="reused an exact"):
        runtime._assert_pairwise_source_development_distinctness(
            pairs, (94001, 94002, 94003)
        )


def test_code_change_report_uses_canonical_quad_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before_halves = torch.zeros((2, 4), dtype=torch.int64)
    after_halves = before_halves.clone()
    after_halves[1, 0] = 1  # --/+ rail of logical weight (row=0, column=0).
    before_paired = torch.zeros((2, 4), dtype=torch.int64)
    after_paired = before_paired.clone()
    after_paired[0, 3] = 1  # +/- rail of logical weight (row=0, column=1).
    zero_weight = torch.zeros((1, 2), dtype=torch.float32)
    views = iter(
        (
            SimpleNamespace(
                requested_index=(before_halves, before_paired),
                realized_normalized_weight=(zero_weight, zero_weight),
            ),
            SimpleNamespace(
                requested_index=(after_halves, after_paired),
                realized_normalized_weight=(zero_weight, zero_weight),
            ),
        )
    )
    monkeypatch.setattr(
        runtime,
        "quantize_winsorized_logical_master",
        lambda *_args, **_kwargs: next(views),
    )

    report = runtime._code_change_report(
        (zero_weight, zero_weight),
        (zero_weight, zero_weight),
        (SimpleNamespace(layout="halves"), SimpleNamespace(layout="paired")),
    )
    assert [
        layer["logical_weights_with_any_requested_code_change"]
        for layer in report["layers"]
    ] == [1, 1]
    assert [
        layer["physical_requested_codes_changed"] for layer in report["layers"]
    ] == [1, 1]
