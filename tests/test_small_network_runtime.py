from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ebl.cli import (
    ImportLegacyCheckpointRequest,
    LinspaceRequest,
    TrainRequest,
    ValidateRequest,
)
from experiments.definitions import get_definition
from experiments.schema import RunMode
from experiments.small_network.components import build_model_stack
from experiments.small_network.config import (
    ComponentSettings,
    parse_small_drn_config,
)
from experiments.small_network.runtime import (
    TrainingEpochReport,
    TrainingOutcome,
    execute_train,
    import_legacy_checkpoint,
    run_linspace,
    run_train,
    run_validate,
)
from training.checkpoint import NAMED_WEIGHTS_SCHEMA


_REPO_ROOT = Path(__file__).parents[1]


def _document():
    payload = json.loads(
        (_REPO_ROOT / "examples" / "small_drn" / "base.json").read_text(
            encoding="utf-8"
        )
    )
    payload["data"].update(
        {"num_points": 10, "batch_size": 2, "shuffle": False}
    )
    payload["solver"].update(
        {"inference_iterations": 1, "training_iterations": 1}
    )
    payload["solver"]["polish"]["enabled"] = False
    payload["modes"]["train"].update(
        {
            "num_epochs": 1,
            "max_batches": 1,
            "max_validation_batches": 1,
        }
    )
    payload["modes"]["linspace"]["samples"] = 2
    payload["modes"]["validate"]["sample_limit"] = 4
    return parse_small_drn_config(payload)


def _hardware_aware_document(
    *,
    std_dev: float = 0.2,
    seed: int | None = 17,
    noisy_evaluation: bool = True,
    num_epochs: int = 1,
):
    document = _document()
    return replace(
        document,
        train=replace(
            document.train,
            num_epochs=num_epochs,
            weight_modifier=ComponentSettings(
                type="add_normal",
                parameters={
                    "std_dev": std_dev,
                    "seed": seed,
                    "noisy_evaluation": noisy_evaluation,
                },
            ),
        ),
    )


def _runs(output_root: Path) -> list[Path]:
    return sorted(
        (path for path in output_root.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime_ns,
    )


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - older torch
        return torch.load(path, map_location="cpu")


def _assert_tensor_mapping_equal(actual: dict, expected: dict) -> None:
    assert actual.keys() == expected.keys()
    for key, expected_tensor in expected.items():
        torch.testing.assert_close(
            actual[key],
            expected_tensor,
            rtol=0.0,
            atol=0.0,
        )


def test_public_handlers_complete_tiny_cpu_pipeline_and_exact_continuation(
    tmp_path: Path,
) -> None:
    definition = get_definition("small_drn.v1")
    document = _document()
    output_root = tmp_path / "runs"
    train_spec = definition.resolve(document, RunMode.TRAIN)

    assert (
        run_train(
            TrainRequest(
                definition=definition,
                spec=train_spec,
                config_path=tmp_path / "config.json",
                output_dir=output_root,
                weights=None,
                base_weights=None,
                resume=None,
                command=("ebl", "train"),
            )
        )
        == 0
    )
    first_run = _runs(output_root)[0]
    first_result = _read_json(first_run / "result.json")
    assert [(item["path"], item["kind"]) for item in first_result["artifacts"]] == [
        ("checkpoints/weights.pt", "weights"),
        ("checkpoints/resume.pt", "resume"),
    ]
    assert first_result["metrics"]["selected"]["evaluation"] == "clean"
    assert (
        first_result["metrics"]["evaluation_protocol"]["validation_effective"]
        == "held_out_test"
    )
    resume_payload = _torch_load(first_run / "checkpoints" / "resume.pt")
    assert resume_payload["selected_weights"] is not None
    assert resume_payload["runtime_state_state"] is not None

    # The target epoch is already complete.  The continuation must still
    # materialize the pre-resume selected snapshot and a fresh full resume.
    assert (
        run_train(
            TrainRequest(
                definition=definition,
                spec=train_spec,
                config_path=tmp_path / "config.json",
                output_dir=output_root,
                weights=None,
                base_weights=None,
                resume=first_run / "checkpoints" / "resume.pt",
                command=("ebl", "train", "--resume"),
            )
        )
        == 0
    )
    continued_run = _runs(output_root)[1]
    first_weights = _torch_load(first_run / "checkpoints" / "weights.pt")
    continued_weights = _torch_load(
        continued_run / "checkpoints" / "weights.pt"
    )
    assert first_weights["metadata"] == continued_weights["metadata"]
    assert first_weights["weights"].keys() == continued_weights["weights"].keys()
    for key in first_weights["weights"]:
        torch.testing.assert_close(
            first_weights["weights"][key],
            continued_weights["weights"][key],
            rtol=0.0,
            atol=0.0,
        )

    selected_path = continued_run / "checkpoints" / "weights.pt"
    linspace_spec = definition.resolve(document, RunMode.LINSPACE)
    assert (
        run_linspace(
            LinspaceRequest(
                definition=definition,
                spec=linspace_spec,
                config_path=tmp_path / "config.json",
                output_dir=output_root,
                weights=selected_path,
                command=("ebl", "linspace"),
            )
        )
        == 0
    )
    linspace_run = _runs(output_root)[2]
    linspace_result = _read_json(linspace_run / "result.json")
    assert linspace_result["metrics"]["examples"] == 4
    assert (
        linspace_result["metrics"]["input_protocol"][
            "legacy_test_assignment_effect"
        ]
        == "no_op"
    )
    with np.load(
        linspace_run / "artifacts" / "linspace_iteration_counts.npz"
    ) as values:
        assert values["iteration_grid"].shape == (2, 2)

    validate_spec = definition.resolve(document, RunMode.VALIDATE)
    assert (
        run_validate(
            ValidateRequest(
                definition=definition,
                spec=validate_spec,
                config_path=tmp_path / "config.json",
                output_dir=output_root,
                weights=selected_path,
                command=("ebl", "validate"),
            )
        )
        == 0
    )
    validate_run = _runs(output_root)[3]
    validate_result = _read_json(validate_run / "result.json")
    assert validate_result["metrics"]["evaluation"] == "clean"
    assert validate_result["metrics"]["split_protocol"] == {
        "requested": "test",
        "effective": "held_out_test",
    }
    assert validate_result["metrics"]["examples"] <= 4


def test_execute_train_reports_every_epoch_independent_of_log_every(
    tmp_path: Path,
) -> None:
    definition = get_definition("small_drn.v1")
    document = _document()
    document = replace(
        document,
        train=replace(document.train, num_epochs=3, log_every=10),
    )
    spec = definition.resolve(document, RunMode.TRAIN)
    reports: list[TrainingEpochReport] = []

    outcome = execute_train(
        TrainRequest(
            definition=definition,
            spec=spec,
            config_path=tmp_path / "config.json",
            output_dir=tmp_path / "runs",
            weights=None,
            base_weights=None,
            resume=None,
            command=("agent", "train"),
        ),
        observers=(reports.append,),
    )

    assert isinstance(outcome, TrainingOutcome)
    assert [report.completed_epochs for report in reports] == [1, 2, 3]
    assert [report.epoch_index for report in reports] == [0, 1, 2]
    assert outcome.last_epoch_report is reports[-1]
    assert outcome.selected_validation_accuracy is not None
    assert 0.0 <= outcome.selected_validation_accuracy <= 1.0
    assert all(
        0.0 <= report.validation_error_fraction <= 1.0
        and 0.0 <= report.validation_accuracy <= 1.0
        for report in reports
    )
    # log_every=10 still writes the terminal metric only; observers are
    # deliberately independent and saw all three epoch boundaries.
    assert (outcome.run_dir / "metrics.jsonl").read_text(
        encoding="utf-8"
    ).count("\n") == 1
    with pytest.raises(FrozenInstanceError):
        reports[0].completed_epochs = 99


def test_observer_exception_marks_run_failed_and_is_reraised(
    tmp_path: Path,
) -> None:
    definition = get_definition("small_drn.v1")
    spec = definition.resolve(_document(), RunMode.TRAIN)

    class StopTrial(RuntimeError):
        pass

    def stop(_report: TrainingEpochReport) -> None:
        raise StopTrial("pruned by observer")

    output_root = tmp_path / "runs"
    with pytest.raises(StopTrial, match="pruned by observer"):
        execute_train(
            TrainRequest(
                definition=definition,
                spec=spec,
                config_path=tmp_path / "config.json",
                output_dir=output_root,
                weights=None,
                base_weights=None,
                resume=None,
                command=("agent", "train"),
            ),
            observers=(stop,),
        )

    run_dir = _runs(output_root)[0]
    status = _read_json(run_dir / "status.json")
    assert status["status"] == "failed"
    assert status["error"] == {
        "type": "StopTrial",
        "message": "pruned by observer",
    }
    assert not (run_dir / "result.json").exists()
    # The callback runs after the completed epoch boundary is resumable.
    assert (run_dir / "checkpoints" / "resume.pt").is_file()


def test_legacy_import_is_explicit_and_produces_named_weights(
    tmp_path: Path,
) -> None:
    definition = get_definition("small_drn.v1")
    document = _document()
    stack = build_model_stack(document.common)
    source = tmp_path / "legacy.pt"
    torch.save(
        [
            binding.state.detach().cpu().clone()
            for binding in stack.bundle.catalog.checkpointed
        ],
        source,
    )
    output = tmp_path / "named.pt"

    assert (
        import_legacy_checkpoint(
            ImportLegacyCheckpointRequest(
                definition=definition,
                document=document,
                config_path=tmp_path / "config.json",
                source=source,
                output=output,
                kind="full",
                command=("ebl", "checkpoint", "import-legacy"),
            )
        )
        == 0
    )
    payload = _torch_load(output)
    assert payload["schema"] == NAMED_WEIGHTS_SCHEMA
    assert payload["metadata"]["legacy_profile"] == "full"
    assert payload["metadata"]["source_sha256"]


def test_split_run_restores_carried_equilibrium_state_exactly(
    tmp_path: Path,
) -> None:
    definition = get_definition("small_drn.v1")
    one_epoch_document = _document()
    two_epoch_document = replace(
        one_epoch_document,
        train=replace(one_epoch_document.train, num_epochs=2),
    )
    one_epoch_spec = definition.resolve(
        one_epoch_document,
        RunMode.TRAIN,
    )
    two_epoch_spec = definition.resolve(
        two_epoch_document,
        RunMode.TRAIN,
    )

    uninterrupted_root = tmp_path / "uninterrupted"
    run_train(
        TrainRequest(
            definition=definition,
            spec=two_epoch_spec,
            config_path=tmp_path / "two.json",
            output_dir=uninterrupted_root,
            weights=None,
            base_weights=None,
            resume=None,
            command=("ebl", "train"),
        )
    )
    uninterrupted = _torch_load(
        _runs(uninterrupted_root)[0] / "checkpoints" / "resume.pt"
    )

    split_root = tmp_path / "split"
    run_train(
        TrainRequest(
            definition=definition,
            spec=one_epoch_spec,
            config_path=tmp_path / "one.json",
            output_dir=split_root,
            weights=None,
            base_weights=None,
            resume=None,
            command=("ebl", "train"),
        )
    )
    boundary = _runs(split_root)[0] / "checkpoints" / "resume.pt"
    run_train(
        TrainRequest(
            definition=definition,
            spec=two_epoch_spec,
            config_path=tmp_path / "two.json",
            output_dir=split_root,
            weights=None,
            base_weights=None,
            resume=boundary,
            command=("ebl", "train", "--resume"),
        )
    )
    resumed = _torch_load(
        _runs(split_root)[1] / "checkpoints" / "resume.pt"
    )

    assert uninterrupted["epoch"] == resumed["epoch"] == 2
    assert uninterrupted["global_step"] == resumed["global_step"] == 2
    assert uninterrupted["metadata"]["resume_config_sha256"] == (
        resumed["metadata"]["resume_config_sha256"]
    )
    assert uninterrupted["runtime_state_state"]["layers"].keys() == (
        resumed["runtime_state_state"]["layers"].keys()
    )
    for key, expected in uninterrupted["weights"]["weights"].items():
        torch.testing.assert_close(
            resumed["weights"]["weights"][key],
            expected,
            rtol=0.0,
            atol=0.0,
        )
    for key, expected in uninterrupted["runtime_state_state"]["layers"].items():
        torch.testing.assert_close(
            resumed["runtime_state_state"]["layers"][key],
            expected,
            rtol=0.0,
            atol=0.0,
        )


def test_hardware_aware_split_resume_restores_modifier_and_layer_state_exactly(
    tmp_path: Path,
) -> None:
    definition = get_definition("small_drn.v1")
    one_epoch_spec = definition.resolve(
        _hardware_aware_document(num_epochs=1),
        RunMode.TRAIN,
    )
    two_epoch_spec = definition.resolve(
        _hardware_aware_document(num_epochs=2),
        RunMode.TRAIN,
    )

    uninterrupted_root = tmp_path / "uninterrupted"
    run_train(
        TrainRequest(
            definition=definition,
            spec=two_epoch_spec,
            config_path=tmp_path / "two.json",
            output_dir=uninterrupted_root,
            weights=None,
            base_weights=None,
            resume=None,
            command=("ebl", "train"),
        )
    )
    uninterrupted = _torch_load(
        _runs(uninterrupted_root)[0] / "checkpoints" / "resume.pt"
    )

    split_root = tmp_path / "split"
    run_train(
        TrainRequest(
            definition=definition,
            spec=one_epoch_spec,
            config_path=tmp_path / "one.json",
            output_dir=split_root,
            weights=None,
            base_weights=None,
            resume=None,
            command=("ebl", "train"),
        )
    )
    boundary = _runs(split_root)[0] / "checkpoints" / "resume.pt"
    run_train(
        TrainRequest(
            definition=definition,
            spec=two_epoch_spec,
            config_path=tmp_path / "two.json",
            output_dir=split_root,
            weights=None,
            base_weights=None,
            resume=boundary,
            command=("ebl", "train", "--resume"),
        )
    )
    resumed = _torch_load(
        _runs(split_root)[1] / "checkpoints" / "resume.pt"
    )

    assert uninterrupted["epoch"] == resumed["epoch"] == 2
    assert uninterrupted["global_step"] == resumed["global_step"] == 2
    assert uninterrupted["progress_state"] == resumed["progress_state"]
    _assert_tensor_mapping_equal(
        resumed["weights"]["weights"],
        uninterrupted["weights"]["weights"],
    )
    _assert_tensor_mapping_equal(
        resumed["runtime_state_state"]["layers"],
        uninterrupted["runtime_state_state"]["layers"],
    )
    assert resumed["modifier_state"]["config"] == (
        uninterrupted["modifier_state"]["config"]
    )
    assert resumed["modifier_state"]["resolved_seed"] == (
        uninterrupted["modifier_state"]["resolved_seed"]
    )
    for stream in ("train", "evaluation"):
        _assert_tensor_mapping_equal(
            resumed["modifier_state"]["generator_states"][stream],
            uninterrupted["modifier_state"]["generator_states"][stream],
        )


def test_noisy_evaluation_is_diagnostic_and_does_not_change_continuation(
    tmp_path: Path,
) -> None:
    definition = get_definition("small_drn.v1")
    noisy_spec = definition.resolve(
        _hardware_aware_document(
            noisy_evaluation=True,
            num_epochs=2,
        ),
        RunMode.TRAIN,
    )
    clean_only_spec = definition.resolve(
        _hardware_aware_document(
            noisy_evaluation=False,
            num_epochs=2,
        ),
        RunMode.TRAIN,
    )

    outcomes = []
    checkpoints = []
    results = []
    for name, spec in (("noisy", noisy_spec), ("clean-only", clean_only_spec)):
        outcome = execute_train(
            TrainRequest(
                definition=definition,
                spec=spec,
                config_path=tmp_path / f"{name}.json",
                output_dir=tmp_path / name,
                weights=None,
                base_weights=None,
                resume=None,
                command=("ebl", "train"),
            )
        )
        outcomes.append(outcome)
        checkpoints.append(_torch_load(outcome.resume_path))
        results.append(_read_json(outcome.result_path)["metrics"])

    noisy_checkpoint, clean_only_checkpoint = checkpoints
    _assert_tensor_mapping_equal(
        noisy_checkpoint["weights"]["weights"],
        clean_only_checkpoint["weights"]["weights"],
    )
    _assert_tensor_mapping_equal(
        noisy_checkpoint["runtime_state_state"]["layers"],
        clean_only_checkpoint["runtime_state_state"]["layers"],
    )
    _assert_tensor_mapping_equal(
        noisy_checkpoint["modifier_state"]["generator_states"]["train"],
        clean_only_checkpoint["modifier_state"]["generator_states"]["train"],
    )
    assert noisy_checkpoint["progress_state"] == (
        clean_only_checkpoint["progress_state"]
    )
    assert outcomes[0].selected_validation_cost == (
        outcomes[1].selected_validation_cost
    )
    assert results[0]["selected"] == results[1]["selected"]
    assert results[0]["last_validation"] == results[1]["last_validation"]
    assert results[0]["last_noisy_validation"] is not None
    assert results[1]["last_noisy_validation"] is None


def test_zero_noise_is_numerically_identical_to_unmodified_training(
    tmp_path: Path,
) -> None:
    definition = get_definition("small_drn.v1")
    base_spec = definition.resolve(_document(), RunMode.TRAIN)
    zero_spec = definition.resolve(
        _hardware_aware_document(
            std_dev=0.0,
            noisy_evaluation=True,
        ),
        RunMode.TRAIN,
    )

    checkpoints = []
    results = []
    for name, spec in (("base", base_spec), ("zero", zero_spec)):
        outcome = execute_train(
            TrainRequest(
                definition=definition,
                spec=spec,
                config_path=tmp_path / f"{name}.json",
                output_dir=tmp_path / name,
                weights=None,
                base_weights=None,
                resume=None,
                command=("ebl", "train"),
            )
        )
        checkpoints.append(_torch_load(outcome.resume_path))
        results.append(_read_json(outcome.result_path)["metrics"])

    base_checkpoint, zero_checkpoint = checkpoints
    assert base_checkpoint["modifier_state"] is None
    assert zero_checkpoint["modifier_state"] is None
    _assert_tensor_mapping_equal(
        zero_checkpoint["weights"]["weights"],
        base_checkpoint["weights"]["weights"],
    )
    _assert_tensor_mapping_equal(
        zero_checkpoint["runtime_state_state"]["layers"],
        base_checkpoint["runtime_state_state"]["layers"],
    )
    assert zero_checkpoint["progress_state"] == base_checkpoint["progress_state"]
    assert results[1]["selected"] == results[0]["selected"]
    assert results[1]["last_validation"] == results[0]["last_validation"]
    assert results[1]["last_noisy_validation"] is None
    modifier_protocol = results[1]["evaluation_protocol"][
        "parameter_modifier"
    ]
    assert not modifier_protocol["active_during_training"]
    assert not modifier_protocol["noisy_evaluation_executed"]


def test_resume_rejects_numerical_config_changes_and_shorter_horizon(
    tmp_path: Path,
) -> None:
    definition = get_definition("small_drn.v1")
    document = _document()
    original_spec = definition.resolve(document, RunMode.TRAIN)
    output_root = tmp_path / "original"
    run_train(
        TrainRequest(
            definition=definition,
            spec=original_spec,
            config_path=tmp_path / "original.json",
            output_dir=output_root,
            weights=None,
            base_weights=None,
            resume=None,
            command=("ebl", "train"),
        )
    )
    boundary = _runs(output_root)[0] / "checkpoints" / "resume.pt"

    changed_document = replace(
        document,
        train=replace(document.train, nudging=0.125, num_epochs=2),
    )
    changed_spec = definition.resolve(changed_document, RunMode.TRAIN)
    with pytest.raises(ValueError, match="numerical configuration"):
        run_train(
            TrainRequest(
                definition=definition,
                spec=changed_spec,
                config_path=tmp_path / "changed.json",
                output_dir=tmp_path / "changed",
                weights=None,
                base_weights=None,
                resume=boundary,
                command=("ebl", "train", "--resume"),
            )
        )

    payload = _torch_load(boundary)
    payload["epoch"] = 2
    future_boundary = tmp_path / "future-resume.pt"
    torch.save(payload, future_boundary)
    with pytest.raises(ValueError, match="greater than or equal"):
        run_train(
            TrainRequest(
                definition=definition,
                spec=original_spec,
                config_path=tmp_path / "shorter.json",
                output_dir=tmp_path / "shorter",
                weights=None,
                base_weights=None,
                resume=future_boundary,
                command=("ebl", "train", "--resume"),
            )
        )


def test_hardware_aware_runtime_records_clean_selection_and_noisy_diagnostic(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from experiments.small_network import runtime as runtime_module

    definition = get_definition("small_drn.v1")
    spec = definition.resolve(
        _hardware_aware_document(),
        RunMode.TRAIN,
    )
    observed_splits: list[str] = []
    original_evaluate = runtime_module.evaluate

    def observe_evaluation(*args, **kwargs):
        observed_splits.append(kwargs["split"])
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(runtime_module, "evaluate", observe_evaluation)
    output_root = tmp_path / "runs"
    run_train(
        TrainRequest(
            definition=definition,
            spec=spec,
            config_path=tmp_path / "config.json",
            output_dir=output_root,
            weights=None,
            base_weights=None,
            resume=None,
            command=("ebl", "train"),
        )
    )

    run_dir = _runs(output_root)[0]
    result = _read_json(run_dir / "result.json")
    protocol = result["metrics"]["evaluation_protocol"]
    assert observed_splits == ["validation_noisy", "validation"]
    assert result["metrics"]["selected"]["evaluation"] == "clean"
    assert result["metrics"]["last_validation"] is not None
    assert result["metrics"]["last_noisy_validation"] is not None
    assert protocol["selection_evaluation"] == "clean"
    assert protocol["noisy_evaluation_order"] == "before_clean"
    assert protocol["parameter_modifier"] == {
        "type": "add_normal",
        "parameters": {
            "std_dev": 0.2,
            "seed": 17,
            "noisy_evaluation": True,
        },
        "active_during_training": True,
        "resolved_seed": 17,
        "noisy_evaluation_requested": True,
        "noisy_evaluation_executed": True,
    }
    resume = _torch_load(run_dir / "checkpoints" / "resume.pt")
    assert resume["modifier_state"]["resolved_seed"] == 17
    assert set(resume["modifier_state"]["generator_states"]) == {
        "train",
        "evaluation",
    }
    assert set(resume["modifier_state"]["generator_states"]["train"]) == {
        "cpu"
    }
    assert set(
        resume["modifier_state"]["generator_states"]["evaluation"]
    ) == {"cpu"}


def test_resumed_result_reports_checkpoint_resolved_modifier_seed(
    tmp_path: Path,
) -> None:
    definition = get_definition("small_drn.v1")
    document = _hardware_aware_document(seed=None)
    document = replace(
        document,
        common=replace(
            document.common,
            runtime=replace(document.common.runtime, seed=None),
        ),
    )
    spec = definition.resolve(document, RunMode.TRAIN)

    torch.manual_seed(101)
    first = execute_train(
        TrainRequest(
            definition=definition,
            spec=spec,
            config_path=tmp_path / "config.json",
            output_dir=tmp_path / "first",
            weights=None,
            base_weights=None,
            resume=None,
            command=("ebl", "train"),
        )
    )
    assert _torch_load(first.resume_path)["modifier_state"][
        "resolved_seed"
    ] == 101

    torch.manual_seed(202)
    resumed = execute_train(
        TrainRequest(
            definition=definition,
            spec=spec,
            config_path=tmp_path / "config.json",
            output_dir=tmp_path / "resumed",
            weights=None,
            base_weights=None,
            resume=first.resume_path,
            command=("ebl", "train", "--resume"),
        )
    )
    protocol = _read_json(resumed.result_path)["metrics"][
        "evaluation_protocol"
    ]["parameter_modifier"]

    assert protocol["parameters"]["seed"] is None
    assert protocol["resolved_seed"] == 101


def test_float64_stack_keeps_configured_dtype_across_network_reset() -> None:
    document = _document()
    common = replace(
        document.common,
        runtime=replace(document.common.runtime, dtype="float64"),
    )
    stack = build_model_stack(common)
    assert {
        binding.state.dtype for binding in stack.bundle.catalog.all
    } == {torch.float64}

    inputs = torch.zeros((3, stack.logical_input_dim), dtype=torch.float32)
    stack.network.set_input(inputs, reset=True)
    assert {layer.state.dtype for layer in stack.network.layers()} == {
        torch.float64
    }
