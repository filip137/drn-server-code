from __future__ import annotations

from dataclasses import replace
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


def test_unimplemented_extension_fails_before_model_construction(
    tmp_path: Path,
) -> None:
    definition = get_definition("small_drn.v1")
    spec = definition.resolve(_document(), RunMode.TRAIN)
    unsupported = replace(
        spec,
        settings=replace(
            spec.settings,
            weight_modifier=ComponentSettings(
                type="add_normal",
                parameters={},
            ),
        ),
    )
    with pytest.raises(NotImplementedError, match="hardware-aware branch"):
        run_train(
            TrainRequest(
                definition=definition,
                spec=unsupported,
                config_path=tmp_path / "config.json",
                output_dir=tmp_path / "runs",
                weights=None,
                base_weights=None,
                resume=None,
                command=("ebl", "train"),
            )
        )
    failed_run = _runs(tmp_path / "runs")[0]
    assert _read_json(failed_run / "status.json")["status"] == "failed"


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
