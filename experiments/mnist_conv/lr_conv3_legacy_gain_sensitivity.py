"""Controlled Conv3 legacy input-gain sensitivity rerun.

This one-point study reuses the exact seven-parameter learning-rate vector from
the completed ordinary-MNIST legacy rescue at ``rho=(5e-5, 3e-4)``.  It changes
only ``input_gain`` from 665.0302124023 to 200.0, without re-probing the rates or
mutating either the v7 parent study or its rescue.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .identity import (
    code_provenance,
    normalize_code_provenance,
    sha256_file,
    sha256_json,
)
from .io import atomic_write_json, read_json
from .lr_engine import (
    build_loader_bundle,
    build_model_runtime,
    parameter_state_diagnostics,
    parameter_tensor_digest,
)
from .lr_stages import (
    _CandidateStepLoop,
    _architecture_checkpoint,
    _batch_hash,
    _bounded_initial_rms_scales,
    execute_candidate_entry,
)
from .lr_study_spec import LRStudySpec


SCHEMA_VERSION = "mnist-conv-lr-conv3-legacy-gain-sensitivity/v1"
ID_SCHEMA_VERSION = "mnist-conv-lr-conv3-legacy-gain-sensitivity-id/v1"
MANIFEST_SCHEMA_VERSION = (
    "mnist-conv-lr-conv3-legacy-gain-sensitivity-manifest/v1"
)
PREFLIGHT_SCHEMA_VERSION = (
    "mnist-conv-lr-conv3-legacy-gain-sensitivity-preflight/v1"
)
RUN_SPEC_SCHEMA_VERSION = (
    "mnist-conv-lr-conv3-legacy-gain-sensitivity-run/v1"
)
RESULT_SCHEMA_VERSION = (
    "mnist-conv-lr-conv3-legacy-gain-sensitivity-result/v1"
)
WRAPPER_SCHEMA_VERSION = (
    "mnist-conv-lr-conv3-legacy-gain-sensitivity-summary/v1"
)
COMPLETION_SCHEMA_VERSION = (
    "mnist-conv-lr-conv3-legacy-gain-sensitivity-completion/v1"
)
PUBLICATION_RETRY_SCHEMA_VERSION = (
    "mnist-conv-lr-conv3-legacy-gain-sensitivity-publication-retry/v1"
)
PROTOCOL_ID = "conv-hardsigmoid-lr-conv3-legacy-gain-sensitivity-bs16-v1"
ENTRY_ID = "conv3_legacy_v4_c0p25--legacy-gain200-source-lower10x"
STAGE = "gain200_rerun"
PUBLICATION_RETRY_PHASE = "result_retry_1"

_TOP_LEVEL_KEYS = {
    "schema_version",
    "name",
    "protocol_id",
    "source",
    "controlled_change",
    "training",
    "artifacts",
}

_EXPECTED_SOURCE = {
    "parent_study_schema_version": "mnist-conv-lr-study/v7",
    "parent_study_id": (
        "lrstudy_6e56b399cf3e522a2c3f39f66a51611147b6171dd3b9e223f20791bc6275caf0"
    ),
    "parent_row_id": "conv3_legacy_v4_c0p25",
    "rescue_id": (
        "lrrescue_e9c1d08dc339f723f61d3d73b8059fe5f82fe95952a5d1fa06107ed877141a04"
    ),
    "rescue_stage": "promotion_retry_1",
    "rescue_entry_id": (
        "conv3_legacy_v4_c0p25--legacy-rescue-lower-10x"
    ),
    "source_summary_sha256": (
        "0a768affc2da66e33cfc48af11181c240e38035f6f632ba0ad41703ca44725b5"
    ),
    "source_rescue_summary_sha256": (
        "f764aa2b19eb18fd9fc1da77db0d197f4c8e7403430140946f7f5dae1d62fc04"
    ),
    "source_completion_sha256": (
        "64fb01989444df8109d85ff7c68855a3153fa892d8d3f759e6d034cb31a5a8f8"
    ),
    "source_input_gain": 665.0302124023,
    "source_final_validation_accuracy": 0.7638,
    "rho_conv_label": 5e-5,
    "rho_dense_label": 3e-4,
}

_EXPECTED_CONTROLLED_CHANGE = {
    "field": "input_gain",
    "value": 200.0,
    "reason": "recorded_best_legacy_low_gain_diagnostic",
    "reuse_exact_source_learning_rate_vector": True,
    "reprobe_learning_rates": False,
    "rho_labels_are_source_provenance_only": True,
    "all_other_row_fields_unchanged": True,
}

_EXPECTED_TRAINING = {
    "model_seed": 0,
    "epochs": 3,
    "steps_per_epoch": 3438,
    "total_steps": 10314,
    "batch_size": 16,
    "validation_batch_size": 64,
    "minimum_final_validation_accuracy": 0.9,
    "accuracy_boundary": "greater_than_or_equal",
    "schedule": "constant_seven_parameter_vector",
    "optimizer": "sgd_no_momentum_no_weight_decay",
    "one_step_gpu_smoke_required": True,
}

_EXPECTED_ARTIFACTS = {
    "hash": "sha256",
    "immutable_manifest": True,
    "completion_marker_written_last": True,
    "resume_completed_work": True,
    "mutate_parent_study": False,
    "mutate_source_rescue": False,
    "official_test_read": False,
}


def _error(expected: str, provided: Any, path: str) -> ValueError:
    return ValueError(
        f"Expected {path} to be {expected}. Provided value: {provided!r}."
    )


def _assert_exact(expected: Any, provided: Any, path: str) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(provided, Mapping) or set(provided) != set(expected):
            raise _error(
                f"an object with exactly keys {sorted(expected)!r}",
                provided,
                path,
            )
        for key, value in expected.items():
            _assert_exact(value, provided[key], f"{path}.{key}")
        return
    if expected != provided:
        raise _error(f"exactly {expected!r}", provided, path)


def _reject_non_finite(value: Any, path: str = "sensitivity") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise _error("finite JSON", value, path)
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_non_finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_non_finite(item, f"{path}[{index}]")


@dataclass(frozen=True)
class GainSensitivitySpec:
    """Strict immutable contract for the one-point gain change."""

    data: dict[str, Any]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GainSensitivitySpec":
        if not isinstance(value, Mapping):
            raise _error("a JSON object", value, "sensitivity")
        data = copy.deepcopy(dict(value))
        _reject_non_finite(data)
        if set(data) != _TOP_LEVEL_KEYS:
            raise _error(
                f"an object with exactly keys {sorted(_TOP_LEVEL_KEYS)!r}",
                {
                    "missing": sorted(_TOP_LEVEL_KEYS - set(data)),
                    "extra": sorted(set(data) - _TOP_LEVEL_KEYS),
                },
                "sensitivity",
            )
        if data["schema_version"] != SCHEMA_VERSION:
            raise _error(
                f"exactly {SCHEMA_VERSION!r}",
                data["schema_version"],
                "sensitivity.schema_version",
            )
        if data["protocol_id"] != PROTOCOL_ID:
            raise _error(
                f"exactly {PROTOCOL_ID!r}",
                data["protocol_id"],
                "sensitivity.protocol_id",
            )
        if not isinstance(data["name"], str) or not data["name"].strip():
            raise _error(
                "a non-empty string", data["name"], "sensitivity.name"
            )
        data["name"] = data["name"].strip()
        for key, expected in (
            ("source", _EXPECTED_SOURCE),
            ("controlled_change", _EXPECTED_CONTROLLED_CHANGE),
            ("training", _EXPECTED_TRAINING),
            ("artifacts", _EXPECTED_ARTIFACTS),
        ):
            _assert_exact(expected, data[key], f"sensitivity.{key}")
        return cls(data)

    @classmethod
    def from_path(cls, path: str | Path) -> "GainSensitivitySpec":
        source = Path(path).expanduser().resolve()
        try:
            value = json.loads(source.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Expected sensitivity config to contain strict JSON. "
                f"Provided value: {source}: {exc}."
            ) from exc
        return cls.from_dict(value)

    @property
    def sensitivity_id(self) -> str:
        payload = copy.deepcopy(self.data)
        payload.pop("name")
        digest = sha256_json(
            {
                "schema_version": ID_SCHEMA_VERSION,
                "sensitivity_spec": payload,
            }
        )
        return f"lrgain_{digest}"


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or read_json(path) != value:
            raise RuntimeError(
                "Expected existing immutable JSON to have identical content. "
                f"Provided value: {path}."
            )
        return
    atomic_write_json(path, value, canonical=True)


def _record(root: Path, path: str) -> dict[str, Any]:
    target = root / path
    if target.is_symlink() or not target.is_file():
        raise RuntimeError(
            "Expected sensitivity input to be an existing regular file. "
            f"Provided value: {target}."
        )
    return {
        "path": path,
        "sha256": sha256_file(target),
        "bytes": target.stat().st_size,
    }


def _verify_records(root: Path, records: Sequence[Mapping[str, Any]]) -> None:
    for expected in records:
        observed = _record(root, str(expected["path"]))
        normalized = {
            "path": str(expected["path"]),
            "sha256": str(expected["sha256"]),
            "bytes": int(expected["bytes"]),
        }
        if observed != normalized:
            raise RuntimeError(
                "Expected immutable sensitivity input to remain hash-identical. "
                f"Provided value: expected={normalized!r}, observed={observed!r}."
            )


def _source_paths(
    source_entry: Path,
) -> tuple[Path, Path, Path, Path]:
    return (
        source_entry / "summary.json",
        source_entry / "rescue_summary.json",
        source_entry / "complete.json",
        source_entry / "minibatches.json",
    )


def _validate_source(
    spec: GainSensitivitySpec,
    parent_root: Path,
    source_entry: Path,
) -> tuple[LRStudySpec, dict[str, Any], dict[str, Any], dict[str, Any]]:
    parent = LRStudySpec.from_path(parent_root / "study.resolved.json")
    source = spec.data["source"]
    if (
        parent.study_id != source["parent_study_id"]
        or parent.data["schema_version"]
        != source["parent_study_schema_version"]
    ):
        raise RuntimeError(
            "Expected the exact v7 parent study. Provided value: "
            f"schema={parent.data['schema_version']!r}, id={parent.study_id!r}."
        )
    summary_path, rescue_path, completion_path, minibatches_path = (
        _source_paths(source_entry)
    )
    expected_hashes = (
        (summary_path, source["source_summary_sha256"], "source summary"),
        (
            rescue_path,
            source["source_rescue_summary_sha256"],
            "source rescue summary",
        ),
        (
            completion_path,
            source["source_completion_sha256"],
            "source completion",
        ),
    )
    for path, expected, label in expected_hashes:
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(
                f"Expected {label} SHA-256 to be {expected!r}. "
                f"Provided value: {observed!r}."
            )
    summary = read_json(summary_path)
    rescue_summary = read_json(rescue_path)
    minibatches = read_json(minibatches_path)
    row = summary["row"]
    rates = summary["learning_rates_by_parameter"]
    expected_rate_names = {
        "ConvWeight_0",
        "Bias_0",
        "ConvWeight_1",
        "Bias_1",
        "ConvWeight_2",
        "Bias_2",
        "DenseWeight_0",
    }
    observed_source = {
        "row_id": row.get("row_id"),
        "input_gain": float(row.get("input_gain")),
        "rho_conv": float(summary.get("rho_conv")),
        "rho_dense": float(summary.get("rho_dense")),
        "final_validation_accuracy": float(
            summary.get("final_validation_accuracy")
        ),
        "rescue_id": rescue_summary.get("rescue_id"),
        "rescue_entry_id": rescue_summary.get("entry_id"),
        "training_completed": summary.get("training_completed"),
        "safety_gate_failure": summary.get("safety_gate_failure"),
        "rate_names": set(rates),
    }
    expected_source = {
        "row_id": source["parent_row_id"],
        "input_gain": float(source["source_input_gain"]),
        "rho_conv": float(source["rho_conv_label"]),
        "rho_dense": float(source["rho_dense_label"]),
        "final_validation_accuracy": float(
            source["source_final_validation_accuracy"]
        ),
        "rescue_id": source["rescue_id"],
        "rescue_entry_id": source["rescue_entry_id"],
        "training_completed": True,
        "safety_gate_failure": None,
        "rate_names": expected_rate_names,
    }
    if observed_source != expected_source:
        raise RuntimeError(
            "Expected the completed clean 76.38% rescue source. "
            f"Provided value: {observed_source!r}."
        )
    if (
        rates["Bias_0"] != rates["ConvWeight_0"]
        or rates["Bias_1"] != rates["ConvWeight_1"]
        or rates["Bias_2"] != rates["ConvWeight_2"]
    ):
        raise RuntimeError(
            "Expected all three source biases to retain tied Conv rates. "
            f"Provided value: {rates!r}."
        )
    return parent, summary, rescue_summary, minibatches


def plan_sensitivity(
    *,
    config: str | Path,
    parent_study: str | Path,
    source_entry: str | Path,
    results_root: str | Path,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a content-addressed one-entry gain-sensitivity manifest."""

    spec = GainSensitivitySpec.from_path(config)
    parent_root = Path(parent_study).expanduser().resolve()
    source_root = Path(source_entry).expanduser().resolve()
    parent, summary, _, source_minibatches = _validate_source(
        spec, parent_root, source_root
    )
    source_row = dict(summary["row"])
    controlled_row = copy.deepcopy(source_row)
    controlled_row["input_gain"] = float(
        spec.data["controlled_change"]["value"]
    )
    rates = {
        str(name): float(value)
        for name, value in summary["learning_rates_by_parameter"].items()
    }
    payload = {
        "row_id": spec.data["source"]["parent_row_id"],
        "candidate_role": "legacy-gain200-source-lower10x",
        "candidate_stage": "legacy_gain_sensitivity",
        "rho_conv": float(spec.data["source"]["rho_conv_label"]),
        "rho_dense": float(spec.data["source"]["rho_dense_label"]),
        "rho_labels_are_source_provenance_only": True,
        "median_units_by_weight": summary["median_units_by_weight"],
        "learning_rates_by_parameter": rates,
        "learning_rates_by_weight": summary["learning_rates_by_weight"],
        "peak_learning_rate": max(rates.values()),
        "source_row": source_row,
        "controlled_row": controlled_row,
        "source_final_validation_accuracy": float(
            summary["final_validation_accuracy"]
        ),
        "source_minibatch_order_sha256": summary[
            "minibatch_order_sha256"
        ],
        "source_first_minibatch": source_minibatches["batches"][0],
        "source_initial_parameter_tensor_sha256": summary[
            "initial_parameter_tensor_sha256"
        ],
    }
    parent_inputs = [
        _record(parent_root, relative)
        for relative in (
            "study.resolved.json",
            "split/indices.json",
            "split/provenance.json",
            "initialization/conv3.pt",
            "initialization/conv3.json",
        )
    ]
    source_inputs = [
        _record(source_root, relative)
        for relative in (
            "summary.json",
            "rescue_summary.json",
            "complete.json",
            "minibatches.json",
        )
    ]
    root = (
        Path(results_root).expanduser().resolve()
        / "gain_sensitivities"
        / f"{spec.data['name']}--{spec.sensitivity_id}"
    )
    root.mkdir(parents=True, exist_ok=True)
    _write_immutable_json(root / "sensitivity.resolved.json", spec.data)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "sensitivity_id": spec.sensitivity_id,
        "entry_id": ENTRY_ID,
        "parent_study_id": parent.study_id,
        "parent_study_dir": str(parent_root),
        "source_entry_dir": str(source_root),
        "parent_inputs": parent_inputs,
        "source_inputs": source_inputs,
        "code_provenance": normalize_code_provenance(
            provenance or code_provenance()
        ),
        "payload": payload,
        "official_test_read": False,
    }
    _write_immutable_json(root / "manifest.json", manifest)
    return {
        "status": "planned",
        "sensitivity_id": spec.sensitivity_id,
        "sensitivity_dir": str(root),
        "entry_id": ENTRY_ID,
        "source_input_gain": source_row["input_gain"],
        "controlled_input_gain": controlled_row["input_gain"],
        "learning_rates_by_parameter": rates,
    }


def _load_sensitivity(
    sensitivity_dir: str | Path,
    *,
    require_current_source: bool = True,
) -> tuple[Path, GainSensitivitySpec, dict[str, Any], LRStudySpec]:
    root = Path(sensitivity_dir).expanduser().resolve()
    spec = GainSensitivitySpec.from_path(root / "sensitivity.resolved.json")
    if not root.name.endswith(f"--{spec.sensitivity_id}"):
        raise RuntimeError(
            "Expected sensitivity directory name to end with its content address. "
            f"Provided value: {root.name!r}."
        )
    manifest = read_json(root / "manifest.json")
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or manifest.get("sensitivity_id") != spec.sensitivity_id
        or manifest.get("entry_id") != ENTRY_ID
    ):
        raise RuntimeError(
            "Expected the exact immutable sensitivity manifest. "
            f"Provided value: {manifest!r}."
        )
    planned = normalize_code_provenance(manifest["code_provenance"])
    current = normalize_code_provenance(code_provenance())
    if require_current_source and (
        planned["effective_code_fingerprint"]
        != current["effective_code_fingerprint"]
    ):
        raise RuntimeError(
            "Expected current source fingerprint to match the sensitivity "
            f"manifest. Provided value: planned={planned!r}, current={current!r}."
        )
    parent_root = Path(manifest["parent_study_dir"]).resolve()
    source_root = Path(manifest["source_entry_dir"]).resolve()
    _verify_records(parent_root, manifest["parent_inputs"])
    _verify_records(source_root, manifest["source_inputs"])
    parent, _, _, _ = _validate_source(spec, parent_root, source_root)
    return root, spec, manifest, parent


def _candidate_dir(root: Path) -> Path:
    return root / f"stages/{STAGE}/entries/{ENTRY_ID}"


def _candidate_output_names() -> tuple[str, ...]:
    return (
        "best_validation.pt",
        "final.pt",
        "minibatches.json",
        "parameter_diagnostics.csv",
        "run_spec.gain_sensitivity.json",
        "step_log.csv",
        "summary.json",
        "validation.json",
    )


def _verify_minibatch_prefix(
    *,
    candidate_minibatches: Mapping[str, Any],
    source_minibatches: Mapping[str, Any],
    training_completed: bool,
) -> dict[str, Any]:
    """Require exact source order for the observed prefix or complete run."""

    observed = candidate_minibatches.get("batches")
    expected = source_minibatches.get("batches")
    if not isinstance(observed, list) or not isinstance(expected, list):
        raise RuntimeError(
            "Expected candidate and source minibatches to be JSON lists. "
            f"Provided value: observed={type(observed).__name__}, "
            f"expected={type(expected).__name__}."
        )
    prefix_equal = observed == expected[: len(observed)]
    complete_length_equal = len(observed) == len(expected)
    if not prefix_equal or (training_completed and not complete_length_equal):
        raise RuntimeError(
            "Expected the controlled rerun to preserve the exact source minibatch "
            "order for every attempted step. Provided value: "
            f"observed_steps={len(observed)}, source_steps={len(expected)}, "
            f"prefix_equal={prefix_equal}, "
            f"training_completed={training_completed}."
        )
    return {
        "observed_steps": len(observed),
        "source_steps": len(expected),
        "prefix_equal": True,
        "complete_order_equal": complete_length_equal,
    }


def _completion_path(root: Path, phase: str) -> Path:
    return root / phase / "complete.json"


def _valid_completion(root: Path, phase: str) -> dict[str, Any] | None:
    path = _completion_path(root, phase)
    if path.is_symlink() or not path.is_file():
        return None
    completion = read_json(path)
    if (
        completion.get("schema_version") != COMPLETION_SCHEMA_VERSION
        or completion.get("phase") != phase
    ):
        return None
    for record in completion.get("outputs", []):
        output = path.parent / str(record["name"])
        if (
            output.is_symlink()
            or not output.is_file()
            or output.stat().st_size != int(record["bytes"])
            or sha256_file(output) != str(record["sha256"])
        ):
            return None
    return completion


def _publish_completion(
    root: Path,
    *,
    phase: str,
    output_dir: Path,
    required_outputs: Sequence[str],
) -> dict[str, Any]:
    records = []
    for name in sorted(required_outputs):
        path = output_dir / name
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(
                "Expected sensitivity output before completion publication. "
                f"Provided value: {path}."
            )
        records.append(
            {
                "name": name,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    completion = {
        "schema_version": COMPLETION_SCHEMA_VERSION,
        "phase": phase,
        "outputs": records,
    }
    _write_immutable_json(_completion_path(root, phase), completion)
    return completion


def run_preflight(
    *,
    sensitivity_dir: str | Path,
    data_root: str | Path,
    download: bool,
    device: str,
) -> dict[str, Any]:
    """Run one real GPU update and verify initialization and minibatch parity."""

    root, spec, manifest, parent = _load_sensitivity(sensitivity_dir)
    completed = _valid_completion(root, "preflight")
    if completed is not None:
        return {
            "status": "resumed_complete",
            "preflight": read_json(root / "preflight/preflight.json"),
        }
    payload = manifest["payload"]
    rates = payload["learning_rates_by_parameter"]
    runtime = build_model_runtime(
        parent.data,
        payload["controlled_row"],
        device=device,
        initialization_checkpoint=_architecture_checkpoint(
            Path(manifest["parent_study_dir"]), payload["source_row"]
        ),
        learning_rate=rates,
    )
    initial_digest = parameter_tensor_digest(runtime.parameters)
    initial_diagnostics = parameter_state_diagnostics(runtime.parameters)
    bounded_scales = _bounded_initial_rms_scales(initial_diagnostics)
    bundle = build_loader_bundle(
        parent.data,
        data_root=data_root,
        download=download,
        return_source_indices=True,
    )
    bundle.reset_train_shuffle()
    loop = _CandidateStepLoop(
        study=parent.data,
        runtime=runtime,
        peak_learning_rate=max(rates.values()),
        learning_rates_by_parameter=rates,
        bounded_initial_scales=bounded_scales,
        initial_diagnostics=initial_diagnostics,
        relative=True,
        layerwise=True,
        weight_median=True,
    )
    epoch_relative = {name: [] for name in bounded_scales}
    result = loop.run_step(
        next(iter(bundle.train_loader)),
        epoch=1,
        batch_in_epoch=1,
        epoch_relative_rhos=epoch_relative,
    )
    passed = (
        result is not None
        and loop.successful_steps == 1
        and loop.inadmissible_reason is None
        and initial_digest
        == payload["source_initial_parameter_tensor_sha256"]
        and list(loop.minibatches[0]) == payload["source_first_minibatch"]
    )
    device_record: dict[str, Any] = {"requested": device}
    if str(device).startswith("cuda"):
        import torch

        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        device_record.update(
            {
                "index": index,
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
            }
        )
    preflight = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "sensitivity_id": spec.sensitivity_id,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "requested_steps": 1,
        "completed_steps": loop.successful_steps,
        "inadmissible_reason": loop.inadmissible_reason,
        "initial_parameter_tensor_sha256": initial_digest,
        "expected_initial_parameter_tensor_sha256": payload[
            "source_initial_parameter_tensor_sha256"
        ],
        "first_minibatch": list(loop.minibatches[0]) if loop.minibatches else None,
        "expected_first_minibatch": payload["source_first_minibatch"],
        "device": device_record,
        "controlled_input_gain": payload["controlled_row"]["input_gain"],
        "learning_rates_by_parameter": rates,
        "official_test_read": False,
    }
    output_dir = root / "preflight"
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "preflight.json", preflight, canonical=True)
    if not passed:
        raise RuntimeError(
            "Expected the one-step gain-sensitivity GPU preflight to pass. "
            f"Provided value: {preflight!r}."
        )
    _publish_completion(
        root,
        phase="preflight",
        output_dir=output_dir,
        required_outputs=("preflight.json",),
    )
    return {"status": "complete", "preflight": preflight}


def run_sensitivity(
    *,
    sensitivity_dir: str | Path,
    data_root: str | Path,
    download: bool,
    device: str,
) -> dict[str, Any]:
    """Execute the three-epoch controlled rerun after the required smoke."""

    root, spec, manifest, parent = _load_sensitivity(sensitivity_dir)
    output_dir = root / "result"
    completed = _valid_completion(root, "result")
    if completed is not None:
        return {
            "status": "resumed_complete",
            "summary": read_json(output_dir / "sensitivity_summary.json"),
        }
    preflight_completion = _valid_completion(root, "preflight")
    if preflight_completion is None:
        raise RuntimeError(
            "Expected a valid one-step GPU preflight completion before the "
            f"three-epoch rerun. Provided value: {root / 'preflight'}."
        )
    preflight = read_json(root / "preflight/preflight.json")
    if preflight.get("passed") is not True:
        raise RuntimeError(
            "Expected the one-step GPU preflight to have passed. "
            f"Provided value: {preflight!r}."
        )
    payload = manifest["payload"]
    split = read_json(
        Path(manifest["parent_study_dir"]) / "split/provenance.json"
    )
    checkpoint = _architecture_checkpoint(
        Path(manifest["parent_study_dir"]), payload["source_row"]
    )
    run_spec = {
        "schema_version": RUN_SPEC_SCHEMA_VERSION,
        "sensitivity_id": spec.sensitivity_id,
        "entry_id": ENTRY_ID,
        "parent_study_id": parent.study_id,
        "source": spec.data["source"],
        "source_row": payload["source_row"],
        "controlled_row": payload["controlled_row"],
        "controlled_change": spec.data["controlled_change"],
        "rho_conv_label": payload["rho_conv"],
        "rho_dense_label": payload["rho_dense"],
        "rho_labels_are_source_provenance_only": True,
        "learning_rates_by_parameter": payload[
            "learning_rates_by_parameter"
        ],
        "training": spec.data["training"],
        "checkpoint_sha256": sha256_file(checkpoint),
        "initial_parameter_tensor_sha256": payload[
            "source_initial_parameter_tensor_sha256"
        ],
        "train_indices_sha256": split["train_indices_sha256"],
        "validation_indices_sha256": split["validation_indices_sha256"],
        "expected_minibatch_order_sha256": payload[
            "source_minibatch_order_sha256"
        ],
        "preflight_completion_sha256": sha256_file(
            root / "preflight/complete.json"
        ),
        "official_test_read": False,
    }
    candidate = execute_candidate_entry(
        parent.data,
        manifest["parent_study_dir"],
        spec.data["source"]["parent_row_id"],
        payload["candidate_role"],
        candidate_payload=payload,
        output_stage=STAGE,
        output_root=root,
        data_root=data_root,
        download=download,
        device=device,
        controlled_runtime_row=payload["controlled_row"],
        controlled_run_spec=run_spec,
        controlled_summary_schema=RESULT_SCHEMA_VERSION,
    )
    candidate_dir = _candidate_dir(root)
    source_minibatches = read_json(
        Path(manifest["source_entry_dir"]) / "minibatches.json"
    )
    order_check = _verify_minibatch_prefix(
        candidate_minibatches=read_json(candidate_dir / "minibatches.json"),
        source_minibatches=source_minibatches,
        training_completed=bool(candidate["training_completed"]),
    )
    if candidate["initial_parameter_tensor_sha256"] != payload[
        "source_initial_parameter_tensor_sha256"
    ]:
        raise RuntimeError(
            "Expected the controlled rerun to preserve the source initialization. "
            f"Provided value: {candidate['initial_parameter_tensor_sha256']!r}."
        )
    summary = {
        "schema_version": WRAPPER_SCHEMA_VERSION,
        "sensitivity_id": spec.sensitivity_id,
        "entry_id": ENTRY_ID,
        "status": "passed" if candidate["admissible"] else "failed",
        "source_input_gain": payload["source_row"]["input_gain"],
        "controlled_input_gain": payload["controlled_row"]["input_gain"],
        "only_changed_field": "input_gain",
        "source_final_validation_accuracy": payload[
            "source_final_validation_accuracy"
        ],
        "final_validation_accuracy": candidate[
            "final_validation_accuracy"
        ],
        "final_validation_loss": candidate["final_validation_loss"],
        "validation_metrics": candidate["validation_metrics"],
        "training_completed": candidate["training_completed"],
        "completed_steps": candidate["completed_steps"],
        "minibatch_order_check": order_check,
        "accuracy_gate_passed": candidate["accuracy_gate_passed"],
        "admissible": candidate["admissible"],
        "inadmissible_reason": candidate["inadmissible_reason"],
        "safety_gate_failure": candidate["safety_gate_failure"],
        "rho_conv_label": payload["rho_conv"],
        "rho_dense_label": payload["rho_dense"],
        "rho_labels_are_source_provenance_only": True,
        "learning_rates_by_parameter": payload[
            "learning_rates_by_parameter"
        ],
        "candidate_summary_sha256": sha256_file(
            candidate_dir / "summary.json"
        ),
        "parent_v7_mutated": False,
        "source_rescue_mutated": False,
        "official_test_read": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        output_dir / "sensitivity_summary.json", summary, canonical=True
    )
    required = _candidate_output_names()
    for name in required:
        source = candidate_dir / name
        if source.is_symlink() or not source.is_file():
            raise RuntimeError(
                "Expected complete controlled candidate output. "
                f"Provided value: {source}."
            )
    output_records = [
        {
            "path": str((candidate_dir / name).relative_to(root)),
            "sha256": sha256_file(candidate_dir / name),
            "bytes": (candidate_dir / name).stat().st_size,
        }
        for name in sorted(required)
    ]
    atomic_write_json(
        output_dir / "candidate_outputs.json",
        {
            "schema_version": (
                "mnist-conv-lr-conv3-legacy-gain-sensitivity-outputs/v1"
            ),
            "outputs": output_records,
        },
        canonical=True,
    )
    _publish_completion(
        root,
        phase="result",
        output_dir=output_dir,
        required_outputs=(
            "candidate_outputs.json",
            "sensitivity_summary.json",
        ),
    )
    return {"status": "complete", "summary": summary}


def plan_publication_retry(
    *,
    sensitivity_dir: str | Path,
) -> dict[str, Any]:
    """Bind the already-finished early-stop outputs to corrected publication code."""

    root, spec, manifest, _ = _load_sensitivity(
        sensitivity_dir, require_current_source=False
    )
    if _valid_completion(root, "result") is not None:
        raise RuntimeError(
            "Expected no completed original result before publication retry. "
            f"Provided value: {root / 'result/complete.json'}."
        )
    candidate_dir = _candidate_dir(root)
    records = []
    for name in _candidate_output_names():
        path = candidate_dir / name
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(
                "Expected the preserved first-attempt output before publication "
                f"retry. Provided value: {path}."
            )
        records.append(
            {
                "path": str(path.relative_to(root)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    candidate = read_json(candidate_dir / "summary.json")
    if (
        candidate.get("training_completed") is not False
        or candidate.get("completed_steps") != 40
        or candidate.get("inadmissible_reason")
        != "loss_ema_explosion:global:onset=33"
    ):
        raise RuntimeError(
            "Expected the preserved step-40 safety stop from the first attempt. "
            f"Provided value: {candidate!r}."
        )
    retry = {
        "schema_version": PUBLICATION_RETRY_SCHEMA_VERSION,
        "phase": PUBLICATION_RETRY_PHASE,
        "sensitivity_id": spec.sensitivity_id,
        "original_manifest_sha256": sha256_file(root / "manifest.json"),
        "code_provenance": normalize_code_provenance(code_provenance()),
        "retry_of": {
            "phase": "result",
            "failure_phase": "post_training_publication",
            "failure_reason": (
                "full_run_minibatch_hash_compared_after_valid_early_safety_stop"
            ),
            "training_rerun": False,
            "preserve_first_attempt_outputs": True,
        },
        "candidate_outputs": sorted(records, key=lambda item: item["path"]),
        "official_test_read": False,
    }
    retry_dir = root / "publication_retry_1"
    retry_dir.mkdir(parents=True, exist_ok=True)
    _write_immutable_json(retry_dir / "manifest.json", retry)
    return {
        "status": "planned",
        "sensitivity_id": spec.sensitivity_id,
        "phase": PUBLICATION_RETRY_PHASE,
        "training_rerun": False,
        "manifest_path": str(retry_dir / "manifest.json"),
    }


def finalize_publication_retry(
    *,
    sensitivity_dir: str | Path,
) -> dict[str, Any]:
    """Publish the preserved safety failure with prefix-aware order validation."""

    root, spec, manifest, _ = _load_sensitivity(
        sensitivity_dir, require_current_source=False
    )
    completed = _valid_completion(root, PUBLICATION_RETRY_PHASE)
    output_dir = root / PUBLICATION_RETRY_PHASE
    if completed is not None:
        return {
            "status": "resumed_complete",
            "summary": read_json(output_dir / "sensitivity_summary.json"),
        }
    retry_path = root / "publication_retry_1/manifest.json"
    retry = read_json(retry_path)
    if (
        retry.get("schema_version") != PUBLICATION_RETRY_SCHEMA_VERSION
        or retry.get("phase") != PUBLICATION_RETRY_PHASE
        or retry.get("sensitivity_id") != spec.sensitivity_id
        or retry.get("original_manifest_sha256")
        != sha256_file(root / "manifest.json")
    ):
        raise RuntimeError(
            "Expected the exact publication retry manifest. "
            f"Provided value: {retry!r}."
        )
    planned = normalize_code_provenance(retry["code_provenance"])
    current = normalize_code_provenance(code_provenance())
    if (
        planned["effective_code_fingerprint"]
        != current["effective_code_fingerprint"]
    ):
        raise RuntimeError(
            "Expected current source fingerprint to match the publication retry. "
            f"Provided value: planned={planned!r}, current={current!r}."
        )
    _verify_records(root, retry["candidate_outputs"])
    candidate_dir = _candidate_dir(root)
    candidate = read_json(candidate_dir / "summary.json")
    source_minibatches = read_json(
        Path(manifest["source_entry_dir"]) / "minibatches.json"
    )
    order_check = _verify_minibatch_prefix(
        candidate_minibatches=read_json(candidate_dir / "minibatches.json"),
        source_minibatches=source_minibatches,
        training_completed=bool(candidate["training_completed"]),
    )
    payload = manifest["payload"]
    if candidate["initial_parameter_tensor_sha256"] != payload[
        "source_initial_parameter_tensor_sha256"
    ]:
        raise RuntimeError(
            "Expected preserved candidate initialization to match the source. "
            f"Provided value: {candidate['initial_parameter_tensor_sha256']!r}."
        )
    summary = {
        "schema_version": WRAPPER_SCHEMA_VERSION,
        "sensitivity_id": spec.sensitivity_id,
        "entry_id": ENTRY_ID,
        "status": "failed",
        "source_input_gain": payload["source_row"]["input_gain"],
        "controlled_input_gain": payload["controlled_row"]["input_gain"],
        "only_changed_field": "input_gain",
        "source_final_validation_accuracy": payload[
            "source_final_validation_accuracy"
        ],
        "final_validation_accuracy": None,
        "final_validation_loss": None,
        "validation_metrics": [],
        "training_completed": False,
        "completed_steps": candidate["completed_steps"],
        "minibatch_order_check": order_check,
        "accuracy_gate_passed": False,
        "admissible": False,
        "inadmissible_reason": candidate["inadmissible_reason"],
        "safety_gate_failure": candidate["safety_gate_failure"],
        "rho_conv_label": payload["rho_conv"],
        "rho_dense_label": payload["rho_dense"],
        "rho_labels_are_source_provenance_only": True,
        "learning_rates_by_parameter": payload[
            "learning_rates_by_parameter"
        ],
        "candidate_summary_sha256": sha256_file(
            candidate_dir / "summary.json"
        ),
        "publication_retry_manifest_sha256": sha256_file(retry_path),
        "training_rerun_for_publication_retry": False,
        "parent_v7_mutated": False,
        "source_rescue_mutated": False,
        "official_test_read": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        output_dir / "sensitivity_summary.json", summary, canonical=True
    )
    output_records = [
        {
            "path": str((candidate_dir / name).relative_to(root)),
            "sha256": sha256_file(candidate_dir / name),
            "bytes": (candidate_dir / name).stat().st_size,
        }
        for name in sorted(_candidate_output_names())
    ]
    atomic_write_json(
        output_dir / "candidate_outputs.json",
        {
            "schema_version": (
                "mnist-conv-lr-conv3-legacy-gain-sensitivity-outputs/v1"
            ),
            "outputs": output_records,
        },
        canonical=True,
    )
    _publish_completion(
        root,
        phase=PUBLICATION_RETRY_PHASE,
        output_dir=output_dir,
        required_outputs=(
            "candidate_outputs.json",
            "sensitivity_summary.json",
        ),
    )
    return {"status": "complete", "summary": summary}


def sensitivity_status(*, sensitivity_dir: str | Path) -> dict[str, Any]:
    root = Path(sensitivity_dir).expanduser().resolve()
    spec = GainSensitivitySpec.from_path(root / "sensitivity.resolved.json")
    result: dict[str, Any] = {
        "sensitivity_id": spec.sensitivity_id,
        "sensitivity_dir": str(root),
        "preflight": (
            "complete"
            if _valid_completion(root, "preflight") is not None
            else "incomplete"
        ),
        "result": (
            "complete"
            if _valid_completion(root, "result") is not None
            else "complete_via_publication_retry"
            if _valid_completion(root, PUBLICATION_RETRY_PHASE) is not None
            else "incomplete"
        ),
    }
    summary = (
        root / "result/sensitivity_summary.json"
        if result["result"] == "complete"
        else root / f"{PUBLICATION_RETRY_PHASE}/sensitivity_summary.json"
    )
    if result["result"] in {"complete", "complete_via_publication_retry"}:
        result["summary"] = read_json(summary)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan")
    plan.add_argument("--config", required=True)
    plan.add_argument("--parent-study", required=True)
    plan.add_argument("--source-entry", required=True)
    plan.add_argument("--results-root", required=True)

    preflight = sub.add_parser("preflight")
    preflight.add_argument("--sensitivity", required=True)
    preflight.add_argument("--data-root", required=True)
    preflight.add_argument("--device", default="cuda")
    preflight.add_argument("--download", action="store_true")

    run = sub.add_parser("run")
    run.add_argument("--sensitivity", required=True)
    run.add_argument("--data-root", required=True)
    run.add_argument("--device", default="cuda")
    run.add_argument("--download", action="store_true")

    replan = sub.add_parser("plan-publication-retry")
    replan.add_argument("--sensitivity", required=True)

    finalize_retry = sub.add_parser("finalize-publication-retry")
    finalize_retry.add_argument("--sensitivity", required=True)

    status = sub.add_parser("status")
    status.add_argument("--sensitivity", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "plan":
        result = plan_sensitivity(
            config=args.config,
            parent_study=args.parent_study,
            source_entry=args.source_entry,
            results_root=args.results_root,
        )
    elif args.command == "preflight":
        result = run_preflight(
            sensitivity_dir=args.sensitivity,
            data_root=args.data_root,
            download=args.download,
            device=args.device,
        )
    elif args.command == "run":
        result = run_sensitivity(
            sensitivity_dir=args.sensitivity,
            data_root=args.data_root,
            download=args.download,
            device=args.device,
        )
    elif args.command == "plan-publication-retry":
        result = plan_publication_retry(
            sensitivity_dir=args.sensitivity,
        )
    elif args.command == "finalize-publication-retry":
        result = finalize_publication_retry(
            sensitivity_dir=args.sensitivity,
        )
    else:
        result = sensitivity_status(sensitivity_dir=args.sensitivity)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GainSensitivitySpec",
    "finalize_publication_retry",
    "plan_sensitivity",
    "plan_publication_retry",
    "run_preflight",
    "run_sensitivity",
    "sensitivity_status",
]
