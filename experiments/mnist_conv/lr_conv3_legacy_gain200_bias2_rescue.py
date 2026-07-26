"""Three-epoch Conv3 legacy gain-200 rerun with only ``Bias_2`` reduced 10x."""

from __future__ import annotations

import argparse
import copy
import csv
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
from .lr_conv3_legacy_gain_sensitivity import _verify_minibatch_prefix
from .lr_engine import (
    build_loader_bundle,
    build_model_runtime,
    parameter_state_diagnostics,
    parameter_tensor_digest,
)
from .lr_stages import (
    _CandidateStepLoop,
    _architecture_checkpoint,
    _bounded_initial_rms_scales,
    execute_candidate_entry,
)
from .lr_study_spec import LRStudySpec


SCHEMA_VERSION = "mnist-conv-lr-conv3-legacy-gain200-bias2-rescue/v1"
ID_SCHEMA_VERSION = "mnist-conv-lr-conv3-legacy-gain200-bias2-rescue-id/v1"
MANIFEST_SCHEMA_VERSION = (
    "mnist-conv-lr-conv3-legacy-gain200-bias2-rescue-manifest/v1"
)
PREFLIGHT_SCHEMA_VERSION = (
    "mnist-conv-lr-conv3-legacy-gain200-bias2-rescue-preflight/v1"
)
RUN_SPEC_SCHEMA_VERSION = (
    "mnist-conv-lr-conv3-legacy-gain200-bias2-rescue-run/v1"
)
RESULT_SCHEMA_VERSION = (
    "mnist-conv-lr-conv3-legacy-gain200-bias2-rescue-result/v1"
)
SUMMARY_SCHEMA_VERSION = (
    "mnist-conv-lr-conv3-legacy-gain200-bias2-rescue-summary/v1"
)
COMPLETION_SCHEMA_VERSION = (
    "mnist-conv-lr-conv3-legacy-gain200-bias2-rescue-completion/v1"
)
PROTOCOL_ID = "conv-hardsigmoid-lr-conv3-legacy-gain200-bias2-rescue-bs16-v1"
ROW_ID = "conv3_legacy_v4_c0p25"
ENTRY_ID = f"{ROW_ID}--legacy-gain200-bias2-0p1x"
ROLE = "legacy-gain200-bias2-0p1x"
STAGE = "bias2_0p1x_rerun"

SOURCE_RATES = {
    "Bias_0": 8.583064829651904e-05,
    "Bias_1": 0.00045001592454456065,
    "Bias_2": 0.000327987781947773,
    "ConvWeight_0": 8.583064829651904e-05,
    "ConvWeight_1": 0.00045001592454456065,
    "ConvWeight_2": 0.000327987781947773,
    "DenseWeight_0": 1.4336360180064786e-05,
}
TARGET_RATES = {
    **SOURCE_RATES,
    "Bias_2": 3.27987781947773e-05,
}

_TOP_LEVEL_KEYS = {
    "schema_version",
    "name",
    "protocol_id",
    "source",
    "controlled_change",
    "training",
    "evidence",
    "artifacts",
}
_EXPECTED_SOURCE = {
    "parent_study_schema_version": "mnist-conv-lr-study/v7",
    "parent_study_id": (
        "lrstudy_6e56b399cf3e522a2c3f39f66a51611147b6171dd3b9e223f20791bc6275caf0"
    ),
    "parent_row_id": ROW_ID,
    "rescue_id": (
        "lrrescue_e9c1d08dc339f723f61d3d73b8059fe5f82fe95952a5d1fa06107ed877141a04"
    ),
    "rescue_stage": "promotion_retry_1",
    "rescue_entry_id": f"{ROW_ID}--legacy-rescue-lower-10x",
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
    "learning_rates_by_parameter": SOURCE_RATES,
}
_EXPECTED_CHANGE = {
    "input_gain": 200.0,
    "parameter": "Bias_2",
    "learning_rate_multiplier": 0.1,
    "source_learning_rate": SOURCE_RATES["Bias_2"],
    "controlled_learning_rate": TARGET_RATES["Bias_2"],
    "learning_rates_by_parameter": TARGET_RATES,
    "all_other_learning_rates_unchanged": True,
    "all_other_runtime_fields_unchanged": True,
    "reprobe_learning_rates": False,
    "rho_labels_are_source_provenance_only": True,
}
_EXPECTED_TRAINING = {
    "model_seed": 0,
    "epochs": 3,
    "steps_per_epoch": 3438,
    "total_steps": 10314,
    "batch_size": 16,
    "validation_batch_size": 64,
    "inference_iterations": 8,
    "training_iterations": 6,
    "minimum_final_validation_accuracy": 0.9,
    "accuracy_boundary": "greater_than_or_equal",
    "schedule": "constant_seven_parameter_vector",
    "optimizer": "sgd_no_momentum_no_weight_decay",
    "one_step_gpu_smoke_required": True,
}
_EXPECTED_EVIDENCE = {
    "debug_schema_version": "conv3-legacy-gain200-debug/v1",
    "debug_summary_sha256": (
        "f4354bcd92ed88c5ef1d3a6acb5c7b6c5c0dd0949ed868be5ac1f2a3c77e38ae"
    ),
    "debug_case_steps_sha256": (
        "1f716f74f52dc9aa833c81c487dcc00351d779c4f9c5bc5ddc1a7f80647de2da"
    ),
    "full_gain200_peak_loss_first_12_steps": 1930.406982421875,
    "bias2_0p1x_peak_loss_first_12_steps": 0.561046838760376,
    "exact_control_replay_required": True,
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
        for key, item in expected.items():
            _assert_exact(item, provided[key], f"{path}.{key}")
        return
    if expected != provided:
        raise _error(f"exactly {expected!r}", provided, path)


def _reject_non_finite(value: Any, path: str = "bias2_rescue") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise _error("finite JSON", value, path)
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_non_finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_non_finite(item, f"{path}[{index}]")


@dataclass(frozen=True)
class Bias2RescueSpec:
    data: dict[str, Any]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Bias2RescueSpec":
        if not isinstance(value, Mapping):
            raise _error("a JSON object", value, "bias2_rescue")
        data = copy.deepcopy(dict(value))
        _reject_non_finite(data)
        if set(data) != _TOP_LEVEL_KEYS:
            raise _error(
                f"an object with exactly keys {sorted(_TOP_LEVEL_KEYS)!r}",
                {
                    "missing": sorted(_TOP_LEVEL_KEYS - set(data)),
                    "extra": sorted(set(data) - _TOP_LEVEL_KEYS),
                },
                "bias2_rescue",
            )
        if data["schema_version"] != SCHEMA_VERSION:
            raise _error(
                f"exactly {SCHEMA_VERSION!r}",
                data["schema_version"],
                "bias2_rescue.schema_version",
            )
        if data["protocol_id"] != PROTOCOL_ID:
            raise _error(
                f"exactly {PROTOCOL_ID!r}",
                data["protocol_id"],
                "bias2_rescue.protocol_id",
            )
        if not isinstance(data["name"], str) or not data["name"].strip():
            raise _error("a non-empty string", data["name"], "bias2_rescue.name")
        data["name"] = data["name"].strip()
        for key, expected in (
            ("source", _EXPECTED_SOURCE),
            ("controlled_change", _EXPECTED_CHANGE),
            ("training", _EXPECTED_TRAINING),
            ("evidence", _EXPECTED_EVIDENCE),
            ("artifacts", _EXPECTED_ARTIFACTS),
        ):
            _assert_exact(expected, data[key], f"bias2_rescue.{key}")
        return cls(data)

    @classmethod
    def from_path(cls, path: str | Path) -> "Bias2RescueSpec":
        source = Path(path).expanduser().resolve()
        try:
            value = json.loads(source.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Expected Bias_2 rescue config to contain strict JSON. "
                f"Provided value: {source}: {exc}."
            ) from exc
        return cls.from_dict(value)

    @property
    def rescue_id(self) -> str:
        payload = copy.deepcopy(self.data)
        payload.pop("name")
        digest = sha256_json(
            {
                "schema_version": ID_SCHEMA_VERSION,
                "bias2_rescue_spec": payload,
            }
        )
        return f"lrbias2_{digest}"


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or read_json(path) != value:
            raise RuntimeError(
                "Expected existing immutable JSON to have identical content. "
                f"Provided value: {path}."
            )
        return
    atomic_write_json(path, value, canonical=True)


def _record(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(
            "Expected Bias_2 rescue input to be a regular file. "
            f"Provided value: {path}."
        )
    return {
        "path": relative,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _verify_records(root: Path, records: Sequence[Mapping[str, Any]]) -> None:
    for record in records:
        observed = _record(root, str(record["path"]))
        expected = {
            "path": str(record["path"]),
            "sha256": str(record["sha256"]),
            "bytes": int(record["bytes"]),
        }
        if observed != expected:
            raise RuntimeError(
                "Expected immutable Bias_2 rescue input to remain hash-identical. "
                f"Provided value: expected={expected!r}, observed={observed!r}."
            )


def _validate_source(
    spec: Bias2RescueSpec,
    parent_root: Path,
    source_root: Path,
    debug_root: Path,
) -> tuple[LRStudySpec, dict[str, Any], dict[str, Any]]:
    parent = LRStudySpec.from_path(parent_root / "study.resolved.json")
    source = spec.data["source"]
    if (
        parent.study_id != source["parent_study_id"]
        or parent.data["schema_version"] != source["parent_study_schema_version"]
    ):
        raise RuntimeError(
            "Expected the exact v7 parent study. Provided value: "
            f"schema={parent.data['schema_version']!r}, id={parent.study_id!r}."
        )
    for name, expected in (
        ("summary.json", source["source_summary_sha256"]),
        ("rescue_summary.json", source["source_rescue_summary_sha256"]),
        ("complete.json", source["source_completion_sha256"]),
    ):
        observed = sha256_file(source_root / name)
        if observed != expected:
            raise RuntimeError(
                f"Expected source {name} SHA-256 to be {expected!r}. "
                f"Provided value: {observed!r}."
            )
    summary = read_json(source_root / "summary.json")
    if (
        summary["row"]["row_id"] != ROW_ID
        or float(summary["row"]["input_gain"]) != source["source_input_gain"]
        or float(summary["rho_conv"]) != source["rho_conv_label"]
        or float(summary["rho_dense"]) != source["rho_dense_label"]
        or summary["learning_rates_by_parameter"] != SOURCE_RATES
        or summary["training_completed"] is not True
        or summary["safety_gate_failure"] is not None
        or float(summary["final_validation_accuracy"])
        != source["source_final_validation_accuracy"]
    ):
        raise RuntimeError(
            "Expected the clean completed 76.38% source candidate. "
            f"Provided value: {summary!r}."
        )
    evidence = spec.data["evidence"]
    if (
        sha256_file(debug_root / "summary.json")
        != evidence["debug_summary_sha256"]
        or sha256_file(debug_root / "case_steps.csv")
        != evidence["debug_case_steps_sha256"]
    ):
        raise RuntimeError(
            "Expected the exact causal Bias_2 ablation evidence. "
            f"Provided value: {debug_root}."
        )
    debug = read_json(debug_root / "summary.json")
    if (
        debug["schema_version"] != evidence["debug_schema_version"]
        or debug["exact_control_replay"]
        != {"gain200": True, "gain665": True}
        or debug["official_test_read"] is not False
    ):
        raise RuntimeError(
            "Expected exact, test-free control replay in the causal evidence. "
            f"Provided value: {debug!r}."
        )
    peak: dict[str, float] = {}
    with (debug_root / "case_steps.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            case = str(row["case"])
            peak[case] = max(peak.get(case, -math.inf), float(row["loss"]))
    expected_peaks = {
        "gain200_full": evidence["full_gain200_peak_loss_first_12_steps"],
        "gain200_bias2_0p1x": evidence[
            "bias2_0p1x_peak_loss_first_12_steps"
        ],
    }
    if {name: peak.get(name) for name in expected_peaks} != expected_peaks:
        raise RuntimeError(
            "Expected causal peak losses to match the frozen evidence. "
            f"Provided value: {peak!r}."
        )
    return parent, summary, debug


def _execution_retry_record(
    retry_of_manifest: str | Path,
    spec: Bias2RescueSpec,
) -> dict[str, Any]:
    source = Path(retry_of_manifest).expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise RuntimeError(
            "Expected retry_of_manifest to be a regular immutable manifest. "
            f"Provided value: {source}."
        )
    previous = read_json(source)
    if (
        previous.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or previous.get("rescue_id") != spec.rescue_id
        or previous.get("entry_id") != ENTRY_ID
    ):
        raise RuntimeError(
            "Expected retry_of_manifest to describe the same Bias_2/10 "
            f"scientific candidate. Provided value: {previous!r}."
        )
    previous_root = source.parent
    if (previous_root / "result/complete.json").exists():
        raise RuntimeError(
            "Expected retry_of_manifest to refer to an incomplete execution. "
            f"Provided value: {source}."
        )
    candidate_dir = (
        previous_root / f"stages/{STAGE}/entries/{ENTRY_ID}"
    )
    candidate_files = (
        []
        if not candidate_dir.exists()
        else sorted(
            str(path.relative_to(previous_root))
            for path in candidate_dir.rglob("*")
            if path.is_file() or path.is_symlink()
        )
    )
    progress = previous_root / "progress.json"
    if candidate_files or progress.exists():
        raise RuntimeError(
            "Expected the superseded execution to have stopped before "
            "training and candidate publication. Provided value: "
            f"candidate_files={candidate_files!r}, progress_exists={progress.exists()!r}."
        )
    preflight = previous_root / "preflight/complete.json"
    if preflight.is_symlink() or not preflight.is_file():
        raise RuntimeError(
            "Expected the superseded execution to retain its passed preflight "
            f"completion. Provided value: {preflight}."
        )
    return {
        "retry_sequence": 1,
        "retry_of_manifest_path": str(source),
        "retry_of_manifest_sha256": sha256_file(source),
        "retry_of_preflight_completion_sha256": sha256_file(preflight),
        "failure_phase": "pre_training_learning_rate_report_validation",
        "completed_training_steps": 0,
        "candidate_outputs_published": False,
        "scientific_candidate_changed": False,
    }


def plan_rescue(
    *,
    config: str | Path,
    parent_study: str | Path,
    source_entry: str | Path,
    debug_dir: str | Path,
    results_root: str | Path,
    provenance: Mapping[str, Any] | None = None,
    retry_of_manifest: str | Path | None = None,
) -> dict[str, Any]:
    spec = Bias2RescueSpec.from_path(config)
    parent_root = Path(parent_study).expanduser().resolve()
    source_root = Path(source_entry).expanduser().resolve()
    debug_root = Path(debug_dir).expanduser().resolve()
    parent, source_summary, _ = _validate_source(
        spec, parent_root, source_root, debug_root
    )
    controlled_row = copy.deepcopy(source_summary["row"])
    controlled_row["input_gain"] = spec.data["controlled_change"]["input_gain"]
    source_batches = read_json(source_root / "minibatches.json")
    payload = {
        "row_id": ROW_ID,
        "candidate_role": ROLE,
        "candidate_stage": "legacy_gain_sensitivity",
        "rho_conv": float(spec.data["source"]["rho_conv_label"]),
        "rho_dense": float(spec.data["source"]["rho_dense_label"]),
        "rho_labels_are_source_provenance_only": True,
        "median_units_by_weight": source_summary["median_units_by_weight"],
        "learning_rates_by_parameter": copy.deepcopy(TARGET_RATES),
        "learning_rates_by_weight": source_summary["learning_rates_by_weight"],
        "peak_learning_rate": max(TARGET_RATES.values()),
        "source_row": source_summary["row"],
        "controlled_row": controlled_row,
        "source_minibatch_order_sha256": source_summary[
            "minibatch_order_sha256"
        ],
        "source_first_minibatch": source_batches["batches"][0],
        "source_initial_parameter_tensor_sha256": source_summary[
            "initial_parameter_tensor_sha256"
        ],
    }
    root = (
        Path(results_root).expanduser().resolve()
        / "bias2_rescues"
        / f"{spec.data['name']}--{spec.rescue_id}"
    )
    root.mkdir(parents=True, exist_ok=True)
    _write_immutable_json(root / "rescue.resolved.json", spec.data)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "rescue_id": spec.rescue_id,
        "entry_id": ENTRY_ID,
        "parent_study_id": parent.study_id,
        "parent_study_dir": str(parent_root),
        "source_entry_dir": str(source_root),
        "debug_dir": str(debug_root),
        "parent_inputs": [
            _record(parent_root, name)
            for name in (
                "study.resolved.json",
                "split/indices.json",
                "split/provenance.json",
                "initialization/conv3.pt",
                "initialization/conv3.json",
            )
        ],
        "source_inputs": [
            _record(source_root, name)
            for name in (
                "summary.json",
                "rescue_summary.json",
                "complete.json",
                "minibatches.json",
            )
        ],
        "debug_inputs": [
            _record(debug_root, name)
            for name in ("summary.json", "case_steps.csv")
        ],
        "code_provenance": normalize_code_provenance(
            provenance or code_provenance()
        ),
        "payload": payload,
        "official_test_read": False,
    }
    if retry_of_manifest is not None:
        manifest["execution_retry"] = _execution_retry_record(
            retry_of_manifest,
            spec,
        )
    _write_immutable_json(root / "manifest.json", manifest)
    return {
        "status": "planned",
        "rescue_id": spec.rescue_id,
        "rescue_dir": str(root),
        "entry_id": ENTRY_ID,
        "learning_rates_by_parameter": TARGET_RATES,
    }


def _load_rescue(
    rescue_dir: str | Path,
) -> tuple[Path, Bias2RescueSpec, dict[str, Any], LRStudySpec]:
    root = Path(rescue_dir).expanduser().resolve()
    spec = Bias2RescueSpec.from_path(root / "rescue.resolved.json")
    if not root.name.endswith(f"--{spec.rescue_id}"):
        raise RuntimeError(
            "Expected Bias_2 rescue directory to end with its content address. "
            f"Provided value: {root.name!r}."
        )
    manifest = read_json(root / "manifest.json")
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or manifest.get("rescue_id") != spec.rescue_id
        or manifest.get("entry_id") != ENTRY_ID
    ):
        raise RuntimeError(
            "Expected the exact immutable Bias_2 rescue manifest. "
            f"Provided value: {manifest!r}."
        )
    planned = normalize_code_provenance(manifest["code_provenance"])
    current = normalize_code_provenance(code_provenance())
    if (
        planned["effective_code_fingerprint"]
        != current["effective_code_fingerprint"]
    ):
        raise RuntimeError(
            "Expected current source fingerprint to match the Bias_2 rescue "
            f"manifest. Provided value: planned={planned!r}, current={current!r}."
        )
    parent_root = Path(manifest["parent_study_dir"]).resolve()
    source_root = Path(manifest["source_entry_dir"]).resolve()
    debug_root = Path(manifest["debug_dir"]).resolve()
    _verify_records(parent_root, manifest["parent_inputs"])
    _verify_records(source_root, manifest["source_inputs"])
    _verify_records(debug_root, manifest["debug_inputs"])
    parent, _, _ = _validate_source(
        spec, parent_root, source_root, debug_root
    )
    return root, spec, manifest, parent


def _completion_path(root: Path, phase: str) -> Path:
    return root / phase / "complete.json"


def _valid_completion(root: Path, phase: str) -> dict[str, Any] | None:
    path = _completion_path(root, phase)
    if path.is_symlink() or not path.is_file():
        return None
    value = read_json(path)
    if (
        value.get("schema_version") != COMPLETION_SCHEMA_VERSION
        or value.get("phase") != phase
    ):
        return None
    for record in value.get("outputs", []):
        output = path.parent / str(record["name"])
        if (
            output.is_symlink()
            or not output.is_file()
            or output.stat().st_size != int(record["bytes"])
            or sha256_file(output) != str(record["sha256"])
        ):
            return None
    return value


def _publish_completion(
    root: Path,
    *,
    phase: str,
    output_dir: Path,
    outputs: Sequence[str],
) -> dict[str, Any]:
    records = []
    for name in sorted(outputs):
        path = output_dir / name
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(
                "Expected Bias_2 rescue output before completion. "
                f"Provided value: {path}."
            )
        records.append(
            {
                "name": name,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    value = {
        "schema_version": COMPLETION_SCHEMA_VERSION,
        "phase": phase,
        "outputs": records,
    }
    _write_immutable_json(_completion_path(root, phase), value)
    return value


def run_preflight(
    *,
    rescue_dir: str | Path,
    data_root: str | Path,
    download: bool,
    device: str,
) -> dict[str, Any]:
    root, spec, manifest, parent = _load_rescue(rescue_dir)
    completed = _valid_completion(root, "preflight")
    if completed is not None:
        return {
            "status": "resumed_complete",
            "preflight": read_json(root / "preflight/preflight.json"),
        }
    payload = manifest["payload"]
    runtime = build_model_runtime(
        parent.data,
        payload["controlled_row"],
        device=device,
        initialization_checkpoint=_architecture_checkpoint(
            Path(manifest["parent_study_dir"]), payload["source_row"]
        ),
        learning_rate=TARGET_RATES,
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
        peak_learning_rate=max(TARGET_RATES.values()),
        learning_rates_by_parameter=TARGET_RATES,
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
        "rescue_id": spec.rescue_id,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "completed_steps": loop.successful_steps,
        "loss": None if result is None else result.loss,
        "initial_parameter_tensor_sha256": initial_digest,
        "first_minibatch": (
            list(loop.minibatches[0]) if loop.minibatches else None
        ),
        "learning_rates_by_parameter": TARGET_RATES,
        "device": device_record,
        "official_test_read": False,
    }
    output = root / "preflight"
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "preflight.json", preflight, canonical=True)
    if not passed:
        raise RuntimeError(
            "Expected the one-step Bias_2/10 GPU preflight to pass. "
            f"Provided value: {preflight!r}."
        )
    _publish_completion(
        root,
        phase="preflight",
        output_dir=output,
        outputs=("preflight.json",),
    )
    return {"status": "complete", "preflight": preflight}


def _candidate_dir(root: Path) -> Path:
    return root / f"stages/{STAGE}/entries/{ENTRY_ID}"


def _candidate_outputs() -> tuple[str, ...]:
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


def run_rescue(
    *,
    rescue_dir: str | Path,
    data_root: str | Path,
    download: bool,
    device: str,
) -> dict[str, Any]:
    root, spec, manifest, parent = _load_rescue(rescue_dir)
    output = root / "result"
    completed = _valid_completion(root, "result")
    if completed is not None:
        return {
            "status": "resumed_complete",
            "summary": read_json(output / "rescue_summary.json"),
        }
    if _valid_completion(root, "preflight") is None:
        raise RuntimeError(
            "Expected a valid one-step GPU preflight before the Bias_2/10 run. "
            f"Provided value: {root / 'preflight'}."
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
        "rescue_id": spec.rescue_id,
        "entry_id": ENTRY_ID,
        "parent_study_id": parent.study_id,
        "source": spec.data["source"],
        "controlled_change": spec.data["controlled_change"],
        "source_row": payload["source_row"],
        "controlled_row": payload["controlled_row"],
        "training": spec.data["training"],
        "learning_rates_by_parameter": TARGET_RATES,
        "checkpoint_sha256": sha256_file(checkpoint),
        "initial_parameter_tensor_sha256": payload[
            "source_initial_parameter_tensor_sha256"
        ],
        "train_indices_sha256": split["train_indices_sha256"],
        "validation_indices_sha256": split["validation_indices_sha256"],
        "preflight_completion_sha256": sha256_file(
            root / "preflight/complete.json"
        ),
        "official_test_read": False,
    }
    candidate = execute_candidate_entry(
        parent.data,
        manifest["parent_study_dir"],
        ROW_ID,
        ROLE,
        candidate_payload=payload,
        output_stage=STAGE,
        output_root=root,
        data_root=data_root,
        download=download,
        device=device,
        controlled_runtime_row=payload["controlled_row"],
        controlled_run_spec=run_spec,
        controlled_summary_schema=RESULT_SCHEMA_VERSION,
        controlled_bias_learning_rate_overrides={
            "Bias_2": TARGET_RATES["Bias_2"]
        },
        progress_path=root / "progress.json",
    )
    candidate_dir = _candidate_dir(root)
    order_check = _verify_minibatch_prefix(
        candidate_minibatches=read_json(candidate_dir / "minibatches.json"),
        source_minibatches=read_json(
            Path(manifest["source_entry_dir"]) / "minibatches.json"
        ),
        training_completed=bool(candidate["training_completed"]),
    )
    if candidate["initial_parameter_tensor_sha256"] != payload[
        "source_initial_parameter_tensor_sha256"
    ]:
        raise RuntimeError(
            "Expected the Bias_2/10 run to preserve the source initialization. "
            f"Provided value: {candidate['initial_parameter_tensor_sha256']!r}."
        )
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "rescue_id": spec.rescue_id,
        "entry_id": ENTRY_ID,
        "status": "passed" if candidate["admissible"] else "failed",
        "input_gain": 200.0,
        "bias2_learning_rate_multiplier": 0.1,
        "learning_rates_by_parameter": TARGET_RATES,
        "training_completed": candidate["training_completed"],
        "completed_steps": candidate["completed_steps"],
        "validation_metrics": candidate["validation_metrics"],
        "final_validation_accuracy": candidate[
            "final_validation_accuracy"
        ],
        "final_validation_loss": candidate["final_validation_loss"],
        "accuracy_gate_passed": candidate["accuracy_gate_passed"],
        "admissible": candidate["admissible"],
        "inadmissible_reason": candidate["inadmissible_reason"],
        "safety_gate_failure": candidate["safety_gate_failure"],
        "minibatch_order_check": order_check,
        "candidate_summary_sha256": sha256_file(
            candidate_dir / "summary.json"
        ),
        "parent_v7_mutated": False,
        "source_rescue_mutated": False,
        "official_test_read": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "rescue_summary.json", summary, canonical=True)
    records = []
    for name in _candidate_outputs():
        path = candidate_dir / name
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(
                "Expected complete Bias_2/10 candidate output. "
                f"Provided value: {path}."
            )
        records.append(
            {
                "path": str(path.relative_to(root)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    atomic_write_json(
        output / "candidate_outputs.json",
        {
            "schema_version": (
                "mnist-conv-lr-conv3-legacy-gain200-bias2-rescue-outputs/v1"
            ),
            "outputs": records,
        },
        canonical=True,
    )
    _publish_completion(
        root,
        phase="result",
        output_dir=output,
        outputs=("candidate_outputs.json", "rescue_summary.json"),
    )
    return {"status": "complete", "summary": summary}


def rescue_status(*, rescue_dir: str | Path) -> dict[str, Any]:
    root = Path(rescue_dir).expanduser().resolve()
    spec = Bias2RescueSpec.from_path(root / "rescue.resolved.json")
    result = {
        "rescue_id": spec.rescue_id,
        "rescue_dir": str(root),
        "preflight": (
            "complete"
            if _valid_completion(root, "preflight") is not None
            else "incomplete"
        ),
        "result": (
            "complete"
            if _valid_completion(root, "result") is not None
            else "incomplete"
        ),
    }
    progress = root / "progress.json"
    if progress.is_file():
        result["progress"] = read_json(progress)
    if result["result"] == "complete":
        result["summary"] = read_json(root / "result/rescue_summary.json")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan")
    plan.add_argument("--config", required=True)
    plan.add_argument("--parent-study", required=True)
    plan.add_argument("--source-entry", required=True)
    plan.add_argument("--debug-dir", required=True)
    plan.add_argument("--results-root", required=True)
    plan.add_argument("--retry-of-manifest")

    preflight = sub.add_parser("preflight")
    preflight.add_argument("--rescue", required=True)
    preflight.add_argument("--data-root", required=True)
    preflight.add_argument("--device", default="cuda")
    preflight.add_argument("--download", action="store_true")

    run = sub.add_parser("run")
    run.add_argument("--rescue", required=True)
    run.add_argument("--data-root", required=True)
    run.add_argument("--device", default="cuda")
    run.add_argument("--download", action="store_true")

    status = sub.add_parser("status")
    status.add_argument("--rescue", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "plan":
        result = plan_rescue(
            config=args.config,
            parent_study=args.parent_study,
            source_entry=args.source_entry,
            debug_dir=args.debug_dir,
            results_root=args.results_root,
            retry_of_manifest=args.retry_of_manifest,
        )
    elif args.command == "preflight":
        result = run_preflight(
            rescue_dir=args.rescue,
            data_root=args.data_root,
            download=args.download,
            device=args.device,
        )
    elif args.command == "run":
        result = run_rescue(
            rescue_dir=args.rescue,
            data_root=args.data_root,
            download=args.download,
            device=args.device,
        )
    else:
        result = rescue_status(rescue_dir=args.rescue)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "Bias2RescueSpec",
    "plan_rescue",
    "rescue_status",
    "run_preflight",
    "run_rescue",
]
