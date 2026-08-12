from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any


PARENT_SOURCE_COMMIT = "08dd52e44818adf43b011542748e140dc0e01aeb"
STUDY_ID = (
    "perfectdiode-bounded-wmax-sweep-signed-scaled-bias-conv123-"
    "seed0-20260809-v1"
)
EVIDENCE_CLASS = "ordinary_mnist_bounded_wmax_sensitivity"
EXPECTED_TARGET_BY_ARCHITECTURE = {
    "conv1": "akib",
    "conv2": "trex",
    "conv3": "jean-zay",
}
EXPECTED_WEIGHT_MAX_VALUES = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_TRAIN_INDICES_SHA256 = (
    "c0940cfdde9fb2a87f846a4dd234a6ff634eed32a98b9f5d72dfed3e2119a809"
)
EXPECTED_VALIDATION_INDICES_SHA256 = (
    "4801b7805d54dbe2197b98320cfe8a34d5d853c4ab568bcf1d7f69e136c94bb4"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_json(path: Path) -> Any:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"Expected a non-empty JSON file: {path}.")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_parent_config(relative_path: str) -> tuple[dict[str, Any], bytes]:
    source_root = Path(__file__).resolve().parents[1]
    local_path = source_root / relative_path
    if local_path.is_file():
        payload = local_path.read_bytes()
    else:
        completed = subprocess.run(
            ["git", "show", f"{PARENT_SOURCE_COMMIT}:{relative_path}"],
            cwd=source_root,
            check=False,
            capture_output=True,
        )
        if completed.returncode:
            detail = completed.stderr.decode("utf-8", errors="replace").strip()
            raise ValueError(
                f"Could not read frozen parent config {relative_path}: {detail}"
            )
        payload = completed.stdout
    parent = json.loads(payload.decode("utf-8"))
    if not isinstance(parent, dict):
        raise ValueError(f"Expected an object in parent config {relative_path}.")
    return parent, payload


def _require_parent_equivalence(
    config: dict[str, Any],
    parent: dict[str, Any],
    *,
    config_path: Path,
) -> None:
    """Require the declared weight ceiling to be the only scientific delta."""
    restored = deepcopy(config)
    restored.pop("wmax_sweep", None)
    try:
        restored["model_base"]["weight_max"] = parent["model_base"]["weight_max"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Missing model weight bounds in {config_path}.") from error
    for key in ("arm_id", "study_id", "reporting"):
        if key not in parent:
            raise ValueError(f"Parent config is missing {key}: {config_path}.")
        restored[key] = deepcopy(parent[key])
    if restored != parent:
        raise ValueError(
            "Generated config differs from its frozen parent outside the declared "
            f"weight_max and reporting fields: {config_path}."
        )


def _ordered_set_sha256(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        "".join(f"{row['config_sha256']}\n" for row in rows).encode("ascii")
    ).hexdigest()


def _architecture_rows(manifest: dict[str, Any], architecture: str) -> list[dict[str, Any]]:
    rows = [
        row for row in manifest.get("runs", []) if row.get("architecture") == architecture
    ]
    rows.sort(key=lambda row: int(row["architecture_array_index"]))
    if [int(row["architecture_array_index"]) for row in rows] != list(range(30)):
        raise ValueError(
            f"Expected architecture-local indices 0..29 for {architecture}."
        )
    return rows


def validate_config_set(
    config_root: Path,
    *,
    architecture: str,
    expected_config_set_sha256: str,
) -> dict[str, Any]:
    if SHA256_RE.fullmatch(expected_config_set_sha256) is None:
        raise ValueError("Expected a lowercase hexadecimal config-set SHA-256.")
    config_root = Path(config_root).resolve()
    manifest = _load_json(config_root / "study_manifest.json")
    if manifest.get("study_id") != STUDY_ID:
        raise ValueError(f"Unexpected study identity in {config_root}.")
    if int(manifest.get("run_count", -1)) != 90:
        raise ValueError(f"Expected a ninety-arm study manifest in {config_root}.")
    if manifest.get("varied_field") != "model_base.weight_max":
        raise ValueError(f"Unexpected varied field in {config_root}.")
    if float(manifest.get("weight_min")) != 1e-5:
        raise ValueError(f"Unexpected study weight_min in {config_root}.")
    if tuple(float(value) for value in manifest.get("weight_max_values", ())) != (
        EXPECTED_WEIGHT_MAX_VALUES
    ):
        raise ValueError(f"Unexpected weight_max grid in {config_root}.")
    if manifest.get("no_scientific_safety_or_rho_runs") is not True:
        raise ValueError(f"Unexpected scientific gate policy in {config_root}.")
    parent_manifest = manifest.get("parent", {})
    if parent_manifest.get("source_commit") != PARENT_SOURCE_COMMIT:
        raise ValueError(f"Unexpected parent source commit in {config_root}.")
    if manifest.get("target_split", {}).get(architecture) != (
        EXPECTED_TARGET_BY_ARCHITECTURE[architecture]
    ):
        raise ValueError(f"Unexpected target split for {architecture}.")
    rows = _architecture_rows(manifest, architecture)
    expected_surface_order = tuple(
        (weight_max, scheme, optimizer)
        for weight_max in EXPECTED_WEIGHT_MAX_VALUES
        for scheme, optimizer in (
            ("baseline", "sgd"),
            ("baseline", "adam"),
            ("ours", "sgd"),
            ("ours", "adam"),
            ("legacy", "sgd"),
            ("legacy", "adam"),
        )
    )
    observed_surface_order = tuple(
        (float(row["weight_max"]), row.get("scheme"), row.get("optimizer"))
        for row in rows
    )
    if observed_surface_order != expected_surface_order:
        raise ValueError(f"Unexpected ordered surface inventory for {architecture}.")
    actual_set_sha256 = _ordered_set_sha256(rows)
    recorded_set_sha256 = manifest.get(
        "ordered_config_set_sha256_by_architecture", {}
    ).get(architecture)
    if actual_set_sha256 != expected_config_set_sha256:
        raise ValueError(
            "Architecture config-set SHA-256 mismatch: "
            f"expected {expected_config_set_sha256}, got {actual_set_sha256}."
        )
    if actual_set_sha256 != recorded_set_sha256:
        raise ValueError(
            f"Manifest config-set SHA-256 mismatch for {architecture}: "
            f"recorded {recorded_set_sha256}, got {actual_set_sha256}."
        )

    for row in rows:
        if row.get("target") != EXPECTED_TARGET_BY_ARCHITECTURE[architecture]:
            raise ValueError(f"Unexpected target in manifest row: {row!r}.")
        if row.get("scheme") not in {"baseline", "ours", "legacy"}:
            raise ValueError(f"Unexpected amplification scheme: {row!r}.")
        if row.get("optimizer") not in {"sgd", "adam"}:
            raise ValueError(f"Unexpected optimizer: {row!r}.")
        if float(row.get("weight_min")) != 1e-5:
            raise ValueError(f"Unexpected row weight_min: {row!r}.")
        if float(row.get("weight_max")) not in EXPECTED_WEIGHT_MAX_VALUES:
            raise ValueError(f"Unexpected row weight_max: {row!r}.")
        config_path = config_root / row["config"]
        if _sha256(config_path) != row["config_sha256"]:
            raise ValueError(f"Config digest mismatch: {config_path}.")
        config = _load_json(config_path)
        if config.get("study_id") != STUDY_ID:
            raise ValueError(f"Unexpected study identity: {config_path}.")
        if config.get("arm_id") != row.get("arm_id"):
            raise ValueError(f"Unexpected arm identity: {config_path}.")
        reporting = config.get("reporting", {})
        if reporting.get("study_id") != STUDY_ID:
            raise ValueError(f"Unexpected reporting study identity: {config_path}.")
        if reporting.get("arm_id") != row.get("arm_id"):
            raise ValueError(f"Unexpected reporting arm identity: {config_path}.")
        if reporting.get("evidence_class") != EVIDENCE_CLASS:
            raise ValueError(f"Unexpected evidence class: {config_path}.")
        if reporting.get("paper_facing") is not False:
            raise ValueError(f"Selection run is marked paper-facing: {config_path}.")
        model = config.get("model_base", {})
        if float(model.get("weight_min")) != 1e-5:
            raise ValueError(f"Unexpected weight_min: {config_path}.")
        if float(model.get("weight_max")) != float(row["weight_max"]):
            raise ValueError(f"Unexpected weight_max: {config_path}.")
        if model.get("weight_init_mode") != "bounded_uniform":
            raise ValueError(f"Unexpected initializer: {config_path}.")
        if config.get("init_checkpoint_path") not in (None, ""):
            raise ValueError(f"Unexpected initialization checkpoint: {config_path}.")
        if int(config.get("seed", -1)) != 0:
            raise ValueError(f"Unexpected seed: {config_path}.")
        if config.get("lr") != config.get("optimizer", {}).get("learning_rate"):
            raise ValueError(f"Learning-rate fields disagree: {config_path}.")
        order = config.get("parameter_order")
        named = config.get("learning_rates_by_parameter")
        if not isinstance(order, list) or not isinstance(named, dict):
            raise ValueError(f"Missing named learning rates: {config_path}.")
        if config["lr"] != [named[name] for name in order]:
            raise ValueError(f"Named learning-rate order disagrees: {config_path}.")
        if config.get("evaluation", {}).get("official_test", {}).get("policy") != "disabled":
            raise ValueError(f"Official test is not disabled: {config_path}.")
        sweep = config.get("wmax_sweep", {})
        if sweep.get("changed_scientific_fields") != ["model_base.weight_max"]:
            raise ValueError(f"Unexpected varied fields: {config_path}.")
        parent_receipt = sweep.get("parent", {})
        if parent_receipt.get("source_commit") != PARENT_SOURCE_COMMIT:
            raise ValueError(f"Unexpected parent source commit: {config_path}.")
        if parent_receipt.get("config_path") != row.get("parent_config"):
            raise ValueError(f"Unexpected parent config path: {config_path}.")
        if parent_receipt.get("config_sha256") != row.get("parent_config_sha256"):
            raise ValueError(f"Unexpected parent config digest: {config_path}.")
        if sweep.get("initialization_policy", {}).get("matched_random_quantiles") is not True:
            raise ValueError(f"Initialization quantiles are not matched: {config_path}.")
        parent, parent_payload = _load_parent_config(str(row["parent_config"]))
        if _sha256_bytes(parent_payload) != row["parent_config_sha256"]:
            raise ValueError(f"Frozen parent config digest mismatch: {config_path}.")
        _require_parent_equivalence(config, parent, config_path=config_path)

    return {
        "architecture": architecture,
        "config_count": len(rows),
        "config_set_sha256": actual_set_sha256,
        "schema_version": "perfectdiode-bounded-wmax-config-validation/v1",
        "semantic_status": "pass",
        "study_id": manifest["study_id"],
    }


def _require_finite_metrics(metrics: dict[str, Any], path: Path) -> None:
    for key in (
        "best_test_accuracy",
        "best_train_accuracy",
        "final_test_accuracy",
        "final_test_loss",
        "final_train_accuracy",
        "final_train_loss",
    ):
        value = metrics.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"Expected finite {key} in {path}, got {value!r}.")


def _validate_checkpoint(path: Path, *, weight_min: float, weight_max: float) -> None:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a versioned checkpoint object: {path}.")
    if payload.get("format") != "drn.function.parameters":
        raise ValueError(f"Unexpected checkpoint format: {path}.")
    if payload.get("version") != 1:
        raise ValueError(f"Unexpected checkpoint version: {path}.")
    if set(payload) != {"format", "version", "schema", "states"}:
        raise ValueError(f"Unexpected checkpoint keys: {path}.")
    schema = payload.get("schema")
    states = payload.get("states")
    if not isinstance(schema, list) or not isinstance(states, list):
        raise ValueError(f"Unexpected checkpoint schema: {path}.")
    if len(schema) != len(states):
        raise ValueError(f"Checkpoint schema/state length mismatch: {path}.")
    conductance_count = 0
    tolerance = max(1e-10, (weight_max - weight_min) * 2e-6)
    for spec, state in zip(schema, states):
        if not isinstance(spec, dict) or set(spec) != {
            "name",
            "type",
            "shape",
            "dtype",
        }:
            raise ValueError(f"Unexpected checkpoint schema entry in {path}.")
        if not torch.is_tensor(state):
            raise ValueError(f"Checkpoint state is not a tensor in {path}.")
        name = str(spec.get("name", "")).strip()
        if list(state.shape) != spec["shape"] or str(state.dtype) != spec["dtype"]:
            raise ValueError(f"Checkpoint tensor does not match its schema in {path}.")
        if not torch.isfinite(state).all().item():
            raise ValueError(f"Non-finite checkpoint tensor {name} in {path}.")
        if name.startswith(("ConvWeight_", "DenseWeight_")):
            conductance_count += int(state.numel())
            observed_min = float(state.min().item())
            observed_max = float(state.max().item())
            if observed_min < weight_min - tolerance:
                raise ValueError(
                    f"Checkpoint {name} is below weight_min in {path}: {observed_min}."
                )
            if observed_max > weight_max + tolerance:
                raise ValueError(
                    f"Checkpoint {name} is above weight_max in {path}: {observed_max}."
                )
    if conductance_count <= 0:
        raise ValueError(f"No conductance tensors found in {path}.")


def validate_run_summary(
    config_root: Path,
    summary_json: Path,
    *,
    architecture: str,
    index: int,
    expected_config_set_sha256: str,
    source_archive_sha256: str,
    smoke: bool,
) -> dict[str, Any]:
    if SHA256_RE.fullmatch(source_archive_sha256) is None:
        raise ValueError("Expected a lowercase hexadecimal source-archive SHA-256.")
    config_root = Path(config_root).resolve()
    config_receipt = validate_config_set(
        config_root,
        architecture=architecture,
        expected_config_set_sha256=expected_config_set_sha256,
    )
    study_manifest = _load_json(config_root / "study_manifest.json")
    rows = _architecture_rows(study_manifest, architecture)
    if not 0 <= int(index) < len(rows):
        raise ValueError(f"Index {index} is outside [0,{len(rows) - 1}].")
    expected = rows[int(index)]

    summary = _load_json(Path(summary_json))
    if not isinstance(summary, list) or len(summary) != 1:
        raise ValueError(f"Expected one exact-run summary row: {summary_json}.")
    row = summary[0]
    if row.get("status") != "complete":
        raise ValueError(f"Run summary is not complete: {row!r}.")
    if int(row.get("index", -1)) != int(index):
        raise ValueError(f"Unexpected selected index: {row!r}.")
    if row.get("config_sha256") != expected["config_sha256"]:
        raise ValueError(f"Unexpected config digest in run summary: {row!r}.")
    if bool(row.get("smoke")) is not bool(smoke):
        raise ValueError(f"Unexpected smoke mode in run summary: {row!r}.")
    summary_git = row.get("git", {})
    if summary_git.get("commit") != PARENT_SOURCE_COMMIT:
        raise ValueError(f"Unexpected source commit in run summary: {row!r}.")
    if summary_git.get("source_archive_sha256") != source_archive_sha256:
        raise ValueError(f"Unexpected source archive in run summary: {row!r}.")
    expected_epochs = 1 if smoke else int(expected["epochs"])
    if int(row.get("epochs", -1)) != expected_epochs:
        raise ValueError(f"Unexpected executed epoch count: {row!r}.")

    run_dir = Path(row["output_dir"])
    required = {
        name: run_dir / filename
        for name, filename in {
            "manifest": "manifest.json",
            "status": "status.json",
            "metrics_jsonl": "metrics.jsonl",
            "metrics": "metrics.json",
            "result": "result.json",
            "best_model": "best_model.pt",
            "final_model": "final_model.pt",
            "loss_train": "loss_train.npy",
            "loss_validation": "loss_test.npy",
            "accuracy_train": "accuracy_train.npy",
            "accuracy_validation": "accuracy_test.npy",
        }.items()
    }
    for name, path in required.items():
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"Missing or empty {name}: {path}.")

    manifest = _load_json(required["manifest"])
    status = _load_json(required["status"])
    metrics = _load_json(required["metrics"])
    result = _load_json(required["result"])
    if manifest.get("study_id") != study_manifest["study_id"]:
        raise ValueError(f"Unexpected study identity in {required['manifest']}.")
    if manifest.get("arm_id") != expected["arm_id"]:
        raise ValueError(f"Unexpected arm identity in {required['manifest']}.")
    if manifest.get("evidence_class") != EVIDENCE_CLASS:
        raise ValueError(f"Unexpected evidence class in {required['manifest']}.")
    if bool(manifest.get("smoke")) is not bool(smoke):
        raise ValueError(f"Unexpected smoke mode in {required['manifest']}.")
    if manifest.get("configuration", {}).get("sha256") != expected["config_sha256"]:
        raise ValueError(f"Unexpected config binding in {required['manifest']}.")
    expected_config = _load_json(config_root / expected["config"])
    configuration = manifest.get("configuration", {})
    if configuration.get("resolved") != expected_config:
        raise ValueError(f"Resolved config does not match the frozen input in {required['manifest']}.")
    if int(configuration.get("epochs", -1)) != expected_epochs:
        raise ValueError(f"Unexpected executed epoch count in {required['manifest']}.")
    git_state = manifest.get("git", {})
    if git_state.get("commit") != PARENT_SOURCE_COMMIT:
        raise ValueError(f"Unexpected production source commit: {git_state!r}.")
    if git_state.get("source_archive_sha256") != source_archive_sha256:
        raise ValueError(f"Unexpected production source archive: {git_state!r}.")
    if status.get("state") != "complete":
        raise ValueError(f"Run status is not complete: {status!r}.")
    for identity_key, expected_value in (
        ("study_id", STUDY_ID),
        ("arm_id", expected["arm_id"]),
        ("run_id", manifest.get("run_id")),
    ):
        if status.get(identity_key) != expected_value:
            raise ValueError(f"Status {identity_key} mismatch: {status!r}.")
    if result.get("completion", {}).get("criteria_met") is not True:
        raise ValueError(f"Completion criteria were not met: {result!r}.")
    for identity_key, expected_value in (
        ("study_id", STUDY_ID),
        ("arm_id", expected["arm_id"]),
        ("run_id", manifest.get("run_id")),
    ):
        if result.get(identity_key) != expected_value:
            raise ValueError(f"Result {identity_key} mismatch: {result!r}.")
    if bool(result.get("smoke")) is not bool(smoke):
        raise ValueError(f"Unexpected smoke mode in {required['result']}.")
    runtime = manifest.get("runtime", {})
    if runtime.get("target") != EXPECTED_TARGET_BY_ARCHITECTURE[architecture]:
        raise ValueError(f"Unexpected execution target in {required['manifest']}.")
    if runtime.get("device") != "cuda":
        raise ValueError(f"Unexpected execution device in {required['manifest']}.")
    dataset = manifest.get("dataset", {})
    if dataset.get("variant") != "ordinary":
        raise ValueError(f"Unexpected dataset variant: {dataset!r}.")
    if dataset.get("factory") != "labs.datasets.MnistTrainValidationDataset":
        raise ValueError(f"Unexpected dataset factory: {dataset!r}.")
    if dataset.get("official_test_read") is not False:
        raise ValueError(f"Official test was not kept unread: {dataset!r}.")
    if int(metrics.get("official_test_evaluations", 0)) != 0:
        raise ValueError(f"Official test was evaluated: {metrics!r}.")
    if metrics.get("optimizer_steps_applied") is not True:
        raise ValueError(f"Optimizer steps were not applied: {metrics!r}.")
    provenance = metrics.get("dataset_provenance", {})
    if provenance.get("train_indices_sha256") != EXPECTED_TRAIN_INDICES_SHA256:
        raise ValueError(f"Unexpected training cohort: {provenance!r}.")
    if provenance.get("validation_indices_sha256") != EXPECTED_VALIDATION_INDICES_SHA256:
        raise ValueError(f"Unexpected validation cohort: {provenance!r}.")
    _require_finite_metrics(metrics, required["metrics"])

    import numpy as np

    for name in (
        "loss_train",
        "loss_validation",
        "accuracy_train",
        "accuracy_validation",
    ):
        values = np.load(required[name], allow_pickle=False)
        if values.shape != (expected_epochs,):
            raise ValueError(
                f"Unexpected {name} epoch coverage in {required[name]}: {values.shape}."
            )
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite values in {required[name]}.")

    weight_min = float(expected["weight_min"])
    weight_max = float(expected["weight_max"])
    _validate_checkpoint(
        required["best_model"], weight_min=weight_min, weight_max=weight_max
    )
    _validate_checkpoint(
        required["final_model"], weight_min=weight_min, weight_max=weight_max
    )

    return {
        "architecture": architecture,
        "arm_id": expected["arm_id"],
        "config_set_sha256": config_receipt["config_set_sha256"],
        "epochs": expected_epochs,
        "index": int(index),
        "run_dir": str(run_dir),
        "schema_version": "perfectdiode-bounded-wmax-run-validation/v1",
        "semantic_status": "pass",
        "smoke": bool(smoke),
        "source_archive_sha256": source_archive_sha256,
        "study_id": study_manifest["study_id"],
        "weight_max": weight_max,
        "weight_min": weight_min,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--architecture", choices=("conv1", "conv2", "conv3"), required=True)
    parser.add_argument("--expected-config-set-sha256", required=True)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--index", type=int)
    parser.add_argument(
        "--source-archive-sha256",
        default=os.environ.get("EXPERIMENT_SOURCE_ARCHIVE_SHA256"),
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.summary_json is None:
        receipt = validate_config_set(
            args.config_root,
            architecture=args.architecture,
            expected_config_set_sha256=args.expected_config_set_sha256,
        )
    else:
        if args.index is None:
            raise ValueError("Expected --index with --summary-json.")
        if not args.source_archive_sha256:
            raise ValueError(
                "Expected --source-archive-sha256 or "
                "EXPERIMENT_SOURCE_ARCHIVE_SHA256."
            )
        receipt = validate_run_summary(
            args.config_root,
            args.summary_json,
            architecture=args.architecture,
            index=args.index,
            expected_config_set_sha256=args.expected_config_set_sha256,
            source_archive_sha256=args.source_archive_sha256,
            smoke=args.smoke,
        )
    serialized = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
