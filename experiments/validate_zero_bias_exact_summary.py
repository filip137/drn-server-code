from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from experiments.reporting import validate_run


EXPECTED_DIAGNOSTICS = {
    "gradient_trace_samples_per_epoch": 5,
    "checkpoint_every_epoch": True,
    "skip_terminal_official_test": True,
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _verify_biases(path: Path, bias_names: list[str]) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    schema = payload.get("schema", [])
    states = payload.get("states", [])
    if len(schema) != len(states):
        raise RuntimeError(f"Checkpoint schema/state mismatch: {path}")
    seen: list[str] = []
    for spec, state in zip(schema, states, strict=True):
        name = str(spec.get("name", "")).strip()
        if not name.startswith("Bias_"):
            continue
        seen.append(name)
        if int(torch.count_nonzero(state).item()) != 0:
            raise RuntimeError(f"Nonzero frozen bias in {path}: {name}")
    if seen != bias_names:
        raise RuntimeError(
            f"Checkpoint bias inventory mismatch in {path}: "
            f"expected {bias_names}, got {seen}"
        )


def _validate_one(
    *,
    row: dict[str, Any],
    config_path: Path,
    study_manifest: dict[str, Any],
    run_mode: str,
) -> Path:
    config = _load_json(config_path)
    config_sha256 = _sha256_file(config_path)
    study_id = study_manifest["study_id"]
    evidence_class = study_manifest["evidence_class"]
    arm_id = config["arm_id"]
    configured_epochs = int(config["lab"]["epochs"])
    expected_epochs = configured_epochs if run_mode == "production" else 1
    expected_trace_per_epoch = 5 if run_mode == "production" else 1
    expected_trace_steps = expected_epochs * expected_trace_per_epoch
    expected_checkpoint_count = expected_epochs + 1
    parameter_order = config["parameter_order"]
    rates = config["learning_rates_by_parameter"]
    bias_names = [
        name for name in parameter_order if name.startswith("Bias_")
    ]

    if configured_epochs <= 0:
        raise RuntimeError(f"Invalid configured epoch count: {configured_epochs}")
    if not bias_names:
        raise RuntimeError(f"No Bias_* parameters in {config_path}")
    if len(parameter_order) != len(set(parameter_order)):
        raise RuntimeError(f"Duplicate parameter names in {config_path}")
    if set(rates) != set(parameter_order):
        raise RuntimeError(f"Learning-rate inventory mismatch in {config_path}")
    if any(float(rates[name]) != 0.0 for name in bias_names):
        raise RuntimeError(f"Nonzero configured bias rate in {config_path}")
    reporting = config.get("reporting", {})
    if (
        config.get("study_id") != study_id
        or reporting.get("study_id") != study_id
        or reporting.get("arm_id") != arm_id
        or reporting.get("evidence_class") != evidence_class
    ):
        raise RuntimeError(f"Config reporting identity mismatch: {config_path}")
    mnist = config.get("datasets", {}).get("mnist", {})
    if mnist.get("factory") != "labs.datasets.MnistTrainValidationDataset":
        raise RuntimeError(f"Unexpected dataset factory in {config_path}")
    if mnist.get("params", {}).get("affine_config", "missing") is not None:
        raise RuntimeError(f"Affine corruption enabled in {config_path}")
    if (
        config.get("evaluation", {})
        .get("official_test", {})
        .get("policy")
        != "disabled"
    ):
        raise RuntimeError(f"Official test not disabled in {config_path}")

    if row.get("status") != "complete":
        raise RuntimeError(f"Task summary is incomplete: {row!r}")
    if row.get("config_sha256") != config_sha256:
        raise RuntimeError(f"Task-summary config digest mismatch: {row!r}")
    if (
        int(row.get("configured_epochs", -1)) != configured_epochs
        or int(row.get("epochs", -1)) != expected_epochs
    ):
        raise RuntimeError(f"Task-summary epoch mismatch: {row!r}")
    if bool(row.get("smoke")) != (run_mode == "smoke"):
        raise RuntimeError(f"Task-summary run mode mismatch: {row!r}")
    if row.get("diagnostics") != EXPECTED_DIAGNOSTICS:
        raise RuntimeError(f"Task-summary diagnostics mismatch: {row!r}")

    run_dir = Path(row["output_dir"])
    required = {
        "trace": run_dir / "gradient_trace.jsonl",
        "trace_metadata": run_dir / "gradient_trace_metadata.json",
        "checkpoint_index": run_dir / "epoch_checkpoint_index.jsonl",
        "manifest": run_dir / "manifest.json",
        "status": run_dir / "status.json",
        "metrics": run_dir / "metrics.json",
        "metrics_jsonl": run_dir / "metrics.jsonl",
        "result": run_dir / "result.json",
        "best_model": run_dir / "best_model.pt",
        "final_model": run_dir / "final_model.pt",
    }
    for name, path in required.items():
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"Missing or empty {name}: {path}")

    metadata = _load_json(required["trace_metadata"])
    if metadata.get("status") != "complete":
        raise RuntimeError(f"Gradient trace metadata incomplete: {run_dir}")
    if metadata.get("parameter_names") != parameter_order:
        raise RuntimeError(f"Trace parameter inventory mismatch: {run_dir}")
    if int(metadata.get("epoch_count", -1)) != expected_epochs:
        raise RuntimeError(f"Trace epoch count mismatch: {run_dir}")
    if int(metadata.get("samples_per_epoch", -1)) != 5:
        raise RuntimeError(f"Trace requested sample count mismatch: {run_dir}")
    if int(metadata.get("recorded_steps", -1)) != expected_trace_steps:
        raise RuntimeError(f"Trace step count mismatch: {run_dir}")
    if int(metadata.get("recorded_rows", -1)) != (
        expected_trace_steps * len(parameter_order)
    ):
        raise RuntimeError(f"Trace row count mismatch: {run_dir}")

    sampled = metadata.get("sampled_source_batches", [])
    expected_epoch_counts = Counter(
        {
            epoch: expected_trace_per_epoch
            for epoch in range(1, expected_epochs + 1)
        }
    )
    sampled_counts = Counter(int(item["epoch"]) for item in sampled)
    if sampled_counts != expected_epoch_counts:
        raise RuntimeError(f"Trace metadata lacks epoch coverage: {run_dir}")
    metadata_pairs = [
        (int(item["epoch"]), int(item["batch"])) for item in sampled
    ]
    if len(metadata_pairs) != len(set(metadata_pairs)):
        raise RuntimeError(f"Duplicate traced transitions: {run_dir}")

    trace_rows = _jsonl(required["trace"])
    if len(trace_rows) != expected_trace_steps * len(parameter_order):
        raise RuntimeError(f"Unexpected trace row count: {run_dir}")
    transition_parameters: dict[tuple[int, int, int], list[str]] = defaultdict(
        list
    )
    for trace_row in trace_rows:
        key = (
            int(trace_row["epoch"]),
            int(trace_row["batch"]),
            int(trace_row["global_step"]),
        )
        name = trace_row["parameter_name"]
        transition_parameters[key].append(name)
        if float(trace_row["learning_rate"]) != float(rates[name]):
            raise RuntimeError(f"Traced learning rate mismatch: {run_dir}")
    if any(names != parameter_order for names in transition_parameters.values()):
        raise RuntimeError(f"Trace parameter order mismatch: {run_dir}")
    trace_pairs = [
        (epoch, batch) for epoch, batch, _ in transition_parameters
    ]
    if Counter(trace_pairs) != Counter(metadata_pairs):
        raise RuntimeError(f"Trace transitions differ from metadata: {run_dir}")
    if Counter(epoch for epoch, _ in trace_pairs) != expected_epoch_counts:
        raise RuntimeError(f"Trace lacks epoch coverage: {run_dir}")
    if metadata.get("trace_sha256") != _sha256_file(required["trace"]):
        raise RuntimeError(f"Trace SHA-256 mismatch: {run_dir}")
    if int(metadata.get("trace_size_bytes", -1)) != required[
        "trace"
    ].stat().st_size:
        raise RuntimeError(f"Trace size mismatch: {run_dir}")

    checkpoint_rows = _jsonl(required["checkpoint_index"])
    if len(checkpoint_rows) != expected_checkpoint_count:
        raise RuntimeError(f"Checkpoint count mismatch: {run_dir}")
    if [int(item["epoch"]) for item in checkpoint_rows] != list(
        range(expected_checkpoint_count)
    ):
        raise RuntimeError(f"Checkpoint epoch inventory mismatch: {run_dir}")
    for item in checkpoint_rows:
        for prefix in ("model", "optimizer"):
            relative = Path(item[f"{prefix}_path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(f"Unsafe checkpoint path: {relative}")
            path = run_dir / relative
            if not path.is_file() or path.stat().st_size <= 0:
                raise RuntimeError(f"Missing checkpoint: {path}")
            if path.stat().st_size != int(item[f"{prefix}_size_bytes"]):
                raise RuntimeError(f"Checkpoint size mismatch: {path}")
            if _sha256_file(path) != item[f"{prefix}_sha256"]:
                raise RuntimeError(f"Checkpoint digest mismatch: {path}")
        _verify_biases(run_dir / item["model_path"], bias_names)
    _verify_biases(required["best_model"], bias_names)
    _verify_biases(required["final_model"], bias_names)

    manifest = _load_json(required["manifest"])
    status = _load_json(required["status"])
    metrics = _load_json(required["metrics"])
    result = _load_json(required["result"])
    for payload_name, payload in (
        ("manifest", manifest),
        ("status", status),
        ("result", result),
    ):
        if (
            payload.get("study_id") != study_id
            or payload.get("arm_id") != arm_id
        ):
            raise RuntimeError(
                f"{payload_name} identity mismatch: {run_dir}"
            )
    if manifest.get("evidence_class") != evidence_class:
        raise RuntimeError(f"Manifest evidence-class mismatch: {run_dir}")
    configuration = manifest.get("configuration", {})
    if (
        configuration.get("sha256") != config_sha256
        or configuration.get("resolved") != config
    ):
        raise RuntimeError(f"Manifest config mismatch: {run_dir}")
    if (
        int(configuration.get("configured_epochs", -1)) != configured_epochs
        or int(configuration.get("epochs", -1)) != expected_epochs
        or configuration.get("diagnostic_overrides")
        != EXPECTED_DIAGNOSTICS
    ):
        raise RuntimeError(f"Manifest execution contract mismatch: {run_dir}")
    dataset = manifest.get("dataset", {})
    if (
        dataset.get("variant") != "ordinary"
        or dataset.get("factory")
        != "labs.datasets.MnistTrainValidationDataset"
        or dataset.get("evaluation_split") != "validation"
        or dataset.get("official_test_read") is not False
        or dataset.get("params", {}).get("affine_config", "missing") is not None
    ):
        raise RuntimeError(f"Runtime dataset contract mismatch: {run_dir}")
    if status.get("state") != "complete":
        raise RuntimeError(f"Run status is not complete: {run_dir}")
    completion = result.get("completion", {})
    if (
        completion.get("criteria_met") is not True
        or completion.get("diagnostic_overrides") != EXPECTED_DIAGNOSTICS
    ):
        raise RuntimeError(f"Completion criteria mismatch: {run_dir}")
    if (
        int(metrics.get("official_test_evaluations", -1)) != 0
        or int(metrics.get("official_test_examples", -1)) != 0
    ):
        raise RuntimeError(f"Official test was evaluated: {run_dir}")
    for key in (
        "official_test_accuracy",
        "official_test_loss",
        "official_test_checkpoint",
    ):
        if metrics.get(key, "missing") is not None:
            raise RuntimeError(f"Official-test field populated: {run_dir}")
    if (
        metrics.get("checkpoint_every_epoch") is not True
        or int(metrics.get("epoch_checkpoint_count", -1))
        != expected_checkpoint_count
        or int(metrics.get("gradient_trace_samples_per_epoch", -1)) != 5
    ):
        raise RuntimeError(f"Metrics diagnostic contract mismatch: {run_dir}")
    metric_rows = _jsonl(required["metrics_jsonl"])
    if [int(item.get("epoch", -1)) for item in metric_rows] != list(
        range(1, expected_epochs + 1)
    ):
        raise RuntimeError(f"Per-epoch metrics coverage mismatch: {run_dir}")

    bundle_errors = validate_run(run_dir)
    if bundle_errors:
        raise RuntimeError(
            f"Canonical bundle validation failed for {run_dir}: "
            + "; ".join(bundle_errors)
        )
    return run_dir


def validate_summary(
    *,
    summary_path: Path,
    study_manifest_path: Path,
    config_paths: list[Path],
    config_set_sha256: str,
    run_mode: str,
) -> list[Path]:
    if run_mode not in {"production", "smoke"}:
        raise ValueError(f"Unsupported run mode: {run_mode}")
    study_manifest = _load_json(study_manifest_path)
    if study_manifest.get("ordered_config_set_sha256") != config_set_sha256:
        raise RuntimeError("Study-manifest config-set digest mismatch.")
    configs_by_sha = {
        _sha256_file(path): path.resolve() for path in config_paths
    }
    if len(configs_by_sha) != len(config_paths):
        raise RuntimeError("Duplicate selected config digest.")
    rows = _load_json(summary_path)
    if len(rows) != len(config_paths):
        raise RuntimeError(
            f"Expected {len(config_paths)} summary rows, got {len(rows)}."
        )
    row_shas = [row.get("config_sha256") for row in rows]
    if Counter(row_shas) != Counter(configs_by_sha.keys()):
        raise RuntimeError("Summary/config digest inventory mismatch.")
    return [
        _validate_one(
            row=row,
            config_path=configs_by_sha[row["config_sha256"]],
            study_manifest=study_manifest,
            run_mode=run_mode,
        )
        for row in rows
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate exact-run summary rows, diagnostics, checkpoint hashes, "
            "and frozen zero-bias tensors."
        )
    )
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--study-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, action="append", required=True)
    parser.add_argument("--config-set-sha256", required=True)
    parser.add_argument(
        "--run-mode", choices=("production", "smoke"), required=True
    )
    args = parser.parse_args()
    run_dirs = validate_summary(
        summary_path=args.summary_json,
        study_manifest_path=args.study_manifest,
        config_paths=args.config,
        config_set_sha256=args.config_set_sha256,
        run_mode=args.run_mode,
    )
    for run_dir in run_dirs:
        print(run_dir)


if __name__ == "__main__":
    main()
