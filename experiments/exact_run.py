#!/usr/bin/env python3
"""Run complete training configs without deriving scientific settings."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from experiments.reporting import (
        complete_run,
        fail_run,
        runtime_context,
        start_run,
    )
except ModuleNotFoundError:  # Support direct execution from the repository root.
    from reporting import complete_run, fail_run, runtime_context, start_run


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_exact_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(
            f"Expected {path} to contain a JSON object. Provided value: {config!r}."
        )

    rates = config.get("lr")
    optimizer = config.get("optimizer")
    optimizer_rates = optimizer.get("learning_rate") if isinstance(optimizer, Mapping) else None
    if not isinstance(rates, list) or not rates:
        raise ValueError(
            "Expected config['lr'] to be a non-empty explicit vector. "
            f"Provided value: {rates!r}."
        )
    if not isinstance(optimizer_rates, list) or optimizer_rates != rates:
        raise ValueError(
            "Expected config['optimizer']['learning_rate'] to equal config['lr']. "
            f"Provided values: optimizer={optimizer_rates!r}, lr={rates!r}."
        )
    invalid = [
        value
        for value in rates
        if isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ]
    if invalid:
        raise ValueError(
            "Expected every learning rate to be finite and non-negative. "
            f"Provided value: {invalid!r}."
        )
    parameter_order = config.get("parameter_order")
    rates_by_parameter = config.get("learning_rates_by_parameter")
    if parameter_order is not None or rates_by_parameter is not None:
        if (
            not isinstance(parameter_order, list)
            or not parameter_order
            or len(set(parameter_order)) != len(parameter_order)
        ):
            raise ValueError(
                "Expected config['parameter_order'] to be a non-empty list of "
                f"unique names. Provided value: {parameter_order!r}."
            )
        if not isinstance(rates_by_parameter, Mapping):
            raise ValueError(
                "Expected config['learning_rates_by_parameter'] to be an object. "
                f"Provided value: {rates_by_parameter!r}."
            )
        if set(parameter_order) != set(rates_by_parameter):
            raise ValueError(
                "Expected parameter_order and learning_rates_by_parameter to name "
                "the same parameters. "
                f"Provided values: order={parameter_order!r}, "
                f"mapping={sorted(rates_by_parameter)!r}."
            )
        expected_rates = [rates_by_parameter[name] for name in parameter_order]
        if expected_rates != rates:
            raise ValueError(
                "Expected config['lr'] to be derived from the named mapping in "
                "config['parameter_order']. "
                f"Provided values: expected={expected_rates!r}, lr={rates!r}."
            )
    optimizer_name = optimizer.get("name") if isinstance(optimizer, Mapping) else None
    if optimizer_name not in {"SGD", "Adam"}:
        raise ValueError(
            "Expected config['optimizer']['name'] to be 'SGD' or 'Adam'. "
            f"Provided value: {optimizer_name!r}."
        )
    if optimizer_name == "Adam":
        for flag in (
            "amsgrad",
            "foreach",
            "fused",
            "maximize",
            "capturable",
            "differentiable",
        ):
            if flag in optimizer and optimizer[flag] is not False:
                raise ValueError(
                    f"Expected config['optimizer']['{flag}'] to be false when supplied. "
                    f"Provided value: {optimizer[flag]!r}."
                )

    epochs = config.get("lab", {}).get("epochs")
    if not isinstance(epochs, int) or epochs <= 0:
        raise ValueError(
            "Expected config['lab']['epochs'] to be a positive integer. "
            f"Provided value: {epochs!r}."
        )
    return config


def selected_configs(
    configs: Sequence[Path],
    index: int | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> list[tuple[int, Path]]:
    items = [
        (position, Path(path).expanduser().resolve())
        for position, path in enumerate(configs)
    ]
    if not items:
        raise ValueError(f"Expected at least one config path. Provided value: {configs!r}.")

    environment = os.environ if environ is None else environ
    raw_index: Any = index
    if raw_index is None:
        raw_index = environment.get("SLURM_ARRAY_TASK_ID")
    if raw_index is None:
        return items
    try:
        selected = int(raw_index)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Expected the run index to be a non-negative integer. "
            f"Provided value: {raw_index!r}."
        ) from error
    if not 0 <= selected < len(items):
        raise ValueError(
            f"Expected the run index to be in [0, {len(items) - 1}]. "
            f"Provided value: {selected!r}."
        )
    return [items[selected]]


def build_train_command(
    config: Path,
    output_dir: Path,
    *,
    device: str | None = None,
    dataset_root: Path | None = None,
    smoke: bool = False,
    reporting_run_dir: Path | None = None,
    epoch_override: int | None = None,
    gradient_trace_samples_per_epoch: int = 0,
    checkpoint_every_epoch: bool = False,
    skip_terminal_official_test: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "labs" / "mnist_train.py"),
        "--config",
        str(config),
        "--output-dir",
        str(output_dir),
    ]
    if reporting_run_dir is not None:
        command.extend(["--reporting-run-dir", str(reporting_run_dir)])
    if device:
        command.extend(["--device", device])
    if dataset_root is not None:
        command.extend(["--dataset-root", str(Path(dataset_root).expanduser().resolve())])
    if epoch_override is not None and not smoke:
        command.extend(["--epochs", str(int(epoch_override))])
    if int(gradient_trace_samples_per_epoch) > 0:
        command.extend(
            [
                "--gradient-trace-samples-per-epoch",
                str(int(gradient_trace_samples_per_epoch)),
            ]
        )
    if checkpoint_every_epoch:
        command.append("--checkpoint-every-epoch")
    if smoke:
        command.extend(
            [
                "--epochs",
                "1",
                "--max-batches",
                "1",
                "--max-test-batches",
                "1",
                "--skip-terminal-official-test",
            ]
        )
    elif skip_terminal_official_test:
        command.append("--skip-terminal-official-test")
    return command


def _git_state() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return {
            "commit": os.environ.get("EXPERIMENT_SOURCE_COMMIT"),
            "dirty": None,
            "source_archive_sha256": os.environ.get(
                "EXPERIMENT_SOURCE_ARCHIVE_SHA256"
            ),
        }
    return {
        "commit": (
            commit.stdout.strip()
            if commit.returncode == 0
            else os.environ.get("EXPERIMENT_SOURCE_COMMIT")
        ),
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        "source_archive_sha256": os.environ.get(
            "EXPERIMENT_SOURCE_ARCHIVE_SHA256"
        ),
    }


def _dataset_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    lab = config.get("lab", {})
    dataset_key = str(lab.get("dataset_key", "mnist"))
    datasets = config.get("datasets", {})
    dataset = datasets.get(dataset_key, {}) if isinstance(datasets, Mapping) else {}
    factory = dataset.get("factory") if isinstance(dataset, Mapping) else None
    params = dataset.get("params", {}) if isinstance(dataset, Mapping) else {}
    train_validation = bool(factory and "TrainValidation" in str(factory))
    affine_config = params.get("affine_config", {}) if isinstance(params, Mapping) else {}
    affine_enabled = bool(
        (factory and "Affine" in str(factory))
        or (
            isinstance(affine_config, Mapping)
            and affine_config.get("enabled")
        )
    )
    evaluation = config.get("evaluation", {})
    official_test = (
        evaluation.get("official_test", {})
        if isinstance(evaluation, Mapping)
        else {}
    )
    official_test_policy = (
        official_test.get("policy", "disabled")
        if isinstance(official_test, Mapping)
        else "disabled"
    )
    return {
        "key": dataset_key,
        "factory": factory,
        "params": params,
        "evaluation_split": "validation" if train_validation else "test",
        "variant": "deterministic_medium_affine" if affine_enabled else "ordinary",
        "official_test_policy": official_test_policy,
        "official_test_read_during_training": False if train_validation else None,
        "official_test_read": (
            "terminal_once" if official_test_policy == "terminal_once" else False
        ),
    }


def _evidence_class(config: Mapping[str, Any], override: str | None) -> str:
    if override:
        return override
    reporting = config.get("reporting", {})
    if isinstance(reporting, Mapping) and reporting.get("evidence_class"):
        return str(reporting["evidence_class"])
    dataset = _dataset_contract(config)
    if dataset["key"] == "mnist" and dataset["variant"] == "deterministic_medium_affine":
        return "paper_medium_affine"
    if dataset["evaluation_split"] == "validation":
        return "ordinary_mnist_selection"
    return "diagnostic"


def _terminal_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    dataset = metrics.get("dataset_provenance", {})
    split = (
        "validation"
        if isinstance(dataset, Mapping)
        and dataset.get("schema") == "mnist-train-validation-split/v1"
        else "test"
    )
    result = {
        "best_epoch": metrics.get("best_epoch"),
        "train": {
            "final_loss": metrics.get("final_train_loss"),
            "final_accuracy": metrics.get("final_train_accuracy"),
            "best_accuracy": metrics.get("best_train_accuracy"),
        },
        split: {
            "final_loss": metrics.get("final_test_loss"),
            "final_accuracy": metrics.get("final_test_accuracy"),
            "best_accuracy": metrics.get("best_test_accuracy"),
        },
    }
    if split == "validation" and metrics.get("official_test_evaluations"):
        result["test"] = {
            "loss": metrics.get("official_test_loss"),
            "accuracy": metrics.get("official_test_accuracy"),
            "checkpoint": metrics.get("official_test_checkpoint"),
            "examples": metrics.get("official_test_examples"),
            "evaluations": metrics.get("official_test_evaluations"),
        }
    return result


def _completion_errors(
    config: Mapping[str, Any],
    metrics: Mapping[str, Any],
    *,
    smoke: bool,
) -> list[str]:
    if smoke:
        return []
    evaluation = config.get("evaluation", {})
    official_test = (
        evaluation.get("official_test", {})
        if isinstance(evaluation, Mapping)
        else {}
    )
    policy = (
        official_test.get("policy", "disabled")
        if isinstance(official_test, Mapping)
        else "disabled"
    )
    if policy != "terminal_once":
        return []

    errors = []
    if metrics.get("official_test_evaluations") != 1:
        errors.append("expected exactly one official-test evaluation")
    if metrics.get("official_test_examples") != 10_000:
        errors.append("expected the official test evaluation to contain 10000 examples")
    if metrics.get("official_test_checkpoint") != "best_validation":
        errors.append("expected official test checkpoint 'best_validation'")
    for key in ("official_test_accuracy", "official_test_loss"):
        value = metrics.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            errors.append(f"expected finite {key}")
    return errors


def run(
    configs: Sequence[Path],
    *,
    output_root: Path,
    device: str | None = None,
    smoke: bool = False,
    index: int | None = None,
    dry_run: bool = False,
    study_id: str | None = None,
    evidence_class: str | None = None,
    target: str | None = None,
    dataset_root: Path | None = None,
    epoch_override: int | None = None,
    gradient_trace_samples_per_epoch: int = 0,
    checkpoint_every_epoch: bool = False,
    skip_terminal_official_test: bool = False,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    chosen = selected_configs(configs, index, environ=environ)
    output_root = Path(output_root).expanduser().resolve()
    results = []
    git_state = _git_state()
    if epoch_override is not None and int(epoch_override) <= 0:
        raise ValueError(
            f"Expected epoch_override to be positive. Provided value: {epoch_override!r}."
        )
    if int(gradient_trace_samples_per_epoch) < 0:
        raise ValueError(
            "Expected gradient_trace_samples_per_epoch to be non-negative. "
            f"Provided value: {gradient_trace_samples_per_epoch!r}."
        )

    for position, config_path in chosen:
        config = load_exact_config(config_path)
        configured_epochs = int(config["lab"]["epochs"])
        if epoch_override is not None and int(epoch_override) > configured_epochs:
            raise ValueError(
                "Expected a limited diagnostic epoch override no larger than the "
                f"configured budget. Provided value for {config_path}: "
                f"override={epoch_override}, configured={configured_epochs}."
            )
        resolved_epochs = (
            1
            if smoke
            else int(epoch_override)
            if epoch_override is not None
            else configured_epochs
        )
        config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
        case_dir = output_root / ("smoke" if smoke else "") / (
            f"{position:03d}_{config_path.stem}_{config_sha256[:8]}"
        )
        command = build_train_command(
            config_path,
            case_dir,
            device=device,
            dataset_root=dataset_root,
            smoke=smoke,
            reporting_run_dir=case_dir,
            epoch_override=epoch_override,
            gradient_trace_samples_per_epoch=gradient_trace_samples_per_epoch,
            checkpoint_every_epoch=checkpoint_every_epoch,
            skip_terminal_official_test=skip_terminal_official_test,
        )
        reporting = config.get("reporting", {})
        arm_id = (
            str(reporting["arm_id"])
            if isinstance(reporting, Mapping) and reporting.get("arm_id")
            else config_path.stem
        )
        resolved_study_id = study_id or output_root.name
        runtime = runtime_context(
            target=target,
            environ=os.environ if environ is None else environ,
        )
        runtime["device"] = device
        record = {
            "index": position,
            "config": str(config_path),
            "config_sha256": config_sha256,
            "learning_rates": config["lr"],
            "optimizer": config["optimizer"]["name"],
            "configured_epochs": configured_epochs,
            "epochs": resolved_epochs,
            "smoke": smoke,
            "diagnostics": {
                "gradient_trace_samples_per_epoch": int(
                    gradient_trace_samples_per_epoch
                ),
                "checkpoint_every_epoch": bool(checkpoint_every_epoch),
                "skip_terminal_official_test": bool(
                    smoke or skip_terminal_official_test
                ),
            },
            "dataset_root_override": (
                str(Path(dataset_root).expanduser().resolve())
                if dataset_root is not None
                else None
            ),
            "command": command,
            "output_dir": str(case_dir),
            "git": git_state,
            "host": socket.gethostname(),
            "slurm_job_id": (os.environ if environ is None else environ).get(
                "SLURM_JOB_ID"
            ),
            "slurm_array_task_id": (
                os.environ if environ is None else environ
            ).get("SLURM_ARRAY_TASK_ID"),
            "status": "planned" if dry_run else "running",
        }
        if dry_run:
            results.append(record)
            continue
        if case_dir.exists() and any(case_dir.iterdir()):
            raise FileExistsError(
                f"Expected a new or empty output directory. Provided value: {case_dir}."
            )

        dataset_contract = _dataset_contract(config)
        if smoke or skip_terminal_official_test:
            dataset_contract["configured_official_test_policy"] = dataset_contract[
                "official_test_policy"
            ]
            dataset_contract["official_test_policy"] = "skipped_for_diagnostic"
            dataset_contract["official_test_read"] = False
        manifest = {
            "study_id": resolved_study_id,
            "run_id": case_dir.name,
            "arm_id": arm_id,
            "evidence_class": _evidence_class(config, evidence_class),
            "smoke": smoke,
            "configuration": {
                "path": str(config_path),
                "sha256": config_sha256,
                "resolved": config,
                "learning_rates": config["lr"],
                "optimizer": config["optimizer"]["name"],
                "configured_epochs": configured_epochs,
                "epochs": record["epochs"],
                "diagnostic_overrides": record["diagnostics"],
            },
            "dataset": dataset_contract,
            "command": command,
            "git": git_state,
            "runtime": runtime,
            "transport_overrides": {
                "dataset_root": record["dataset_root_override"],
            },
            "inputs": [
                {
                    "role": "scientific_config",
                    "path": str(config_path),
                    "sha256": config_sha256,
                }
            ],
        }
        start_run(case_dir, manifest)
        record_path = case_dir / "exact_run.json"
        _write_json(record_path, record)
        try:
            completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
        except BaseException as error:
            record["status"] = "failed"
            record["error"] = str(error)
            _write_json(record_path, record)
            fail_run(case_dir, error=error)
            raise
        record["returncode"] = completed.returncode
        record["status"] = "complete" if completed.returncode == 0 else "failed"
        _write_json(record_path, record)
        if completed.returncode:
            fail_run(
                case_dir,
                error=f"Training command exited with status {completed.returncode}.",
                returncode=completed.returncode,
            )
        else:
            metrics_path = case_dir / "metrics.json"
            metrics = (
                json.loads(metrics_path.read_text(encoding="utf-8"))
                if metrics_path.exists()
                else {}
            )
            completion_errors = _completion_errors(
                config,
                metrics,
                smoke=bool(smoke or skip_terminal_official_test),
            )
            if completion_errors:
                error = "; ".join(completion_errors)
                record["status"] = "failed"
                record["error"] = error
                _write_json(record_path, record)
                fail_run(case_dir, error=error, returncode=completed.returncode)
                results.append(record)
                break
            complete_run(
                case_dir,
                terminal_metrics=_terminal_metrics(metrics),
                completion={
                    "criteria_met": True,
                    "returncode": completed.returncode,
                    "smoke": smoke,
                    "diagnostic_overrides": record["diagnostics"],
                },
            )
        results.append(record)
        if completed.returncode:
            break

    return {
        "status": (
            "planned"
            if dry_run
            else "complete"
            if results and all(item["status"] == "complete" for item in results)
            else "failed"
        ),
        "runs": results,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", type=Path, nargs="+")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="Transport-only dataset root override, primarily for local canaries.",
    )
    parser.add_argument("--index", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--epoch-override",
        type=int,
        help=(
            "Diagnostic trajectory prefix. Must be positive and no larger than "
            "the epoch budget frozen in every source config."
        ),
    )
    parser.add_argument(
        "--gradient-trace-samples-per-epoch",
        type=int,
        default=0,
        help="Sample this many real optimizer transitions per training epoch.",
    )
    parser.add_argument(
        "--checkpoint-every-epoch",
        action="store_true",
        help="Save model and optimizer state at initialization and after every epoch.",
    )
    parser.add_argument(
        "--skip-terminal-official-test",
        action="store_true",
        help="Skip the terminal official test for a diagnostic-only run.",
    )
    parser.add_argument("--study-id")
    parser.add_argument("--evidence-class")
    parser.add_argument("--target")
    parser.add_argument(
        "--summary-json",
        type=Path,
        help="Optional path for the selected invocation's one-row run summary.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(
        args.configs,
        output_root=args.output_root,
        device=args.device,
        smoke=args.smoke,
        index=args.index,
        dry_run=args.dry_run,
        study_id=args.study_id,
        evidence_class=args.evidence_class,
        target=args.target,
        dataset_root=args.dataset_root,
        epoch_override=args.epoch_override,
        gradient_trace_samples_per_epoch=args.gradient_trace_samples_per_epoch,
        checkpoint_every_epoch=args.checkpoint_every_epoch,
        skip_terminal_official_test=args.skip_terminal_official_test,
    )
    if args.summary_json is not None:
        _write_json(args.summary_json.expanduser().resolve(), result["runs"])
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["status"] in {"complete", "planned"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
