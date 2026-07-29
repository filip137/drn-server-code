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
    smoke: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "labs" / "mnist_train.py"),
        "--config",
        str(config),
        "--output-dir",
        str(output_dir),
    ]
    if device:
        command.extend(["--device", device])
    if smoke:
        command.extend(
            [
                "--epochs",
                "1",
                "--max-batches",
                "1",
                "--max-test-batches",
                "1",
            ]
        )
    return command


def _git_state() -> dict[str, Any]:
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
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def run(
    configs: Sequence[Path],
    *,
    output_root: Path,
    device: str | None = None,
    smoke: bool = False,
    index: int | None = None,
    dry_run: bool = False,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    chosen = selected_configs(configs, index, environ=environ)
    output_root = Path(output_root).expanduser().resolve()
    results = []
    git_state = _git_state()

    for position, config_path in chosen:
        config = load_exact_config(config_path)
        config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
        case_dir = output_root / ("smoke" if smoke else "") / (
            f"{position:03d}_{config_path.stem}_{config_sha256[:8]}"
        )
        command = build_train_command(
            config_path,
            case_dir,
            device=device,
            smoke=smoke,
        )
        record = {
            "index": position,
            "config": str(config_path),
            "config_sha256": config_sha256,
            "learning_rates": config["lr"],
            "optimizer": config["optimizer"]["name"],
            "epochs": 1 if smoke else config["lab"]["epochs"],
            "smoke": smoke,
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

        record_path = case_dir / "exact_run.json"
        _write_json(record_path, record)
        completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
        record["returncode"] = completed.returncode
        record["status"] = "complete" if completed.returncode == 0 else "failed"
        _write_json(record_path, record)
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
    parser.add_argument("--index", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
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
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["status"] in {"complete", "planned"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
