"""Submit the Conv2 optimizer benchmark or three main Jean Zay packs."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.mnist_conv.identity import code_provenance, sha256_file
from experiments.mnist_conv.lr_optimizer_boundary import (
    load_execution_plan,
    load_main_manifest,
    publish_once,
)
from experiments.mnist_conv.lr_optimizer_boundary_jeanzay import (
    benchmark_exports,
    pack_exports,
    sbatch_command,
)
from experiments.mnist_conv.lr_v6_jeanzay import verify_login_r3_allocation
from experiments.run_mnist_conv_lr_optimizer_boundary import (
    load_benchmark_report,
)


SCHEMES = ("baseline", "ours", "legacy")


def _error(expected: str, provided: Any) -> ValueError:
    return ValueError(f"Expected {expected}. Provided value: {provided!r}.")


def _common_paths(args: argparse.Namespace) -> dict[str, Path]:
    values = {
        name: Path(getattr(args, name.replace("-", "_"))).expanduser().resolve()
        for name in (
            "repo-root",
            "study-config",
            "study-dir",
            "manifest",
            "reuse-root",
            "data-root",
        )
    }
    for name in ("repo-root", "study-dir", "reuse-root", "data-root"):
        if not values[name].is_dir():
            raise _error(f"{name} to be an existing directory", values[name])
    for name in ("study-config", "manifest"):
        if not values[name].is_file():
            raise _error(f"{name} to be an existing file", values[name])
    return values


def prepare_benchmark_submission(
    args: argparse.Namespace,
) -> tuple[list[list[str]], dict[str, Any]]:
    paths = _common_paths(args)
    wrapper = (
        paths["repo-root"]
        / "experiments"
        / "run_mnist_conv_lr_optimizer_boundary_benchmark_jeanzay.slurm"
    )
    if not wrapper.is_file():
        raise _error("the benchmark Slurm wrapper to exist", wrapper)
    output = Path(args.output).expanduser().resolve()
    log_dir = Path(args.log_dir).expanduser().resolve()
    exports = benchmark_exports(
        repo_root=paths["repo-root"],
        python=args.python,
        study_config=paths["study-config"],
        study_dir=paths["study-dir"],
        manifest=paths["manifest"],
        reuse_root=paths["reuse-root"],
        data_root=paths["data-root"],
        output=output,
        module=args.module,
    )
    command = sbatch_command(
        wrapper=wrapper,
        exports=exports,
        job_name="conv2-opt-boundary-bench",
    )
    command.insert(-1, f"--output={log_dir}/%x-%j.out")
    command.insert(-1, f"--error={log_dir}/%x-%j.err")
    return [command], {
        "operation": "benchmark",
        "benchmark_output": str(output),
        "planned_concurrency_levels": [1, 2, 4, 8, 12, 16],
        "log_dir": str(log_dir),
        "study_id": load_main_manifest(
            paths["manifest"], reuse_root=paths["reuse-root"]
        )["study_id"],
        "study_config_sha256": sha256_file(paths["study-config"]),
        "manifest_sha256": sha256_file(paths["manifest"]),
    }


def prepare_main_pack_submissions(
    args: argparse.Namespace,
) -> tuple[list[list[str]], dict[str, Any]]:
    paths = _common_paths(args)
    manifest = load_main_manifest(
        paths["manifest"], reuse_root=paths["reuse-root"]
    )
    benchmark_path = Path(args.benchmark).expanduser().resolve()
    report, binding = load_benchmark_report(
        benchmark_path=benchmark_path,
        study_config=paths["study-config"],
        execution_plan=manifest,
        execution_plan_path=paths["manifest"],
    )
    wrapper = (
        paths["repo-root"]
        / "experiments"
        / "run_mnist_conv_lr_optimizer_boundary_jeanzay.slurm"
    )
    if not wrapper.is_file():
        raise _error("the production-pack Slurm wrapper to exist", wrapper)
    log_root = Path(args.log_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    commands: list[list[str]] = []
    pack_outputs: dict[str, str] = {}
    for scheme in SCHEMES:
        log_dir = log_root / scheme
        output = output_root / f"main-grid-{scheme}.json"
        exports = pack_exports(
            repo_root=paths["repo-root"],
            python=args.python,
            study_config=paths["study-config"],
            study_dir=paths["study-dir"],
            manifest=paths["manifest"],
            benchmark=benchmark_path,
            reuse_root=paths["reuse-root"],
            data_root=paths["data-root"],
            scheme=scheme,
            stage="main_grid",
            concurrency=binding["selected_concurrency"],
            log_dir=log_dir,
            output=output,
            module=args.module,
        )
        command = sbatch_command(
            wrapper=wrapper,
            exports=exports,
            job_name=f"conv2-opt-{scheme}",
        )
        command.insert(-1, f"--output={log_dir}/%x-%j.out")
        command.insert(-1, f"--error={log_dir}/%x-%j.err")
        commands.append(command)
        pack_outputs[scheme] = str(output)
    if len(commands) != 3:
        raise RuntimeError("Expected exactly three independent production packs.")
    return commands, {
        "operation": "main_packs",
        "study_id": manifest["study_id"],
        "study_config_sha256": sha256_file(paths["study-config"]),
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": sha256_file(paths["manifest"]),
        "benchmark_id": report["benchmark_id"],
        "benchmark_sha256": binding["benchmark_sha256"],
        "selected_concurrency": binding["selected_concurrency"],
        "schemes": list(SCHEMES),
        "pack_outputs": pack_outputs,
        "log_root": str(log_root),
    }


def prepare_conditional_pack_submissions(
    args: argparse.Namespace,
) -> tuple[list[list[str]], dict[str, Any]]:
    paths = _common_paths(args)
    plan = load_execution_plan(
        paths["manifest"], reuse_root=paths["reuse-root"]
    )
    if plan.get("stage") != args.stage:
        raise _error(
            "conditional plan stage to match --stage",
            {"plan": plan.get("stage"), "argument": args.stage},
        )
    benchmark_path = Path(args.benchmark).expanduser().resolve()
    report, binding = load_benchmark_report(
        benchmark_path=benchmark_path,
        study_config=paths["study-config"],
        execution_plan=plan,
        execution_plan_path=paths["manifest"],
    )
    wrapper = (
        paths["repo-root"]
        / "experiments"
        / "run_mnist_conv_lr_optimizer_boundary_jeanzay.slurm"
    )
    if not wrapper.is_file():
        raise _error("the production-pack Slurm wrapper to exist", wrapper)
    active_schemes = [
        scheme
        for scheme in SCHEMES
        if any(
            entry.get("scheme") == scheme
            and entry.get("mode") == "new_five_epoch_training"
            for entry in plan["entries"]
        )
    ]
    log_root = Path(args.log_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    commands: list[list[str]] = []
    pack_outputs: dict[str, str] = {}
    for scheme in active_schemes:
        log_dir = log_root / scheme
        output = output_root / f"{args.stage}-{scheme}.json"
        exports = pack_exports(
            repo_root=paths["repo-root"],
            python=args.python,
            study_config=paths["study-config"],
            study_dir=paths["study-dir"],
            manifest=paths["manifest"],
            benchmark=benchmark_path,
            reuse_root=paths["reuse-root"],
            data_root=paths["data-root"],
            scheme=scheme,
            stage=args.stage,
            concurrency=binding["selected_concurrency"],
            log_dir=log_dir,
            output=output,
            module=args.module,
        )
        command = sbatch_command(
            wrapper=wrapper,
            exports=exports,
            job_name=f"conv2-opt-{args.stage[:8]}-{scheme}",
        )
        command.insert(-1, f"--output={log_dir}/%x-%j.out")
        command.insert(-1, f"--error={log_dir}/%x-%j.err")
        commands.append(command)
        pack_outputs[scheme] = str(output)
    return commands, {
        "operation": "conditional_packs",
        "stage": args.stage,
        "study_id": plan["study_id"],
        "study_config_sha256": sha256_file(paths["study-config"]),
        "plan_id": plan["plan_id"],
        "plan_sha256": sha256_file(paths["manifest"]),
        "main_manifest_id": plan["main_manifest_id"],
        "benchmark_id": report["benchmark_id"],
        "benchmark_sha256": binding["benchmark_sha256"],
        "selected_concurrency": binding["selected_concurrency"],
        "schemes": active_schemes,
        "entry_count": len(plan["entries"]),
        "pack_outputs": pack_outputs,
        "log_root": str(log_root),
    }


def _submitted_job_id(
    completed: subprocess.CompletedProcess[str],
) -> str:
    value = completed.stdout.strip().split(";", maxsplit=1)[0]
    if not value.isdigit():
        raise RuntimeError(
            "Expected sbatch --parsable to return a numeric job id. "
            f"Provided value: {completed.stdout!r}."
        )
    return value


def submit_prepared(
    *,
    commands: Sequence[Sequence[str]],
    metadata: Mapping[str, Any],
    record_path: str | Path,
    submit: bool,
    runner: Any = subprocess.run,
) -> dict[str, Any]:
    target = Path(record_path).expanduser().resolve()
    attempts_dir = target.with_name(target.name + ".attempts")
    if target.exists() or attempts_dir.exists():
        raise RuntimeError(
            "Expected submission record and attempt directory to be absent "
            "before any sbatch call; refusing a potentially duplicate "
            f"submission. Provided value: record={target}, "
            f"attempts={attempts_dir}."
        )
    base = {
        "schema_version": (
            "mnist-conv-lr-optimizer-boundary-jeanzay-submission/v1"
        ),
        "status": "dry_run" if not submit else "submitted",
        "submitted": submit,
        "commands": [list(command) for command in commands],
        "jobs": [],
        "code_provenance": code_provenance(),
        **dict(metadata),
        "official_test_read": False,
    }
    if not submit:
        return base
    if not commands:
        record = {**base, "status": "no_work"}
        publish_once(target, record)
        return record
    audit = verify_login_r3_allocation(runner=runner)
    if metadata["operation"] == "benchmark":
        Path(str(metadata["log_dir"])).mkdir(parents=True, exist_ok=True)
        Path(str(metadata["benchmark_output"])).parent.mkdir(
            parents=True, exist_ok=True
        )
    else:
        log_root = Path(str(metadata["log_root"]))
        for scheme in SCHEMES:
            (log_root / scheme).mkdir(parents=True, exist_ok=True)
        for output in metadata["pack_outputs"].values():
            Path(str(output)).parent.mkdir(parents=True, exist_ok=True)
    job_records = []
    attempts_dir.mkdir(parents=True, exist_ok=False)
    for index, command in enumerate(commands):
        completed = runner(
            list(command), check=True, text=True, capture_output=True
        )
        job = {
            "job_id": _submitted_job_id(completed),
            "command": list(command),
        }
        publish_once(
            attempts_dir / f"job-{index:03d}.json",
            {
                "schema_version": (
                    "mnist-conv-lr-optimizer-boundary-submission-attempt/v1"
                ),
                "operation": metadata["operation"],
                "job_index": index,
                **job,
            },
        )
        job_records.append(job)
    record = {**base, "jobs": job_records, "login_audit": audit}
    publish_once(target, record)
    return record


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    for name in ("benchmark", "main-packs", "conditional-packs"):
        command = subparsers.add_parser(name)
        command.add_argument("--repo-root", required=True)
        command.add_argument("--python", required=True)
        command.add_argument("--study-config", required=True)
        command.add_argument("--study-dir", required=True)
        command.add_argument("--manifest", required=True)
        command.add_argument("--reuse-root", required=True)
        command.add_argument("--data-root", required=True)
        command.add_argument(
            "--module", default="pytorch-gpu/py3/2.5.0"
        )
        command.add_argument("--record", required=True)
        command.add_argument("--submit", action="store_true")
    benchmark = subparsers.choices["benchmark"]
    benchmark.add_argument("--output", required=True)
    benchmark.add_argument("--log-dir", required=True)
    packs = subparsers.choices["main-packs"]
    packs.add_argument("--benchmark", required=True)
    packs.add_argument("--output-root", required=True)
    packs.add_argument("--log-root", required=True)
    conditional = subparsers.choices["conditional-packs"]
    conditional.add_argument("--benchmark", required=True)
    conditional.add_argument(
        "--stage",
        required=True,
        choices=("upper_sentinel", "bias_capped_confirmation"),
    )
    conditional.add_argument("--output-root", required=True)
    conditional.add_argument("--log-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.operation == "benchmark":
            commands, metadata = prepare_benchmark_submission(args)
        elif args.operation == "main-packs":
            commands, metadata = prepare_main_pack_submissions(args)
        else:
            commands, metadata = prepare_conditional_pack_submissions(args)
        result = submit_prepared(
            commands=commands,
            metadata=metadata,
            record_path=args.record,
            submit=args.submit,
        )
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(str(exc), flush=True)
        return 2
    print(json.dumps(result, indent=None if args.submit else 2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
