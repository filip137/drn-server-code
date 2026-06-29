#!/usr/bin/env python3
"""Poll Jean Zay Conv3 jobs and write a completion report to current_state.md."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CURRENT_STATE = REPO_ROOT / "docs" / "current_state.md"
START_MARKER = "<!-- jeanzay-conv3-finish-report:start -->"
END_MARKER = "<!-- jeanzay-conv3-finish-report:end -->"

JOB_IDS = [
    "1004671",
    "1004672",
    "1004864",
    "1004880",
    "1004881",
    "1004882",
    "1004883",
    "1017778",
    "1017779",
    "1017780",
    "1017895",
    "1017896",
    "1017897",
    "1053390",
    "1053391",
    "1053392",
]

ROOTS = {
    "hard_sigmoid legacy low gain": "/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/mnist_bp_conv3_pad1_hardsigmoid_legacy_lowgain_voff4_k8_seed0_10epoch",
    "hard_sigmoid sat30 LR": "/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/mnist_bp_conv3_pad1_hardsigmoid_nonlegacy_sat30_lr_screen_voff4_k8_seed0_10epoch",
    "perfect_diode input gain": "/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/mnist_bp_conv3_pad1_perfect_diode_inputgain_screen_k8_seed0_10epoch",
    "perfect_diode per-amp LR": "/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/mnist_bp_conv3_pad1_perfect_diode_nonlegacy_peramp_lr_screen_k8_seed0_10epoch",
    "perfect_diode per-amp final 30 epoch": "/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/mnist_bp_conv3_pad1_perfect_diode_nonlegacy_peramp_bestlr_k8_seed0_30epoch",
}

EXPECTED_COUNTS = {
    "hard_sigmoid legacy low gain": 5,
    "hard_sigmoid sat30 LR": 15,
    "perfect_diode input gain": 15,
    "perfect_diode per-amp LR": 15,
    "perfect_diode per-amp final 30 epoch": 3,
}


def run_ssh(script: str, *, check: bool = True) -> str:
    proc = run_local(["ssh", "jean-zay", "bash", "-s"], check=False, input_text=script)
    if check and proc.returncode != 0:
        raise RuntimeError(f"ssh command failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    return proc.stdout


def run_local(
    command: list[str],
    *,
    check: bool = True,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        input=input_text,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def queue_has_jobs() -> bool:
    job_expr = ",".join(JOB_IDS)
    script = f"squeue -h -j {job_expr} -o '%T|%R' || true\n"
    output = run_ssh(script, check=False)
    active_lines = []
    for line in output.splitlines():
        state, _, reason = line.partition("|")
        if state == "PENDING" and reason in {"Dependency", "DependencyNeverSatisfied"}:
            continue
        active_lines.append(line)
    return bool(active_lines)


def gather_remote_report() -> dict:
    payload = json.dumps({"roots": ROOTS, "jobs": JOB_IDS})
    script = f"""python - <<'PY'
import json
import math
from pathlib import Path

payload = json.loads({payload!r})
roots = payload["roots"]

def parse_part(parts, prefix):
    for part in parts:
        if part.startswith(prefix):
            return part[len(prefix):].replace("p", ".").replace("m", "-")
    return ""

def read_metric(path):
    data = json.loads(path.read_text())
    config_path = path.parent / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {{}}
    architecture = config.get("architecture", {{}})
    optimizer = config.get("optimizer", {{}})
    training = config.get("training", {{}})
    amplification = config.get("amplification", {{}})
    lr_values = optimizer.get("learning_rate", "")
    if isinstance(lr_values, list):
        lr_value = lr_values[0] if lr_values else ""
    else:
        lr_value = lr_values
    parts = path.parts
    run_name = path.parents[1].name if len(path.parents) > 1 else ""
    nonlinearity = path.parents[2].name if len(path.parents) > 2 else ""
    return {{
        "metrics_path": str(path),
        "run_name": config.get("run_name") or run_name,
        "nonlinearity": architecture.get("non_linearity") or nonlinearity,
        "best_test_accuracy": data.get("best_test_accuracy"),
        "final_test_accuracy": data.get("final_test_accuracy"),
        "best_epoch": data.get("best_epoch"),
        "input_gain": parse_part(parts, "input_gain_") or architecture.get("input_gain", ""),
        "lr_mult": parse_part(parts, "lr_mult_"),
        "lr": parse_part(parts, "lr_") or lr_value,
        "epochs": training.get("epochs", ""),
        "iteration_count": training.get("num_iterations_training", ""),
        "padding": architecture.get("padding", ""),
        "voltage_amp": amplification.get("voltage_amp", ""),
        "current_amp": amplification.get("current_amp", ""),
        "target": parse_part(parts, "target_") or parse_part(parts, "target_sat"),
    }}

summary = {{}}
for label, root_text in roots.items():
    root = Path(root_text)
    metrics = []
    if root.exists():
        for path in root.rglob("metrics.json"):
            try:
                metrics.append(read_metric(path))
            except Exception as exc:
                metrics.append({{"metrics_path": str(path), "error": str(exc)}})
    valid = [m for m in metrics if isinstance(m.get("best_test_accuracy"), (int, float))]
    valid.sort(
        key=lambda m: (
            float(m.get("best_test_accuracy") or float("nan")),
            float(m.get("final_test_accuracy") or -1.0),
        ),
        reverse=True,
    )
    summary[label] = {{
        "root": str(root),
        "exists": root.exists(),
        "metric_count": len(metrics),
        "top_rows": valid[:5],
    }}
print(json.dumps(summary))
PY
sacct -j {','.join(JOB_IDS)} --format=JobID,JobName,State,ExitCode,Elapsed,AllocTRES%60 -P || true
"""
    output = run_ssh(script)
    lines = output.splitlines()
    summary = json.loads(lines[0]) if lines else {}
    sacct = "\n".join(lines[1:]).strip()
    return {"summary": summary, "sacct": sacct}


def pct(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math_is_finite(number):
        return "n/a"
    return f"{number * 100:.2f}%"


def math_is_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def format_top_rows(rows: list[dict]) -> str:
    if not rows:
        return "- No completed metric rows found."
    lines = [
        "| Run | Best / final acc | Best epoch | Gain | LR mult | LR | K | Epochs | Target |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {run} | {best} / {final} | {epoch} | `{gain}` | `{mult}` | `{lr}` | `{k}` | `{epochs}` | `{target}` |".format(
                run=row.get("run_name", ""),
                best=pct(row.get("best_test_accuracy")),
                final=pct(row.get("final_test_accuracy")),
                epoch=row.get("best_epoch", ""),
                gain=row.get("input_gain", ""),
                mult=row.get("lr_mult", ""),
                lr=row.get("lr", ""),
                k=row.get("iteration_count", ""),
                epochs=row.get("epochs", ""),
                target=row.get("target", ""),
            )
        )
    return "\n".join(lines)


def build_report(remote: dict) -> str:
    now = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    summary = remote["summary"]
    lines = [
        START_MARKER,
        "## Jean Zay Conv3 Completion Report",
        "",
        f"Updated: {now}",
        "",
        "Tracked jobs: `" + "`, `".join(JOB_IDS) + "`.",
        "",
        "### Metrics Counts",
        "",
        "| Result set | Metrics | Expected | Root |",
        "|---|---:|---:|---|",
    ]
    for label, root in ROOTS.items():
        item = summary.get(label, {})
        lines.append(
            f"| {label} | {item.get('metric_count', 0)} | {EXPECTED_COUNTS.get(label, '')} | `{root}` |"
        )
    lines.extend(["", "### Best Rows", ""])
    for label in ROOTS:
        item = summary.get(label, {})
        lines.extend([f"#### {label}", "", format_top_rows(item.get("top_rows", [])), ""])
    if remote.get("sacct"):
        lines.extend(["### Slurm Accounting", "", "```text", remote["sacct"], "```", ""])
    lines.append(END_MARKER)
    return "\n".join(lines).rstrip() + "\n"


def update_current_state(report: str) -> None:
    text = CURRENT_STATE.read_text()
    if START_MARKER in text and END_MARKER in text:
        before = text.split(START_MARKER, 1)[0].rstrip()
        after = text.split(END_MARKER, 1)[1].lstrip()
        new_text = before + "\n\n" + report + "\n" + after
    else:
        insertion = "\n## Shared Result Roots\n"
        if insertion in text:
            before, after = text.split(insertion, 1)
            new_text = before.rstrip() + "\n\n" + report + "\n" + insertion + after
        else:
            new_text = text.rstrip() + "\n\n" + report
    CURRENT_STATE.write_text(new_text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=600)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    while True:
        if not queue_has_jobs():
            remote = gather_remote_report()
            update_current_state(build_report(remote))
            print(f"updated {CURRENT_STATE}")
            return
        print(f"{dt.datetime.now().isoformat()} jobs still active")
        if args.once:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
