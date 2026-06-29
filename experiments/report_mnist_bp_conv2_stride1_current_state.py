#!/usr/bin/env python3
"""Report Conv2 stride-1 best-run repeats into docs/current_state.md."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CURRENT_STATE = REPO_ROOT / "docs" / "current_state.md"
START = "<!-- conv2-stride1-best30-autoupdate:start -->"
END = "<!-- conv2-stride1-best30-autoupdate:end -->"
ROOT = (
    REPO_ROOT
    / "results"
    / "mnist_bp_conv2_hardsigmoid_stride1_best_sat30_voff4_k6_seed0_30epoch"
)

RUNS = [
    {
        "label": "A=1,B=1",
        "run_name": "mnist_bp_amp_v1_c1",
        "gain": "221.068",
        "lr_mult": "4",
        "lr": "0.0260553",
        "path": ROOT
        / "target_sat30"
        / "input_gain_221p068374634"
        / "lr_mult_4"
        / "lr_0p0260552872365"
        / "hard_sigmoid"
        / "mnist_bp_amp_v1_c1"
        / "seed_0"
        / "metrics.json",
    },
    {
        "label": "A=4,B=1",
        "run_name": "mnist_bp_amp_v4_c1",
        "gain": "158.035",
        "lr_mult": "0.5",
        "lr": "0.00455595",
        "path": ROOT
        / "target_sat30"
        / "input_gain_158p035232544"
        / "lr_mult_0p5"
        / "lr_0p00455594609132"
        / "hard_sigmoid"
        / "mnist_bp_amp_v4_c1"
        / "seed_0"
        / "metrics.json",
    },
]


def pct(value: object) -> str:
    try:
        return f"{100.0 * float(value):.2f}%"
    except (TypeError, ValueError):
        return "n/a"


def read_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def build_block() -> tuple[str, bool]:
    now = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    rows = []
    complete = True
    for run in RUNS:
        metrics = read_metrics(run["path"])
        if metrics is None:
            complete = False
            rows.append(
                "| {label} | `{run_name}` | running | n/a | n/a | `{gain}` | `{lr_mult}` | `{lr}` |".format(
                    **run
                )
            )
        else:
            rows.append(
                "| {label} | `{run_name}` | complete | {best} / {final} | {epoch} | `{gain}` | `{lr_mult}` | `{lr}` |".format(
                    **run,
                    best=pct(metrics.get("best_test_accuracy")),
                    final=pct(metrics.get("final_test_accuracy")),
                    epoch=metrics.get("best_epoch", "n/a"),
                )
            )

    lines = [
        START,
        f"- Conv2 stride-1 hard-sigmoid repeats, launched locally in tmux `conv2_stride1_best30`, updated {now}.",
        "",
        "| Amp | Run | State | Best / final acc | Best epoch | Input gain | LR mult | LR |",
        "|---|---|---|---:|---:|---:|---:|---:|",
        *rows,
        "",
        f"- Output root: `{ROOT}`",
        END,
    ]
    return "\n".join(lines), complete


def replace_block(current_state: Path, block: str) -> None:
    text = current_state.read_text()
    if START in text and END in text:
        before = text.split(START, 1)[0].rstrip()
        after = text.split(END, 1)[1].lstrip()
        current_state.write_text(before + "\n\n" + block + "\n\n" + after)
        return

    marker = "\nCurrent Conv2 hard-sigmoid status:\n\n"
    if marker in text:
        before, after = text.split(marker, 1)
        current_state.write_text(before + marker + block + "\n\n" + after)
        return

    current_state.write_text(text.rstrip() + "\n\n" + block + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=600)
    args = parser.parse_args()

    while True:
        block, complete = build_block()
        replace_block(CURRENT_STATE, block)
        print(f"updated {CURRENT_STATE}; complete={complete}")
        if complete or not args.watch:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
