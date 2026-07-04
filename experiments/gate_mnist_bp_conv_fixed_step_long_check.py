#!/usr/bin/env python3
"""Gate fixed-step Conv redo final runs on the seed-0 long-check summary."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-lr-csv", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--fail-on-epoch-limited", action="store_true", default=True)
    parser.add_argument("--allow-epoch-limited", action="store_false", dest="fail_on_epoch_limited")
    parser.add_argument("--fail-on-unstable", action="store_true", default=True)
    parser.add_argument("--allow-unstable", action="store_false", dest="fail_on_unstable")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_path = Path(args.selected_lr_csv).expanduser().resolve()
    report_path = Path(args.report_json).expanduser().resolve()
    with selected_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    failures = []
    for row in rows:
        reasons = []
        if _truthy(row.get("collapsed")):
            reasons.append("collapsed")
        if args.fail_on_unstable and _truthy(row.get("unstable")):
            reasons.append("unstable")
        if args.fail_on_epoch_limited and _truthy(row.get("epoch_limited")):
            reasons.append("epoch_limited")
        if reasons:
            failures.append(
                {
                    "conv_depth": row.get("conv_depth"),
                    "non_linearity": row.get("non_linearity"),
                    "run_name": row.get("run_name"),
                    "reasons": reasons,
                    "best_test_accuracy": row.get("best_test_accuracy"),
                    "final_test_accuracy": row.get("final_test_accuracy"),
                    "best_epoch": row.get("best_epoch"),
                    "epochs": row.get("epochs"),
                    "run_dir": row.get("run_dir"),
                }
            )
    report = {
        "selected_lr_csv": str(selected_path),
        "num_rows": len(rows),
        "num_failures": len(failures),
        "failures": failures,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"[long-check-gate] rows={len(rows)} failures={len(failures)} report={report_path}")
    if failures:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
