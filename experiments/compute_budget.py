"""Small CSV accounting for hard GPU-hour reservations; never launches jobs."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
import fcntl
import math
from pathlib import Path
from zoneinfo import ZoneInfo

FIELDS = ("name", "state", "reserved_gpu_hours", "actual_gpu_hours")


def committed_hours(rows):
    total = 0.
    names = set()
    for row in rows:
        if row["name"] in names:
            raise ValueError("Duplicate budget entry")
        names.add(row["name"])
        if row["state"] not in {"reserved", "settled"}:
            raise ValueError("Unknown accounting state")
        value = float(row["actual_gpu_hours"] if row["state"] == "settled" else row["reserved_gpu_hours"])
        if not math.isfinite(value) or value < 0:
            raise ValueError("Invalid GPU time")
        total += value
    return total


def reserve(rows, name, hours, limit):
    if any(row["name"] == name for row in rows):
        raise ValueError("Reservation already exists; do not launch twice")
    if not math.isfinite(hours) or hours <= 0 or not math.isfinite(limit) or limit <= 0:
        raise ValueError("Invalid reservation or limit")
    candidate = [*rows, dict(name=name, state="reserved", reserved_gpu_hours=hours, actual_gpu_hours="")]
    if committed_hours(candidate) > limit:
        raise ValueError(f"Hard GPU-hour limit {limit} exceeded")
    return candidate


def extra_5090_deadline(now, wall_seconds, margin_seconds=3600):
    if now.tzinfo is None:
        raise ValueError("Timezone-aware time is required")
    local = now.astimezone(ZoneInfo("Europe/Paris"))
    if local.weekday() < 5:
        raise ValueError("An extra RTX 5090 is forbidden on weekdays")
    monday = (local + timedelta(days=7-local.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    if wall_seconds <= 0 or local.timestamp() + wall_seconds + margin_seconds > monday.timestamp():
        raise ValueError("Full wall limit plus one-hour margin must fit before Monday")
    return monday


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", type=Path)
    parser.add_argument("--limit", type=float, default=300.)
    sub = parser.add_subparsers(dest="action", required=True)
    add = sub.add_parser("reserve")
    add.add_argument("name")
    add.add_argument("hours", type=float)
    close = sub.add_parser("settle")
    close.add_argument("name")
    close.add_argument("hours", type=float)
    sub.add_parser("show")
    args = parser.parse_args()
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    # All admissions use this one local ledger; remote workers have fixed limits.
    with args.ledger.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = list(csv.DictReader(args.ledger.open())) if args.ledger.exists() else []
        if args.action == "reserve":
            rows = reserve(rows, args.name, args.hours, args.limit)
        elif args.action == "settle":
            matches = [r for r in rows if r["name"] == args.name]
            if len(matches) != 1 or matches[0]["state"] != "reserved":
                raise ValueError("Expected one outstanding reservation")
            if not math.isfinite(args.hours) or not 0 <= args.hours <= float(matches[0]["reserved_gpu_hours"]):
                raise ValueError("Actual time exceeds reservation; reconcile the wall-limit breach")
            matches[0].update(state="settled", actual_gpu_hours=args.hours)
        if args.action != "show":
            temporary = args.ledger.with_suffix(".tmp")
            with temporary.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            temporary.replace(args.ledger)
        total = committed_hours(rows)
        print(f"Committed {total:.6f} / {args.limit:.6f} GPU-hours; available {args.limit-total:.6f}")


if __name__ == "__main__":
    main()
