#!/usr/bin/env python3
"""
Iterative Anderson parameter search:
1) run linspace simulation
2) compute error vs spice
3) decide next (heuristic) + log to markdown
4) repeat
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import subprocess
from datetime import datetime
import time
from pathlib import Path
from typing import Any


def _default_python_bin() -> str:
    preferred = Path("/home/filip/miniconda3/envs/py312/bin/python")
    if preferred.exists():
        return str(preferred)
    return "python3"


def _default_compare_script() -> str:
    return "/home/filip/server_code/labs/tools/compare_npz.sh"


def _default_small_network() -> str:
    return "/home/filip/server_code/labs/small_network.py"


def _parse_run_dir(output: str) -> Path | None:
    # Look for: Saved metadata to <run_dir>/run_metadata.json
    match = re.search(r"Saved metadata to (.+/run_metadata\\.json)", output)
    if not match:
        return None
    meta_path = Path(match.group(1)).expanduser().resolve()
    return meta_path.parent


def _find_latest_run_dir(output_root: Path, *, since_ts: float) -> Path | None:
    if not output_root.exists():
        return None
    candidates = []
    for run_dir in output_root.glob("*_linspace"):
        meta = run_dir / "run_metadata.json"
        if not meta.exists():
            continue
        mtime = meta.stat().st_mtime
        if mtime >= since_ts - 1.0:
            candidates.append((mtime, run_dir))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _load_percentile(summary_path: Path, metric: str) -> float | None:
    data = json.loads(summary_path.read_text())
    percentiles = data.get("node_weighted_rel_l1_percentiles", {})
    value = percentiles.get(metric)
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return math.nan
    return value


def _write_markdown_header(md_path: Path) -> None:
    if md_path.exists():
        return
    header = (
        "# Anderson Search Log\n\n"
        "Run log for iterative Anderson sweeps (linspace + error vs SPICE).\n\n"
        "| step | m | omega | reg | run_dir | metric | value | decision |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
    )
    md_path.write_text(header)


def _append_markdown_row(
    md_path: Path,
    *,
    step: int,
    m: int,
    omega: float,
    reg: float,
    run_dir: Path | None,
    metric: str,
    value: float | None,
    decision: str,
) -> None:
    run_dir_text = str(run_dir) if run_dir else "n/a"
    value_text = "NaN" if value is not None and math.isnan(value) else (
        f"{value:.6g}" if value is not None else "n/a"
    )
    line = f"| {step} | {m} | {omega:.3g} | {reg:.3g} | {run_dir_text} | {metric} | {value_text} | {decision} |\n"
    with md_path.open("a", encoding="utf-8") as handle:
        handle.write(line)


def _prepare_config(
    base_cfg: dict[str, Any],
    *,
    m: int,
    omega: float,
    reg: float,
    linspace_samples: int | None,
) -> dict[str, Any]:
    cfg = dict(base_cfg)
    cfg["minimizer_impl"] = "andersson"
    cfg["anderson_m"] = int(m)
    cfg["anderson_omega"] = float(omega)
    cfg["anderson_reg"] = float(reg)
    cfg.setdefault("adaptive_equilibrium", True)
    if linspace_samples is not None:
        cfg["linspace_samples"] = int(linspace_samples)
    return cfg


def _run_command(cmd: list[str], *, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True, env=env)


def _decide_next(
    *,
    current_m: int,
    current_omega: float,
    current_reg: float,
    metric_value: float | None,
    best_value: float | None,
    min_m: int,
    max_m: int,
    min_omega: float,
    max_omega: float,
) -> tuple[int, float, float, str, float | None]:
    # Heuristic:
    # - If NaN: reduce omega, reduce m.
    # - If improved: keep m, slightly increase omega.
    # - If worse: reduce omega; if already low, reduce m.
    decision = ""
    next_m = current_m
    next_omega = current_omega
    next_reg = current_reg
    next_best = best_value

    if metric_value is None or math.isnan(metric_value):
        next_omega = max(min_omega, current_omega * 0.5)
        next_m = max(min_m, current_m - 1)
        decision = "NaN -> reduce omega, reduce m"
        return next_m, next_omega, next_reg, decision, next_best

    if best_value is None or metric_value < best_value:
        next_best = metric_value
        next_omega = min(max_omega, current_omega * 1.25)
        decision = "improved -> try higher omega"
        return next_m, next_omega, next_reg, decision, next_best

    # worse
    if current_omega > min_omega * 1.1:
        next_omega = max(min_omega, current_omega * 0.7)
        decision = "worse -> reduce omega"
    else:
        next_m = max(min_m, current_m - 1)
        decision = "worse -> reduce m"
    return next_m, next_omega, next_reg, decision, next_best


def _param_key(m: int, omega: float, reg: float, *, omega_digits: int = 6, reg_digits: int = 12) -> tuple[int, float, float]:
    return (int(m), round(float(omega), omega_digits), round(float(reg), reg_digits))


def _ensure_unique_params(
    *,
    m: int,
    omega: float,
    reg: float,
    seen: set[tuple[int, float, float]],
    min_m: int,
    max_m: int,
    min_omega: float,
    max_omega: float,
    max_attempts: int = 50,
) -> tuple[int, float, float, str]:
    key = _param_key(m, omega, reg)
    if key not in seen:
        return m, omega, reg, ""
    for _ in range(max_attempts):
        m = random.randint(min_m, max_m)
        omega = random.uniform(min_omega, max_omega)
        key = _param_key(m, omega, reg)
        if key not in seen:
            return m, omega, reg, "duplicate -> random restart"
    raise SystemExit("Unable to sample unique (m, omega, reg) after 50 attempts.")


def _random_restart(
    *,
    min_m: int,
    max_m: int,
    min_omega: float,
    max_omega: float,
    reg: float,
) -> tuple[int, float, float, str]:
    new_m = random.randint(min_m, max_m)
    new_omega = random.uniform(min_omega, max_omega)
    decision = "repeat detected -> random restart"
    return new_m, new_omega, reg, decision


def main() -> int:
    p = argparse.ArgumentParser(description="Iterative Anderson parameter search.")
    p.add_argument("--config", required=True, help="Base config JSON path.")
    p.add_argument("--weights", required=True, help="Model .pt path.")
    p.add_argument("--spice-npz", required=True, help="Reference SPICE npz.")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--metric", default="p95", help="Percentile key to optimize (e.g., p90, p95, p99).")
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--m-start", type=int, default=None)
    p.add_argument("--omega-start", type=float, default=None)
    p.add_argument("--reg", type=float, default=None)
    p.add_argument(
        "--linspace-samples",
        type=int,
        default=None,
        help="Override linspace_samples in the config (per-axis grid count).",
    )
    p.add_argument("--m-min", type=int, default=1)
    p.add_argument("--m-max", type=int, default=8)
    p.add_argument("--omega-min", type=float, default=0.1)
    p.add_argument("--omega-max", type=float, default=1.0)
    p.add_argument("--md-log", default=None, help="Markdown log path.")
    p.add_argument("--python-bin", default=None)
    p.add_argument("--compare-script", default=None)
    p.add_argument("--small-network", default=None)
    args = p.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    weights_path = Path(args.weights).expanduser().resolve()
    spice_path = Path(args.spice_npz).expanduser().resolve()
    if not config_path.exists():
        raise SystemExit(f"Expected config path to exist; got {config_path}")
    if not weights_path.exists():
        raise SystemExit(f"Expected weights path to exist; got {weights_path}")
    if not spice_path.exists():
        raise SystemExit(f"Expected spice npz path to exist; got {spice_path}")

    base_cfg = json.loads(config_path.read_text())
    m = int(args.m_start) if args.m_start is not None else int(base_cfg.get("anderson_m", 4))
    omega = float(args.omega_start) if args.omega_start is not None else float(base_cfg.get("anderson_omega", 0.5))
    reg = float(args.reg) if args.reg is not None else float(base_cfg.get("anderson_reg", 1e-6))

    md_path = Path(args.md_log) if args.md_log else config_path.parent / "anderson_search.md"
    _write_markdown_header(md_path)

    python_bin = args.python_bin or _default_python_bin()
    compare_script = args.compare_script or _default_compare_script()
    small_network = args.small_network or _default_small_network()

    env = os.environ.copy()
    env.setdefault("KMP_DISABLE_SHM", "1")
    env.setdefault("KMP_SHM_DISABLE", "1")
    env.setdefault("OMP_NUM_THREADS", "1")

    best_value: float | None = None
    repeat_value: float | None = None
    repeat_count = 0
    repeat_eps = 1e-9
    seen_params: set[tuple[int, float, float]] = set()
    for step in range(1, args.steps + 1):
        m, omega, reg, pre_decision = _ensure_unique_params(
            m=m,
            omega=omega,
            reg=reg,
            seen=seen_params,
            min_m=args.m_min,
            max_m=args.m_max,
            min_omega=args.omega_min,
            max_omega=args.omega_max,
        )
        seen_params.add(_param_key(m, omega, reg))

        run_cfg = _prepare_config(
            base_cfg,
            m=m,
            omega=omega,
            reg=reg,
            linspace_samples=args.linspace_samples,
        )
        tmp_cfg = Path("/tmp") / f"anderson_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{step}.json"
        tmp_cfg.write_text(json.dumps(run_cfg, indent=2))

        cmd = [
            python_bin,
            small_network,
            "--mode",
            "linspace",
            "--config",
            str(tmp_cfg),
            "--weights",
            str(weights_path),
            "--batch-size",
            str(args.batch_size),
        ]
        start_ts = time.time()
        res = _run_command(cmd, env=env)
        combined_output = (res.stdout or "") + "\n" + (res.stderr or "")
        run_dir = _parse_run_dir(combined_output)
        if run_dir is None:
            output_root = Path(base_cfg.get("output_dir", config_path.parent)).expanduser().resolve()
            run_dir = _find_latest_run_dir(output_root, since_ts=start_ts)
        if res.returncode != 0 or run_dir is None:
            decision = f"run failed (code {res.returncode})"
            _append_markdown_row(
                md_path,
                step=step,
                m=m,
                omega=omega,
                reg=reg,
                run_dir=run_dir,
                metric=args.metric,
                value=None,
                decision=decision,
            )
            if res.stdout:
                print(res.stdout)
            if res.stderr:
                print(res.stderr)
            break

        error_dir = run_dir / "error_npz"
        run_npz = run_dir / "linspace_states.npz"
        compare_cmd = [
            compare_script,
            str(run_npz),
            str(spice_path),
            str(error_dir),
        ]
        res_cmp = _run_command(compare_cmd, env=env)
        if res_cmp.returncode != 0:
            decision = f"error compare failed (code {res_cmp.returncode})"
            _append_markdown_row(
                md_path,
                step=step,
                m=m,
                omega=omega,
                reg=reg,
                run_dir=run_dir,
                metric=args.metric,
                value=None,
                decision=decision,
            )
            print(res_cmp.stdout)
            print(res_cmp.stderr)
            break

        summary_path = error_dir / "cross_layer_rel_l1_percentiles_node_weighted.json"
        metric_value = _load_percentile(summary_path, args.metric)

        next_m, next_omega, next_reg, decision, best_value = _decide_next(
            current_m=m,
            current_omega=omega,
            current_reg=reg,
            metric_value=metric_value,
            best_value=best_value,
            min_m=args.m_min,
            max_m=args.m_max,
            min_omega=args.omega_min,
            max_omega=args.omega_max,
        )

        if metric_value is not None and not math.isnan(metric_value):
            if repeat_value is not None and abs(metric_value - repeat_value) <= repeat_eps:
                repeat_count += 1
            else:
                repeat_value = metric_value
                repeat_count = 1

            if repeat_count >= 5:
                next_m, next_omega, next_reg, restart_decision = _random_restart(
                    min_m=args.m_min,
                    max_m=args.m_max,
                    min_omega=args.omega_min,
                    max_omega=args.omega_max,
                    reg=reg,
                )
                decision = f"{decision}; {restart_decision}"
                repeat_count = 0
                repeat_value = None
        if pre_decision:
            decision = f"{pre_decision}; {decision}"

        _append_markdown_row(
            md_path,
            step=step,
            m=m,
            omega=omega,
            reg=reg,
            run_dir=run_dir,
            metric=args.metric,
            value=metric_value,
            decision=decision,
        )

        m, omega, reg = next_m, next_omega, next_reg

    print(f"Wrote log to {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
