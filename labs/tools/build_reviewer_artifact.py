#!/usr/bin/env python3
"""Build a reviewer-facing artifact folder for the digits paper.

The output is intentionally smaller than the full research tree.  It contains
exact paper figures/tables, a runnable digits-core reproduction pack, and
curated source inputs for supplemental panels that are not part of the core
pack.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import textwrap
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAPER_ROOT = PROJECT_ROOT / "labs" / "my_paper_tex"
DEFAULT_TEX = PAPER_ROOT / "main2.tex"
DEFAULT_OUTPUT = PROJECT_ROOT / "papers" / "reviewer_artifact"
CORE_PACK = PROJECT_ROOT / "papers" / "paper_core_repro_pack"

FIGURE_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
INPUT_RE = re.compile(r"\\input\{([^}]+)\}")

CORE_FIGURE_PREFIXES = (
    "figures/experiments/digits_error_vs_iter/",
    "figures/experiments/digits_error_vs_vol_tol/",
    "figures/experiments/digits_timing_cpu/",
)

CORE_TABLES = {
    "tables/table1_accuracy_longtable.tex",
    "tables/table2_runtime_decomposition_longtable.tex",
    "tables/table_timing_run_accuracies_input_gain_longtable.tex",
}

CURATED_DIRS = (
    (
        "mnist_figures",
        PROJECT_ROOT / "labs" / "figures_for_paper_digits" / "mnist_figures",
        "MNIST-scale accuracy and PCA sweep summary JSON plus rendered PNG/SVG assets.",
    ),
    (
        "conditioning_vs_runtime_experimental",
        PROJECT_ROOT / "labs" / "figures_for_paper_digits" / "conditioning_vs_runtime_experimental",
        "Conditioning-versus-runtime CSV/JSON inputs and rendered supplementary figures.",
    ),
    (
        "component_timing_double_diode_h3_w128",
        PROJECT_ROOT
        / "labs"
        / "figures_for_paper_digits"
        / "inner_outer_iter"
        / "double_diode_exponential_hidden_3_hidden_128_component_timing_float64_overrelaxed",
        "Component timing CSV/JSON summary and rendered comparison bars.",
    ),
    (
        "overrelaxation_error_vs_iter",
        PROJECT_ROOT
        / "labs"
        / "figures_for_paper_digits"
        / "supplementary_figures"
        / "error_vs_overrelaxation",
        "Overrelaxation error-vs-iteration summary JSON and rendered figure.",
    ),
)

ACCURACY_LADDER_SOURCE = (
    PROJECT_ROOT
    / "labs"
    / "figures_for_paper_digits"
    / "timings"
    / "selected_for_paper"
    / "accuracy_ladder_2x128_initial_50_70_80_90_best_spice_cd_20260521"
    / "spice_expclip165_20260521"
)

SOURCE_TOOL_NAMES = (
    "build_digits_core_repro_pack.py",
    "build_timing_paper_assets_from_bundle.py",
    "plot_spice_vs_coordinate_descent_loglog.py",
    "plot_mnist_accuracy_with_perfect_diode.py",
    "plot_pca_error_p90_grid.py",
    "plot_overrelaxation_error_vs_iter.py",
    "plot_conditioning_vs_runtime.py",
    "plot_component_timing_comparison_bar.py",
    "build_accuracy_ladder_effective_hessian.py",
)


@dataclass(frozen=True)
class TexRef:
    line: int
    path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tex-file", type=Path, default=DEFAULT_TEX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true", help="Replace an existing artifact directory.")
    return parser.parse_args()


def strip_tex_comment(line: str) -> str:
    """Drop unescaped TeX comments for simple include parsing."""
    for index, char in enumerate(line):
        if char == "%" and (index == 0 or line[index - 1] != "\\"):
            return line[:index]
    return line


def parse_tex_refs(tex_file: Path) -> tuple[list[TexRef], list[TexRef]]:
    figures: list[TexRef] = []
    inputs: list[TexRef] = []
    seen_figures: set[str] = set()
    seen_inputs: set[str] = set()

    for line_number, raw_line in enumerate(tex_file.read_text(encoding="utf-8").splitlines(), start=1):
        line = strip_tex_comment(raw_line)
        for path in FIGURE_RE.findall(line):
            if path not in seen_figures:
                figures.append(TexRef(line_number, path))
                seen_figures.add(path)
        for path in INPUT_RE.findall(line):
            if path not in seen_inputs:
                inputs.append(TexRef(line_number, path))
                seen_inputs.add(path)
    return figures, inputs


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Expected {label} to be an existing file. Provided value: {path}")
    return path


def require_dir(path: Path, label: str) -> Path:
    if not path.is_dir():
        raise NotADirectoryError(f"Expected {label} to be an existing directory. Provided value: {path}")
    return path


def copy_file(src: Path, dst: Path) -> None:
    require_file(src, "source file")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path) -> None:
    require_dir(src, "source directory")
    shutil.copytree(
        src,
        dst,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", ".mypy_cache"),
    )


def copy_paper_exact(tex_file: Path, output_dir: Path, figures: list[TexRef], inputs: list[TexRef]) -> None:
    paper_source = output_dir / "paper_source"
    copy_file(tex_file, paper_source / tex_file.name)
    refs = PAPER_ROOT / "refs.bib"
    if refs.exists():
        copy_file(refs, paper_source / refs.name)

    exact_root = output_dir / "paper_exact"
    for ref in figures:
        copy_file(PAPER_ROOT / ref.path, exact_root / ref.path)
    for ref in inputs:
        copy_file(PAPER_ROOT / ref.path, exact_root / ref.path)


def copy_accuracy_ladder(output_dir: Path) -> list[str]:
    copied: list[str] = []
    dst_root = output_dir / "curated_inputs" / "accuracy_ladder_timing"
    require_dir(ACCURACY_LADDER_SOURCE, "accuracy-ladder source directory")

    for src in sorted(ACCURACY_LADDER_SOURCE.iterdir()):
        if src.is_file() and src.suffix.lower() in {".csv", ".json", ".md", ".tex"}:
            dst = dst_root / src.name
            copy_file(src, dst)
            copied.append(str(dst.relative_to(output_dir)))

    hessian_dir = ACCURACY_LADDER_SOURCE / "effective_hessian_20260521"
    if hessian_dir.exists():
        copy_tree(hessian_dir, dst_root / hessian_dir.name)
        copied.append(str((dst_root / hessian_dir.name).relative_to(output_dir)))

    cd_index = ACCURACY_LADDER_SOURCE / "cd_rerun_maxicao_20260521" / "cd_run_index.json"
    if cd_index.exists():
        dst = dst_root / "cd_rerun_maxicao_20260521" / cd_index.name
        copy_file(cd_index, dst)
        copied.append(str(dst.relative_to(output_dir)))
    return copied


def copy_curated_inputs(output_dir: Path) -> list[dict[str, str]]:
    copied: list[dict[str, str]] = []
    for name, src, description in CURATED_DIRS:
        dst = output_dir / "curated_inputs" / name
        copy_tree(src, dst)
        copied.append(
            {
                "name": name,
                "path": str(dst.relative_to(output_dir)),
                "description": description,
            }
        )
    copied.append(
        {
            "name": "accuracy_ladder_timing",
            "path": "curated_inputs/accuracy_ladder_timing",
            "description": "Accuracy-ladder table CSV/JSON/TeX/Markdown plus small hessian summaries; heavy raw rerun payload omitted.",
        }
    )
    copy_accuracy_ladder(output_dir)
    return copied


def copy_source_tools(output_dir: Path) -> list[str]:
    copied: list[str] = []
    tools_dir = PROJECT_ROOT / "labs" / "tools"
    dst_dir = output_dir / "source_tools" / "labs_tools"
    for name in SOURCE_TOOL_NAMES:
        src = tools_dir / name
        if src.exists():
            copy_file(src, dst_dir / name)
            copied.append(str((dst_dir / name).relative_to(output_dir)))
    return copied


def classify_figure(path: str) -> str:
    if path.startswith(CORE_FIGURE_PREFIXES):
        return "regenerated_by_paper_core_repro_pack"
    if "/mnist/" in path:
        return "exact_asset_plus_curated_mnist_inputs"
    if "digits_overrelaxation" in path:
        return "exact_asset_plus_curated_overrelaxation_summary"
    if "conditioning_vs_runtime" in path:
        return "exact_asset_plus_curated_conditioning_inputs"
    if "supplementary_timing" in path:
        return "exact_asset_plus_curated_component_timing_inputs"
    return "exact_static_asset_bundled"


def classify_input(path: str) -> str:
    if path in CORE_TABLES:
        return "copied_by_paper_core_repro_pack_and_exact_table_bundled"
    if path == "tables/table_accuracy_ladder_timing.tex":
        return "exact_table_plus_curated_accuracy_ladder_inputs"
    return "exact_table_bundled"


def write_readme(output_dir: Path, tex_file: Path, figures: list[TexRef], inputs: list[TexRef]) -> None:
    text = f"""
    # Reviewer Artifact

    This folder is intended to accompany the paper submission.  It is a compact
    reviewer-facing artifact, not the full research workspace.

    ## Contents

    - `paper_source/`: the active TeX source copied from `{tex_file.relative_to(PROJECT_ROOT)}` plus `refs.bib`.
    - `paper_exact/`: exact figure and table files referenced by the active TeX source.
    - `paper_core_repro_pack/`: runnable self-contained reproduction pack for the core digits experiments.
    - `curated_inputs/`: additional CSV/JSON/NPZ/PNG/SVG/PDF assets for paper panels outside the core pack.
    - `source_tools/`: plotting/build scripts from the research tree that produced the curated non-core assets.
    - `COVERAGE.md`: line-by-line figure/table coverage for the active TeX source.
    - `manifest.json`: machine-readable inventory.

    ## Quick Checks

    From this directory:

    ```bash
    cd paper_core_repro_pack
    python scripts/reproduce.py verify
    python scripts/reproduce.py figures
    python scripts/reproduce.py list
    ```

    The core pack regenerates the digits error-vs-iteration, voltage-tolerance,
    and CPU timing panels from bundled curated inputs.  It can also rerun
    validation/NPZ comparisons for the bundled digits jobs.

    ## Validation Reruns

    The runnable pack includes 195 bundled digits validation jobs:

    - `timing`: 42 jobs
    - `error_vs_iter`: 108 jobs
    - `vol_tol`: 45 jobs

    Recommended CPU smoke checks:

    ```bash
    cd paper_core_repro_pack
    python scripts/reproduce.py validate --group timing --device cpu --limit 1
    python scripts/reproduce.py compare --group timing --limit 1
    python scripts/reproduce.py validate --group error_vs_iter --device cpu --limit 1
    python scripts/reproduce.py compare --group error_vs_iter --limit 1
    python scripts/reproduce.py validate --group vol_tol --device cpu --limit 1
    python scripts/reproduce.py compare --group vol_tol --limit 1
    ```

    Full reruns are available with `python scripts/reproduce.py all --device cpu`
    or `--device cuda`, but they are more expensive than the smoke checks.  The
    generated validation states are written under `paper_core_repro_pack/outputs/validation/`;
    comparison summaries are written under `paper_core_repro_pack/outputs/comparisons/`.

    ## Coverage Summary

    The active TeX source references {len(figures)} unique figure assets and
    {len(inputs)} table inputs.  All referenced files are copied exactly under
    `paper_exact/`.

    The nine core digits result panels are regenerated by
    `paper_core_repro_pack/scripts/reproduce.py figures`.  Other panels are
    bundled as exact paper assets with their curated source CSV/JSON/NPZ inputs
    where those inputs are available.

    ## Deliberate Exclusions

    This artifact omits the full training logs, Optuna/TensorBoard histories,
    full raw SPICE regeneration workflows, and large historical rerun archives.
    Timing plots use curated timing CSVs because wall-clock measurements are
    machine-dependent.
    """
    (output_dir / "README.md").write_text(textwrap.dedent(text).lstrip(), encoding="utf-8")


def write_coverage(output_dir: Path, figures: list[TexRef], inputs: list[TexRef]) -> None:
    lines = [
        "# Paper Coverage",
        "",
        "This file maps the active TeX references to the reviewer artifact contents.",
        "",
        "## Figures",
        "",
        "| TeX line | Paper path | Artifact path | Coverage |",
        "| ---: | --- | --- | --- |",
    ]
    for ref in figures:
        lines.append(
            f"| {ref.line} | `{ref.path}` | `paper_exact/{ref.path}` | {classify_figure(ref.path)} |"
        )

    lines.extend(
        [
            "",
            "## Tables",
            "",
            "| TeX line | Paper path | Artifact path | Coverage |",
            "| ---: | --- | --- | --- |",
        ]
    )
    for ref in inputs:
        lines.append(
            f"| {ref.line} | `{ref.path}` | `paper_exact/{ref.path}` | {classify_input(ref.path)} |"
        )
    (output_dir / "COVERAGE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(output_dir: Path) -> None:
    lines: list[str] = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
            continue
        rel_path = path.relative_to(output_dir)
        if rel_path == Path("checksums.sha256"):
            continue
        if len(rel_path.parts) >= 2 and rel_path.parts[:2] == ("paper_core_repro_pack", "outputs"):
            continue
        lines.append(f"{sha256(path)}  {rel_path}")
    (output_dir / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(
    output_dir: Path,
    tex_file: Path,
    figures: list[TexRef],
    inputs: list[TexRef],
    curated_inputs: list[dict[str, str]],
    source_tools: list[str],
) -> None:
    manifest = {
        "artifact": "reviewer_artifact",
        "active_tex": str(tex_file.relative_to(PROJECT_ROOT)),
        "paper_figures": [
            {
                "line": ref.line,
                "paper_path": ref.path,
                "artifact_path": f"paper_exact/{ref.path}",
                "coverage": classify_figure(ref.path),
            }
            for ref in figures
        ],
        "paper_tables": [
            {
                "line": ref.line,
                "paper_path": ref.path,
                "artifact_path": f"paper_exact/{ref.path}",
                "coverage": classify_input(ref.path),
            }
            for ref in inputs
        ],
        "runnable_packs": [
            {
                "name": "paper_core_repro_pack",
                "path": "paper_core_repro_pack",
                "verify_command": "python scripts/reproduce.py verify",
                "figure_command": "python scripts/reproduce.py figures",
            }
        ],
        "curated_inputs": curated_inputs,
        "source_tools": source_tools,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def build_artifact(tex_file: Path, output_dir: Path, *, force: bool) -> None:
    require_file(tex_file, "active TeX file")
    require_dir(CORE_PACK, "digits core reproduction pack")

    if output_dir.exists():
        if not force:
            raise FileExistsError(
                f"Expected output directory not to exist. Provided value: {output_dir}. Use --force to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    figures, inputs = parse_tex_refs(tex_file)
    copy_paper_exact(tex_file, output_dir, figures, inputs)
    copy_tree(CORE_PACK, output_dir / "paper_core_repro_pack")
    curated_inputs = copy_curated_inputs(output_dir)
    source_tools = copy_source_tools(output_dir)
    write_readme(output_dir, tex_file, figures, inputs)
    write_coverage(output_dir, figures, inputs)
    write_manifest(output_dir, tex_file, figures, inputs, curated_inputs, source_tools)
    write_checksums(output_dir)


def main() -> int:
    args = parse_args()
    build_artifact(args.tex_file.resolve(), args.output_dir.resolve(), force=args.force)
    print(f"wrote reviewer artifact: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
