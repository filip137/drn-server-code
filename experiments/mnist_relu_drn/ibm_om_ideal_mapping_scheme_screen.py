"""Read-only ideal-mapping screen for IBM OM deployment schemes.

The screen separates two axes before any HWA or program-and-verify noise is
introduced:

* four versus eight physical conductance branches per logical dual-rail
  synapse; and
* a lower-bound control versus a fixed reference conductance ``r``.

For a positive logical weight, the four-device fixed-reference construction
is ``[[a, r], [r, a]]``; for a negative weight the roles are swapped.  The
eight-device construction places a fixed reference branch under every one of
the four dual-rail edges.  Its transfer uses ``G_a-G_r`` and its loading uses
``G_a+G_r`` through :class:`SignedDenseResistive`.

This is an ideal, model-relative circuit screen.  It does not sample devices,
program pulses, claim an absolute conductance calibration, or train a model.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from itertools import product
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from experiments.artifacts import atomic_write_json, sha256_file
from experiments.definitions import resolve_experiment_config
from experiments.mnist_relu_drn.components import (
    apply_targets,
    build_student_stack,
    collect_calibration,
    fit_positive_logit_gain,
    signed_dual_rail_lift,
)
from experiments.mnist_relu_drn.runtime import _evaluate, _load_teacher
from experiments.mnist_shared import build_mnist_loaders
from experiments.schema import RunMode


SCHEMA = "ebl.mnist_relu_drn.ibm_om_ideal_mapping_scheme_screen"
SCHEMA_VERSION = 1
_LAYOUTS = ("halves", "paired")


@dataclass(frozen=True)
class Scheme:
    """One reference-use by device-count cell in the screen."""

    name: str
    device_count_per_logical_weight: int
    use_fixed_reference: bool
    reference_fraction: float

    @property
    def encoding(self) -> str:
        return (
            "single"
            if self.device_count_per_logical_weight == 4
            else "differential"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare four/eight-device and with/without-fixed-reference "
            "schemes at ideal mapped targets only."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--teacher-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--reference-fractions",
        type=float,
        nargs="+",
        default=(0.25, 0.5),
        help=(
            "Fixed r locations as fractions of the configured nonnegative "
            "conductance span. The lower-bound controls are always included."
        ),
    )
    parser.add_argument(
        "--scale-fractions",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Optional mapping-scale grid. By default use the fractions in "
            "the supplied config."
        ),
    )
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser


def _finite_fraction(
    value: float,
    *,
    name: str,
    include_zero: bool,
    include_one: bool = False,
) -> float:
    numeric = float(value)
    lower_ok = numeric >= 0.0 if include_zero else numeric > 0.0
    upper_ok = numeric <= 1.0 if include_one else numeric < 1.0
    if not math.isfinite(numeric) or not lower_ok or not upper_ok:
        left = "[" if include_zero else "("
        right = "]" if include_one else ")"
        interval = f"{left}0, 1{right}"
        raise ValueError(
            f"Expected {name} to be a finite fraction in {interval}. "
            f"Provided value: {value!r}."
        )
    return numeric


def _normalized(weight: torch.Tensor) -> torch.Tensor:
    if weight.ndim != 2 or not bool(torch.isfinite(weight).all()):
        raise ValueError("Expected a finite rank-2 logical weight matrix.")
    maximum = weight.detach().abs().max()
    if float(maximum.item()) == 0.0:
        raise ValueError("Expected a nonzero logical weight matrix.")
    return weight.detach() / maximum


def _quad_contrast(matrix: torch.Tensor, *, layout: str) -> torch.Tensor:
    """Return ``G++-G+--G-++G--`` for a physical rail matrix."""

    if matrix.ndim != 2 or matrix.shape[0] % 2 or matrix.shape[1] % 2:
        raise ValueError("Expected an even-by-even rank-2 rail matrix.")
    rows = matrix.shape[0] // 2
    columns = matrix.shape[1] // 2
    plus_columns = (
        torch.arange(columns, device=matrix.device)
        if layout == "halves"
        else torch.arange(columns, device=matrix.device) * 2
    )
    if layout not in _LAYOUTS:
        raise ValueError(f"Expected a canonical dual-rail layout: {layout!r}.")
    minus_columns = (
        plus_columns + columns if layout == "halves" else plus_columns + 1
    )
    return (
        matrix[:rows][:, plus_columns]
        - matrix[:rows][:, minus_columns]
        - matrix[rows:][:, plus_columns]
        + matrix[rows:][:, minus_columns]
    )


def build_scheme_targets(
    logical_weights: Sequence[torch.Tensor],
    *,
    scheme: Scheme,
    scale_fractions: tuple[float, float],
    conductance_min: float,
    conductance_max: float,
) -> tuple[tuple[torch.Tensor, ...], dict[str, Any]]:
    """Map the same two logical matrices into one ideal physical scheme."""

    if len(logical_weights) != 2 or len(scale_fractions) != 2:
        raise ValueError("Expected exactly two logical layers and scale fractions.")
    if scheme.device_count_per_logical_weight not in {4, 8}:
        raise ValueError("Expected a four- or eight-device scheme.")
    lower = float(conductance_min)
    upper = float(conductance_max)
    if not math.isfinite(lower) or not math.isfinite(upper) or lower < 0 or upper <= lower:
        raise ValueError("Expected finite nonnegative increasing conductance bounds.")
    span = upper - lower
    reference_fraction = _finite_fraction(
        scheme.reference_fraction,
        name="scheme.reference_fraction",
        include_zero=True,
    )
    if not scheme.use_fixed_reference and reference_fraction != 0.0:
        raise ValueError("Expected a no-reference control to use fraction zero.")
    reference = (
        lower + reference_fraction * span
        if scheme.use_fixed_reference
        else lower
    )
    available = upper - reference

    targets: list[torch.Tensor] = []
    layer_reports = []
    for layer_index, (weight, fraction, layout) in enumerate(
        zip(logical_weights, scale_fractions, _LAYOUTS)
    ):
        scale_fraction = _finite_fraction(
            float(fraction),
            name=f"scale_fractions[{layer_index}]",
            include_zero=False,
            include_one=True,
        )
        lifted = signed_dual_rail_lift(
            _normalized(weight),
            target_layout=layout,
        )
        peak = float(lifted.max().item())
        scale = scale_fraction * available / peak
        active = reference + scale * lifted
        if scheme.device_count_per_logical_weight == 4:
            layer_targets = (active,)
            difference = active
            loading = active
        else:
            fixed_reference = torch.full_like(active, reference)
            layer_targets = (active, fixed_reference)
            difference = active - fixed_reference
            loading = active + fixed_reference
        targets.extend(layer_targets)

        logical_contrast = _quad_contrast(difference, layout=layout)
        difference_rms = float(difference.double().square().mean().sqrt().item())
        loading_mean = float(loading.double().mean().item())
        layer_reports.append(
            {
                "layer": layer_index,
                "layout": layout,
                "reference_conductance": reference,
                "scale_fraction": scale_fraction,
                "selected_active_scale": scale,
                "active_minimum": float(active.min().item()),
                "active_maximum": float(active.max().item()),
                "difference_rms": difference_rms,
                "mean_edge_denominator_loading": loading_mean,
                "difference_rms_over_mean_loading": (
                    difference_rms / loading_mean if loading_mean else None
                ),
                "mean_source_row_denominator_loading": float(
                    loading.double().sum(dim=1).mean().item()
                ),
                "mean_destination_column_denominator_loading": float(
                    loading.double().sum(dim=0).mean().item()
                ),
                "logical_contrast_rms": float(
                    logical_contrast.double().square().mean().sqrt().item()
                ),
                "logical_contrast_definition": "G++-G+--G-++G--",
            }
        )

    return tuple(targets), {
        "scheme": scheme.name,
        "encoding": scheme.encoding,
        "device_count_per_logical_weight": scheme.device_count_per_logical_weight,
        "use_fixed_reference": scheme.use_fixed_reference,
        "reference_fraction": reference_fraction,
        "reference_conductance": reference,
        "layers": layer_reports,
    }


def _schemes(reference_fractions: Iterable[float]) -> tuple[Scheme, ...]:
    references = tuple(
        _finite_fraction(value, name="--reference-fractions", include_zero=False)
        for value in reference_fractions
    )
    if len(set(references)) != len(references):
        raise ValueError("Expected unique --reference-fractions.")
    controls = (
        Scheme("four_without_fixed_r", 4, False, 0.0),
        Scheme("eight_without_fixed_r", 8, False, 0.0),
    )
    fixed = tuple(
        Scheme(f"four_with_fixed_r_{value:g}", 4, True, value)
        for value in references
    ) + tuple(
        Scheme(f"eight_with_fixed_r_{value:g}", 8, True, value)
        for value in references
    )
    return controls + fixed


def _ordered_labels(loader: Iterable) -> torch.Tensor:
    labels = [batch_labels.detach().cpu() for _inputs, batch_labels in loader]
    if not labels:
        raise ValueError("Expected the calibration loader to contain examples.")
    return torch.cat(labels).to(torch.long)


def run_screen(
    *,
    config_path: Path,
    teacher_weights_path: Path,
    reference_fractions: Sequence[float],
    scale_fractions: Sequence[float] | None,
    device: str,
    sample_limit: int | None,
) -> dict[str, Any]:
    """Run the deterministic ideal mapping screen and return its report."""

    config_path = config_path.expanduser().resolve()
    teacher_weights_path = teacher_weights_path.expanduser().resolve()
    if not config_path.is_file() or not teacher_weights_path.is_file():
        raise FileNotFoundError("Expected existing config and teacher checkpoint files.")
    definition, source_spec = resolve_experiment_config(config_path, RunMode.TRAIN)
    if definition.experiment_id != "mnist_relu_drn_kd.v1":
        raise ValueError("Expected the MNIST ReLU DRN KD composition root.")
    resolved_runtime = replace(source_spec.runtime, device=device)
    base_spec = replace(source_spec, runtime=resolved_runtime)
    fractions = tuple(
        _finite_fraction(
            value,
            name="--scale-fractions",
            include_zero=False,
            include_one=True,
        )
        for value in (
            source_spec.mapping.scale_fractions
            if scale_fractions is None
            else scale_fractions
        )
    )
    if len(set(fractions)) != len(fractions):
        raise ValueError("Expected unique scale fractions.")
    fraction_pairs = tuple(product(fractions, repeat=2))
    if sample_limit is not None and sample_limit < 1:
        raise ValueError("Expected --sample-limit to be positive or omitted.")

    loaders = build_mnist_loaders(
        source_spec.data,
        data_seed=source_spec.runtime.data_seed,
        calibration_examples=source_spec.mapping.calibration_examples,
        calibration_batch_size=source_spec.mapping.calibration_batch_size,
    )
    teacher, teacher_metadata = _load_teacher(
        teacher_weights_path,
        device=torch.device(device),
        spec=base_spec,
    )
    logical_weights = tuple(value.detach() for value in teacher.parameters())
    if len(logical_weights) != 2:
        raise ValueError("Expected the bias-free teacher to expose two matrices.")
    calibration_labels = _ordered_labels(loaders.calibration)

    arm_reports = []
    for scheme in _schemes(reference_fractions):
        torch.manual_seed(source_spec.runtime.seed)
        arm_spec = replace(
            base_spec,
            model=replace(base_spec.model, encoding=scheme.encoding),
        )
        stack = build_student_stack(arm_spec, enable_measured=False)
        candidates = []
        selected_targets = None
        selected_mapping = None
        selected_key = None
        for pair in fraction_pairs:
            targets, mapping = build_scheme_targets(
                logical_weights,
                scheme=scheme,
                scale_fractions=(float(pair[0]), float(pair[1])),
                conductance_min=arm_spec.model.conductance_min,
                conductance_max=arm_spec.model.conductance_max,
            )
            apply_targets(stack.bundle.catalog, targets)
            raw_scores, teacher_logits = collect_calibration(
                stack,
                teacher,
                loaders.calibration,
            )
            calibration = fit_positive_logit_gain(
                raw_scores,
                teacher_logits,
                gain_min=arm_spec.mapping.logit_gain_min,
                gain_max=arm_spec.mapping.logit_gain_max,
                steps=arm_spec.mapping.logit_gain_steps,
            )
            if raw_scores.shape[0] != calibration_labels.shape[0]:
                raise RuntimeError(
                    "Expected calibration scores and labels to have equal length."
                )
            student_prediction = raw_scores.detach().cpu().argmax(dim=1)
            teacher_prediction = teacher_logits.detach().cpu().argmax(dim=1)
            calibration.update(
                {
                    "student_accuracy": float(
                        student_prediction.eq(calibration_labels).double().mean().item()
                    ),
                    "teacher_accuracy": float(
                        teacher_prediction.eq(calibration_labels).double().mean().item()
                    ),
                    "teacher_agreement": float(
                        student_prediction.eq(teacher_prediction).double().mean().item()
                    ),
                }
            )
            candidate = {
                "scale_fractions": [float(pair[0]), float(pair[1])],
                "calibration": calibration,
                "mapping": mapping,
            }
            candidates.append(candidate)
            key = (
                -calibration["student_accuracy"],
                calibration["calibrated_kl"],
                pair,
            )
            if selected_key is None or key < selected_key:
                selected_key = key
                selected_targets = tuple(value.detach().clone() for value in targets)
                selected_mapping = mapping
        if selected_targets is None or selected_mapping is None:
            raise RuntimeError("Expected at least one mapping candidate.")
        selected_index = min(
            range(len(candidates)),
            key=lambda index: (
                -candidates[index]["calibration"]["student_accuracy"],
                candidates[index]["calibration"]["calibrated_kl"],
                tuple(candidates[index]["scale_fractions"]),
            ),
        )
        selected = candidates[selected_index]
        apply_targets(stack.bundle.catalog, selected_targets)
        stack.cost.gain = float(selected["calibration"]["gain"])
        metrics = _evaluate(
            stack,
            teacher,
            loaders.test,
            sample_limit=sample_limit,
        )
        arm_reports.append(
            {
                "scheme": scheme.name,
                "device_count_per_logical_weight": (
                    scheme.device_count_per_logical_weight
                ),
                "use_fixed_reference": scheme.use_fixed_reference,
                "reference_fraction": scheme.reference_fraction,
                "selection_domain": "training_split_calibration_subset",
                "selection_metric": (
                    "ideal_mapped_student_accuracy_then_calibrated_kl"
                ),
                "selected_index": selected_index,
                "selected": selected,
                "ideal_mapped_test": metrics,
                "candidates": candidates,
            }
        )
        print(
            f"{scheme.name}: ideal mapped accuracy "
            f"{100.0 * metrics['student_accuracy']:.2f}% "
            f"at scales {selected['scale_fractions']}",
            flush=True,
        )

    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "exploratory_read_only_ideal_mapping",
        "claim_boundary": (
            "No device sampling, pulse programming, HWA, training, or "
            "absolute-conductance calibration. Fixed-r fractions are a "
            "model-relative loading sensitivity screen."
        ),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "teacher_weights": str(teacher_weights_path),
        "teacher_weights_sha256": sha256_file(teacher_weights_path),
        "teacher_architecture": teacher_metadata.get("architecture"),
        "device": device,
        "sample_limit": sample_limit,
        "conductance_bounds": [
            source_spec.model.conductance_min,
            source_spec.model.conductance_max,
        ],
        "reference_fractions": list(reference_fractions),
        "scale_fractions": list(fractions),
        "scale_fraction_pairs": [list(pair) for pair in fraction_pairs],
        "scheme_contract": {
            "four_with_reference_positive_weight": (
                "G++=G--=a; G+-=G-+=r"
            ),
            "four_with_reference_negative_weight": (
                "G++=G--=r; G+-=G-+=a"
            ),
            "eight_device_transfer": "D=G_a-G_r",
            "eight_device_denominator_loading": "S=G_a+G_r",
            "reference_policy": "fixed; only active branch is raised",
        },
        "arms": arm_reports,
    }


def main() -> None:
    args = _parser().parse_args()
    if args.torch_threads < 1:
        raise ValueError("Expected --torch-threads to be positive.")
    torch.set_num_threads(args.torch_threads)
    report = run_screen(
        config_path=args.config,
        teacher_weights_path=args.teacher_weights,
        reference_fractions=args.reference_fractions,
        scale_fractions=args.scale_fractions,
        device=args.device,
        sample_limit=args.sample_limit,
    )
    atomic_write_json(args.output.expanduser().resolve(), report)


if __name__ == "__main__":
    main()
