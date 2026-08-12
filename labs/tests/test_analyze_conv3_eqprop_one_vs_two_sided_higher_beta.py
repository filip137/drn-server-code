from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from experiments import analyze_conv3_eqprop_one_vs_two_sided_higher_beta as analysis


def _parameter_row(parameter: str, *, scheme: str = "baseline") -> dict[str, str]:
    return {
        "schema": "perfectdiode-conv-current-eqprop-true-dtype-audit/v3",
        "architecture": "conv3",
        "scheme": scheme,
        "checkpoint_role": "best_validation",
        "eqprop_variant": "centered",
        "T": "64",
        "K": "64",
        "batch_index": "0",
        "batch_payload_sha256": "fixed-batch",
        "batch_source_indices_sha256": "fixed-indices",
        "parameter_name": parameter,
        "parameter_type": "DenseWeight" if parameter.startswith("Dense") else "ConvWeight",
        "bias_excluded": "True",
        "actual_beta": "300",
        "effective_beta": "300",
        "amplification_factor": "1",
        "beta_source": "metadata_not_run_name",
        "residual_limited": "False",
        "float32_eqprop_gradient_dtype": "torch.float32",
        "float64_eqprop_gradient_dtype": "torch.float64",
        "float32_bptt_gradient_dtype": "torch.float32",
        "float64_bptt_gradient_dtype": "torch.float64",
        "float32_eqprop_l2": "2",
        "float64_eqprop_l2": "2.5",
        "float32_bptt_l2": "1",
        "float64_bptt_l2": "1.25",
        "float32_exact_zero_fraction": "0",
        "float64_exact_zero_fraction": "0",
        "float32_eqprop_vs_bptt_cosine": "0.7",
        "float64_eqprop_vs_bptt_cosine": "0.9",
        "float32_eqprop_over_bptt_norm_ratio": "2",
        "float64_eqprop_over_bptt_norm_ratio": "2",
        "float32_eqprop_vs_bptt_symmetric_norm_delta": "0.6",
        "float64_eqprop_vs_bptt_symmetric_norm_delta": "0.6",
    }


def _displacement_row(
    state_layer: str,
    *,
    phase: str,
    reference: str,
    value: float,
    precision: str,
    scheme: str = "baseline",
) -> dict[str, str]:
    return {
        "architecture": "conv3",
        "scheme": scheme,
        "checkpoint_role": "best_validation",
        "eqprop_variant": "centered",
        "T": "64",
        "K": "64",
        "precision": precision,
        "batch_index": "0",
        "batch_payload_sha256": "fixed-batch",
        "batch_source_indices_sha256": "fixed-indices",
        "phase": phase,
        "actual_beta": "-300" if phase == "negative" else "300",
        "reference_kind": reference,
        "outcome": "ok",
        "state_layer_name": state_layer,
        "displacement_l2": str(value * 10.0),
        "relative_displacement": str(value),
        "active_set_transition_fraction": str(value / 10.0),
    }


def _centered_bundle() -> dict[str, object]:
    displacement_rows = []
    for precision in analysis.PRECISIONS:
        for state_layer in analysis.STATE_LAYER_BY_PARAMETER.values():
            displacement_rows.extend(
                [
                    _displacement_row(
                        state_layer,
                        phase="negative",
                        reference="matched_zero_K",
                        value=0.2,
                        precision=precision,
                    ),
                    _displacement_row(
                        state_layer,
                        phase="positive",
                        reference="matched_zero_K",
                        value=0.3,
                        precision=precision,
                    ),
                    _displacement_row(
                        state_layer,
                        phase="positive_minus_negative",
                        reference="negative_phase",
                        value=0.5,
                        precision=precision,
                    ),
                ]
            )
    return {
        "source_kind": "primary",
        "run_id": "a-name-that-does-not-encode-the-variant",
        "variant": "centered",
        "result_sha256": "result-sha",
        "batch_source_indices_sha256": "fixed-indices",
        "parameter_rows": [_parameter_row(name) for name in analysis.PARAMETERS],
        "displacement_rows": displacement_rows,
        "case_identity_by_scheme": {
            "baseline": {
                "source_selection_row_sha256": "selection-baseline",
                "best_checkpoint_sha256": "checkpoint-baseline",
            }
        },
    }


def _normalized_row(
    *,
    source_kind: str,
    scheme: str,
    precision: str,
    variant: str,
    beta: float,
    parameter: str,
) -> dict[str, object]:
    return {
        "source_kind": source_kind,
        "run_id": f"opaque-{source_kind}-{scheme}-{precision}-{variant}-{beta}-{parameter}",
        "eqprop_variant": variant,
        "scheme": scheme,
        "precision": precision,
        "T": 64,
        "K": 64,
        "injected_beta": beta,
        "parameter_name": parameter,
        "batch_payload_sha256": "fixed-batch",
        "batch_source_indices_sha256": "fixed-indices",
        "bptt_l2": 1.0,
        "source_selection_row_sha256": f"selection-{scheme}",
        "best_checkpoint_sha256": f"checkpoint-{scheme}",
    }


def _complete_normalized_surface() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scheme in analysis.SCHEMES:
        for precision in analysis.PRECISIONS:
            for variant in analysis.VARIANTS:
                for parameter in analysis.PARAMETERS:
                    rows.append(
                        _normalized_row(
                            source_kind="primary",
                            scheme=scheme,
                            precision=precision,
                            variant=variant,
                            beta=300.0,
                            parameter=parameter,
                        )
                    )
            for parameter in analysis.PARAMETERS:
                rows.append(
                    _normalized_row(
                        source_kind="anchor",
                        scheme=scheme,
                        precision=precision,
                        variant="positive_one_sided",
                        beta=100.0,
                        parameter=parameter,
                    )
                )
    return rows


def test_variant_is_resolved_from_metadata_not_run_name() -> None:
    manifest = {
        "arm_id": "conv_centered_eqprop_float32_float64_shadow",
        "configuration": {"resolved": {"eqprop_variant": "centered"}},
    }
    result = {"terminal_metrics": {"eqprop_variant": "centered"}}
    rows = [{"eqprop_variant": "centered"}]
    assert analysis._variant_from_metadata(manifest, result, rows) == "centered"


def test_parameter_schema_explicitly_allows_v2_positive_and_v3_centered(tmp_path) -> None:
    path = tmp_path / "parameter_precision_comparison.csv"
    v2 = [{"schema": "perfectdiode-conv-positive-eqprop-float64-shadow-audit/v2"}]
    assert (
        analysis._validate_parameter_schema(v2, variant="positive_one_sided", path=path)
        == "perfectdiode-conv-positive-eqprop-float64-shadow-audit/v2"
    )
    with pytest.raises(ValueError, match="cannot represent"):
        analysis._validate_parameter_schema(v2, variant="centered", path=path)
    v3 = [
        {
            "schema": "perfectdiode-conv-current-eqprop-true-dtype-audit/v3",
            "eqprop_variant": "centered",
        }
    ]
    assert analysis._validate_parameter_schema(v3, variant="centered", path=path).endswith(
        "/v3"
    )


def test_displacement_context_cannot_be_relabelled_from_another_tk() -> None:
    bundle = _centered_bundle()
    bad_rows = deepcopy(bundle["displacement_rows"])
    bad_rows[0]["T"] = "128"
    with pytest.raises(ValueError, match="Displacement T differs"):
        analysis._validate_displacement_parameter_context(
            bundle["parameter_rows"],
            bad_rows,
            variant="centered",
            path=Path("state_displacement.csv"),
        )


def test_centered_negative_displacement_requires_negative_beta_sign() -> None:
    bundle = _centered_bundle()
    bad_rows = deepcopy(bundle["displacement_rows"])
    negative = next(row for row in bad_rows if row["phase"] == "negative")
    negative["actual_beta"] = "300"
    with pytest.raises(ValueError, match="beta sign/magnitude mismatch"):
        analysis._validate_displacement_parameter_context(
            bundle["parameter_rows"],
            bad_rows,
            variant="centered",
            path=Path("state_displacement.csv"),
        )


def test_centered_normalization_keeps_negative_positive_and_span_separate() -> None:
    rows = analysis.normalize_bundle(_centered_bundle())
    c0_float32 = next(
        row
        for row in rows
        if row["parameter_name"] == "ConvWeight_0" and row["precision"] == "float32"
    )
    assert c0_float32["matched_zero_negative_relative_displacement"] == 0.2
    assert c0_float32["matched_zero_positive_relative_displacement"] == 0.3
    assert c0_float32["signed_span_present"] is True
    assert c0_float32["signed_span_relative_displacement"] == 0.5
    assert c0_float32["eqprop_vs_bptt_cosine"] == 0.7


def test_infer_tk_is_fail_closed_without_an_explicit_coordinate() -> None:
    rows = [{"T": 64, "K": 64}, {"T": 128, "K": 128}]
    with pytest.raises(ValueError, match="one T/K coordinate"):
        analysis._infer_tk(rows, None)
    assert analysis._infer_tk(rows, [128, 128]) == (128, 128)


def test_coverage_requires_paired_one_and_two_sided_beta_sets() -> None:
    rows = _complete_normalized_surface()
    analysis.validate_scientific_coverage(rows)
    broken = [
        row
        for row in deepcopy(rows)
        if not (
            row["source_kind"] == "primary"
            and row["scheme"] == "ours"
            and row["eqprop_variant"] == "centered"
        )
    ]
    with pytest.raises(ValueError, match="both EqProp variants"):
        analysis.validate_scientific_coverage(broken)


def test_merge_selects_same_tk_lower_beta_anchors_from_metadata() -> None:
    surface = _complete_normalized_surface()
    primary = [row for row in surface if row["source_kind"] == "primary"]
    anchors = [row for row in surface if row["source_kind"] == "anchor"]
    off_coordinate = deepcopy(anchors[0])
    off_coordinate["T"] = 128
    off_coordinate["K"] = 128
    anchors.append(off_coordinate)
    merged, duplicates = analysis.merge_rows(primary, anchors, tk=(64, 64))
    assert not duplicates
    assert all((row["T"], row["K"]) == (64, 64) for row in merged)
    assert {row["injected_beta"] for row in merged if row["source_kind"] == "anchor"} == {
        100.0
    }


def test_centered_anchor_beta_can_pair_with_prior_one_sided_bundle() -> None:
    rows = _complete_normalized_surface()
    extra_centered = []
    for scheme in analysis.SCHEMES:
        for precision in analysis.PRECISIONS:
            for parameter in analysis.PARAMETERS:
                extra_centered.append(
                    _normalized_row(
                        source_kind="primary",
                        scheme=scheme,
                        precision=precision,
                        variant="centered",
                        beta=100.0,
                        parameter=parameter,
                    )
                )
    analysis.validate_scientific_coverage([*rows, *extra_centered])
