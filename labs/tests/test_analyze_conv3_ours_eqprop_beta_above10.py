from __future__ import annotations

from copy import deepcopy

import pytest

from experiments import analyze_conv3_eqprop_one_vs_two_sided_higher_beta as common
from experiments import analyze_conv3_ours_eqprop_beta_above10 as analysis


def _complete_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for beta in analysis.EXPECTED_BETAS:
        for variant in common.VARIANTS:
            for precision in common.PRECISIONS:
                for parameter in common.PARAMETERS:
                    rows.append(
                        {
                            "scheme": "ours",
                            "T": 64,
                            "K": 64,
                            "injected_beta": beta,
                            "actual_base_beta": beta / analysis.AMPLIFICATION_FACTOR,
                            "eqprop_variant": variant,
                            "precision": precision,
                            "parameter_name": parameter,
                            "batch_payload_sha256": "batch",
                            "batch_source_indices_sha256": "indices",
                            "best_checkpoint_sha256": "checkpoint",
                            "source_selection_row_sha256": "selection",
                            "gradient_status": "numeric",
                            "bias_excluded": True,
                            "residual_limited": False,
                            "bptt_l2": 1.0,
                            "matched_zero_positive_relative_displacement": beta * 1.0e-6,
                        }
                    )
    return rows


def test_joined_rows_require_complete_paired_grid_and_exact_scaling() -> None:
    rows = _complete_rows()
    analysis._validate_rows(rows)

    incomplete = [
        row
        for row in deepcopy(rows)
        if not (
            row["injected_beta"] == 300.0
            and row["eqprop_variant"] == "centered"
            and row["precision"] == "float64"
        )
    ]
    with pytest.raises(ValueError, match="Incomplete beta/variant/precision"):
        analysis._validate_rows(incomplete)

    bad_scale = deepcopy(rows)
    bad_scale[0]["actual_base_beta"] = 3.0
    with pytest.raises(ValueError, match="beta scaling"):
        analysis._validate_rows(bad_scale)


def test_zero_and_positive_state_hashes_must_match_between_estimators() -> None:
    hashes = {
        (beta, variant, precision, phase): f"{beta}-{precision}-{phase}"
        for beta in analysis.EXPECTED_BETAS
        for variant in common.VARIANTS
        for precision in common.PRECISIONS
        for phase in ("zero", "positive")
    }
    analysis._validate_paired_endpoint_hashes(hashes)

    broken = dict(hashes)
    broken[(30.0, "centered", "float32", "positive")] = "different"
    with pytest.raises(ValueError, match="state hashes differ"):
        analysis._validate_paired_endpoint_hashes(broken)
