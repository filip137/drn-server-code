from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from experiments import analyze_conv3_centered_float64_post_square_noise as analysis


def test_unsigned_endpoint_quantizer_uses_shared_range_and_can_erase_difference() -> None:
    q_plus = np.asarray([1000.1], dtype=np.float64)
    q_minus = np.asarray([1000.0], dtype=np.float64)
    full_scale = 1.05 * float(q_plus.max())

    quantized_plus, step_plus = analysis._quantize_unsigned(q_plus, 8, full_scale)
    quantized_minus, step_minus = analysis._quantize_unsigned(q_minus, 8, full_scale)

    assert step_plus == step_minus
    assert np.array_equal(quantized_plus, quantized_minus)
    assert not np.array_equal(q_plus, q_minus)


def test_signed_midtread_quantizer_preserves_zero() -> None:
    values = np.asarray([-1.0, -0.2, 0.0, 0.2, 1.0], dtype=np.float64)

    quantized, step = analysis._quantize_signed_midtread(values, 4, 1.05)

    assert step == pytest.approx(1.05 / 7.0)
    assert quantized[2] == 0.0
    assert quantized[0] == pytest.approx(-1.05)
    assert quantized[-1] == pytest.approx(1.05)


def test_sustained_bits_rejects_isolated_pass() -> None:
    flags = {bit: bit >= 10 for bit in analysis.BITS}
    flags[8] = True

    threshold = analysis._sustained_minimum_bits(flags)

    assert threshold["sustained_minimum_bits"] == 10
    assert threshold["lower_failing_bits"] == 9
    assert threshold["pass_to_fail_reversal_count"] == 1


def _write_synthetic_summary(path: Path) -> None:
    beta_grid = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
    largest = {
        ("baseline", analysis.ROLES[0]): 100.0,
        ("baseline", analysis.ROLES[1]): 100.0,
        ("ours", analysis.ROLES[0]): 10.0,
        ("ours", analysis.ROLES[1]): 30.0,
        ("legacy", analysis.ROLES[0]): 0.03,
        ("legacy", analysis.ROLES[1]): 0.01,
    }
    rows = []
    factors = {"baseline": 1.0, "ours": 64.0, "legacy": 4096.0}
    for scheme in analysis.SCHEMES:
        for role in analysis.ROLES:
            for beta in beta_grid:
                passing = beta <= largest[(scheme, role)]
                for parameter in analysis.PARAMETERS:
                    rows.append(
                        {
                            "eqprop_variant": "centered",
                            "precision": "float64",
                            "scheme": scheme,
                            "checkpoint_role": role,
                            "parameter_name": parameter,
                            "injected_beta": beta,
                            "actual_base_beta": beta / factors[scheme],
                            "amplification_factor": factors[scheme],
                            "run_id": f"B-{beta}",
                            "cosine_defined": True,
                            "eqprop_vs_bptt_cosine": 0.999 if passing else 0.98,
                            "eqprop_vs_bptt_symmetric_norm_delta": 0.01 if passing else 0.2,
                        }
                    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_clean_beta_selection_labels_open_and_bracketed_edges(tmp_path: Path) -> None:
    source = tmp_path / "summary.csv"
    _write_synthetic_summary(source)

    selected, _ = analysis.select_clean_contexts(source)
    by_context = {(row["scheme"], row["checkpoint_role"]): row for row in selected}

    assert by_context[("baseline", analysis.ROLES[0])]["selected_injected_beta"] == 100.0
    assert by_context[("baseline", analysis.ROLES[0])]["selection_boundary"] == "open_upper_tested_edge"
    assert by_context[("ours", analysis.ROLES[0])]["selected_injected_beta"] == 10.0
    assert by_context[("ours", analysis.ROLES[0])]["next_tested_beta"] == 30.0
    assert by_context[("ours", analysis.ROLES[0])]["selection_boundary"] == "bracketed_by_next_failing_beta"


def test_fixed_seed_gaussian_projection_is_reproducible_and_degrades() -> None:
    clean = np.asarray([1.0, -2.0, 3.0, -4.0], dtype=np.float64)
    bptt = clean.copy()

    left = analysis._gaussian_quantiles(clean, bptt, common_q_rms=100.0, injected_beta=1.0, seed=17)
    right = analysis._gaussian_quantiles(clean, bptt, common_q_rms=100.0, injected_beta=1.0, seed=17)

    assert left == right
    assert left[0]["combined_gate_passed"] is True
    assert left[-1]["combined_gate_passed"] is False
    threshold = analysis._leading_noise_threshold(left)
    assert threshold["threshold_resolved"] is True
    assert threshold["maximum_sustained_sigma_over_q_common_rms"] < 0.1


def test_interaction_normalization_contract_includes_legacy_amplification() -> None:
    assert analysis.ALPHA["legacy"] == {
        "C0": 1.0,
        "C1": 1.0 / 16.0,
        "C2": 1.0 / 256.0,
        "Dense": 1.0 / 4096.0,
    }
    scale = 2.0 / (analysis.ALPHA["legacy"]["Dense"] * analysis.SPATIAL_COUNT["Dense"])
    assert scale == 8192.0
