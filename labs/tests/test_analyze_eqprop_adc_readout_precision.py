from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from experiments.analyze_eqprop_adc_readout_precision import (
    _gradient_metrics,
    _minimum_passing_rows,
    _quantize_symmetric_midtread,
    _refinement_bits,
    _resolve_beta_rows,
    _write_source_snapshots,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
STUDY_CONFIG = (
    REPOSITORY_ROOT
    / "configs/conv/perfectdiode_conv3_legacy_eqprop_adc_readout_precision_20260811_v1.json"
)


def test_symmetric_midtread_quantizer_preserves_exact_zero() -> None:
    values = torch.tensor([-0.0, 0.0, -0.25, 0.25], dtype=torch.float64)

    quantized, codes, metadata = _quantize_symmetric_midtread(
        values,
        bits=4,
        full_scale=1.0,
    )

    assert torch.equal(quantized[:2], torch.zeros(2, dtype=torch.float64))
    assert torch.equal(codes[:2], torch.zeros_like(codes[:2]))
    assert metadata["step"] == pytest.approx(1.0 / 7.0)


def test_source_snapshot_retains_exact_bytes(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_bytes(b"print('frozen')\n")

    [row] = _write_source_snapshots(tmp_path, {"test": source})

    snapshot = tmp_path / row["snapshot_path"]
    assert snapshot.read_bytes() == source.read_bytes()
    assert row["snapshot_matches_source"] is True
    assert row["snapshot_sha256"] == row["source_sha256"]


def test_shared_endpoint_adc_loses_a_sub_lsb_state_difference() -> None:
    zero_endpoint = torch.tensor([0.20], dtype=torch.float64)
    positive_endpoint = torch.tensor([0.21], dtype=torch.float64)

    quantized_zero, zero_codes, _ = _quantize_symmetric_midtread(
        zero_endpoint,
        bits=4,
        full_scale=1.0,
    )
    quantized_positive, positive_codes, _ = _quantize_symmetric_midtread(
        positive_endpoint,
        bits=4,
        full_scale=1.0,
    )

    assert not torch.equal(positive_endpoint, zero_endpoint)
    assert torch.equal(positive_codes, zero_codes)
    assert torch.equal(quantized_positive - quantized_zero, torch.zeros_like(zero_endpoint))


def test_symmetric_midtread_quantizer_reports_step_and_clipping() -> None:
    values = torch.tensor([-2.0, -1.0, -0.4, 0.0, 0.4, 1.0, 2.0])

    quantized, codes, metadata = _quantize_symmetric_midtread(
        values,
        bits=4,
        full_scale=1.0,
    )

    assert metadata["full_scale"] == pytest.approx(1.0)
    assert metadata["step"] == pytest.approx(1.0 / 7.0)
    assert metadata["clipping_fraction"] == pytest.approx(2.0 / 7.0)
    assert quantized[0].item() == pytest.approx(-1.0)
    assert quantized[-1].item() == pytest.approx(1.0)
    assert codes[0].item() == -7
    assert codes[-1].item() == 7


def test_identical_gradient_is_an_exact_fidelity_gate_pass() -> None:
    reference = torch.tensor([3.0, -4.0, 0.0], dtype=torch.float64)

    metrics = _gradient_metrics(reference.clone(), reference)

    assert metrics["cosine"] == pytest.approx(1.0)
    assert metrics["relative_l2"] == pytest.approx(0.0)
    assert metrics["norm_ratio"] == pytest.approx(1.0)
    assert metrics["symmetric_norm_delta"] == pytest.approx(0.0)
    assert metrics["exact_zero_fraction"] == pytest.approx(1.0 / 3.0)
    assert metrics["gate_passed"] is True


def test_refinement_bits_fill_the_first_fail_to_pass_transition() -> None:
    coarse_bits = [8, 12, 16, 20]
    pass_by_bit = {8: False, 12: False, 16: True, 20: True}

    assert _refinement_bits(coarse_bits, pass_by_bit) == [13, 14, 15, 17, 18, 19]
    assert _refinement_bits(coarse_bits, {bit: False for bit in coarse_bits}) == []


def test_refinement_bits_extend_to_model_floor_when_lowest_coarse_bit_passes() -> None:
    coarse_bits = [4, 6, 8]

    assert _refinement_bits(
        coarse_bits, {4: True, 6: True, 8: True}
    ) == [2, 3, 5, 7]


def test_minimum_bits_requires_a_sustained_passing_region() -> None:
    rows = [
        {
            "acquisition_model": "absolute_endpoint_shared_adc",
            "actual_beta": 1.0e-6,
            "effective_beta": 1.0e-3,
            "beta_hat_base_requested": 1.0e-9,
            "adc_bits": bit,
            "all_weight_layers_and_minibatches_passed": passed,
            "minimum_ideal_eqprop_vs_bptt_cosine": 0.999,
        }
        for bit, passed in ((40, False), (41, True), (42, False), (43, True), (44, True))
    ]

    [summary] = _minimum_passing_rows(rows)

    assert summary["first_isolated_passing_adc_bits"] == 41
    assert summary["minimum_passing_adc_bits"] == 43
    assert summary["lower_failing_adc_bits"] == 42
    assert summary["pass_to_fail_reversal_count"] == 1
    assert summary["gate_resolved"] is True


def test_minimum_bits_keeps_an_untested_lower_edge_open() -> None:
    rows = [
        {
            "acquisition_model": "analog_delta_ideal_common",
            "actual_beta": 1.0e-6,
            "effective_beta": 1.0e-3,
            "beta_hat_base_requested": 1.0e-9,
            "adc_bits": bit,
            "all_weight_layers_and_minibatches_passed": True,
            "minimum_ideal_eqprop_vs_bptt_cosine": 0.999,
        }
        for bit in (4, 6, 8)
    ]

    [summary] = _minimum_passing_rows(rows)

    assert summary["minimum_passing_adc_bits"] is None
    assert summary["passing_upper_bound_adc_bits"] == 4
    assert summary["threshold_interpretation"] == "at_or_below_tested_minimum"
    assert summary["gate_resolved"] is False


def test_resolve_conv3_legacy_requested_base_beta_hats_from_source_config() -> None:
    study_config = json.loads(STUDY_CONFIG.read_text(encoding="utf-8"))
    source_config_path = REPOSITORY_ROOT / study_config["source"]["beta_config"]
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    source_case = next(
        case
        for case in source_config["cases"]
        if case["architecture"] == "conv3" and case["scheme"] == "legacy"
    )
    requested_hats = study_config["case"]["base_beta_hat_requested"]

    rows = _resolve_beta_rows(source_config, source_case, requested_hats)

    assert [row["beta_hat_requested"] for row in rows] == pytest.approx(
        [3.0e-10, 1.0e-9, 3.0e-9]
    )
    assert [row["beta"] for row in rows] == pytest.approx(
        [8.4139263e-9, 2.8046421e-8, 8.4139263e-8]
    )
    assert [row["beta_effective"] for row in rows] == pytest.approx(
        [3.44634421248e-5, 1.14878191616e-4, 3.44634421248e-4]
    )
    assert all(row["amplification_factor"] == pytest.approx(4096.0) for row in rows)
    assert all(row["capped"] is False for row in rows)
