import csv
from pathlib import Path

import matplotlib.image as mpimg

from experiments import plot_eqprop_true_dtype_cosine as plotter


FIELDS = [
    "architecture",
    "scheme",
    "checkpoint_role",
    "parameter_name",
    "beta_hat_requested",
    "residual_limited",
    "float32_eqprop_l2",
    "float64_eqprop_l2",
    "float32_exact_zero_fraction",
    "float64_exact_zero_fraction",
    "float32_eqprop_vs_bptt_cosine",
    "float64_eqprop_vs_bptt_cosine",
    "float32_eqprop_over_bptt_norm_ratio",
    "float64_eqprop_over_bptt_norm_ratio",
]


def _write_evidence(path: Path, beta_hat: float, *, inject_missing: bool) -> None:
    rows = []
    for architecture in plotter.ARCHITECTURES:
        for scheme in plotter.SCHEMES:
            for parameter_name in plotter.PARAMETER_ORDER[architecture]:
                row = {
                    "architecture": architecture,
                    "scheme": scheme,
                    "checkpoint_role": "best_validation",
                    "parameter_name": parameter_name,
                    "beta_hat_requested": beta_hat,
                    "residual_limited": architecture == "conv3"
                    and scheme == "baseline",
                    "float32_eqprop_l2": 1.0,
                    "float64_eqprop_l2": 1.0,
                    "float32_exact_zero_fraction": 0.0,
                    "float64_exact_zero_fraction": 0.0,
                    "float32_eqprop_vs_bptt_cosine": 0.5,
                    "float64_eqprop_vs_bptt_cosine": 0.999,
                    "float32_eqprop_over_bptt_norm_ratio": 1.2,
                    "float64_eqprop_over_bptt_norm_ratio": 1.0,
                }
                if inject_missing and (
                    architecture,
                    scheme,
                    parameter_name,
                ) == ("conv2", "baseline", "ConvWeight_0"):
                    row["float32_eqprop_vs_bptt_cosine"] = ""
                    row["float32_eqprop_l2"] = 0.0
                    row["float32_exact_zero_fraction"] = 1.0
                if inject_missing and (
                    architecture,
                    scheme,
                    parameter_name,
                ) == ("conv3", "ours", "ConvWeight_1"):
                    row["float64_eqprop_vs_bptt_cosine"] = ""
                rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_render_preserves_exact_zero_and_other_undefined_cosines(tmp_path):
    low_path = tmp_path / "low.csv"
    high_path = tmp_path / "high.csv"
    output_path = tmp_path / "comparison.png"
    _write_evidence(low_path, 1.0e-5, inject_missing=True)
    _write_evidence(high_path, 1.0e-3, inject_missing=False)

    summary = plotter.render_true_dtype_figure(
        (low_path, high_path), output_path
    )

    assert summary == {
        "row_count": 21,
        "numeric_cell_count": 82,
        "exact_zero_cell_count": 1,
        "undefined_cell_count": 1,
        "residual_limited_case_count": 1,
    }
    assert output_path.stat().st_size > 0
    image = mpimg.imread(output_path)
    assert image.ndim == 3
    assert image.shape[0] > 100
    assert image.shape[1] > 100


def test_cell_classification_does_not_treat_blank_as_zero():
    row = {
        "float32_eqprop_vs_bptt_cosine": "",
        "float32_eqprop_over_bptt_norm_ratio": "",
        "float32_eqprop_l2": "2.0",
        "float32_exact_zero_fraction": "0.5",
    }
    cell = plotter.cell_value(row, "float32")
    assert cell.cosine is None
    assert cell.status == "undefined"
