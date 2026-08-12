import csv
from pathlib import Path

import matplotlib.image as mpimg

from experiments import plot_conv3_eqprop_true_dtype_beta_curve as plotter


FIELDS = sorted(plotter.REQUIRED_COLUMNS)


def _write_complete_curve(path: Path) -> None:
    rows = []
    for scheme, betas in plotter.EXPECTED_BETAS.items():
        for beta in betas:
            for parameter in plotter.PARAMETERS:
                exact_zero = (
                    scheme == "legacy"
                    and beta == betas[0]
                    and parameter == "ConvWeight_0"
                )
                row = {field: "" for field in FIELDS}
                row.update(
                    {
                        "architecture": "conv3",
                        "scheme": scheme,
                        "checkpoint_role": "best_validation",
                        "T": 12 if scheme == "baseline" else 8,
                        "K": 8,
                        "batch_index": 0,
                        "batch_payload_sha256": "fixed-batch",
                        "parameter_name": parameter,
                        "bias_excluded": True,
                        "actual_beta": beta * 28.046421,
                        "effective_beta": min(
                            beta
                            * 28.046421
                            * {"baseline": 1, "ours": 64, "legacy": 4096}[scheme],
                            0.01,
                        ),
                        "beta_hat_requested": beta,
                        "beta_capped": (
                            (scheme == "ours" and beta == betas[-1])
                            or (scheme == "legacy" and beta == betas[-1])
                        ),
                        "residual_limited": scheme == "baseline",
                        "float32_eqprop_l2": 0.0 if exact_zero else 1.0,
                        "float64_eqprop_l2": 1.0,
                        "float32_bptt_l2": 1.0,
                        "float64_bptt_l2": 1.0,
                        "float32_exact_zero_fraction": 1.0 if exact_zero else 0.0,
                        "float64_exact_zero_fraction": 0.0,
                        "float32_eqprop_vs_bptt_cosine": "" if exact_zero else 0.4,
                        "float64_eqprop_vs_bptt_cosine": 0.999,
                        "float32_eqprop_over_bptt_norm_ratio": "" if exact_zero else 1.0,
                        "float64_eqprop_over_bptt_norm_ratio": 1.0,
                    }
                )
                rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_complete_curve_preserves_exact_zero_and_renders(tmp_path):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "curve.png"
    markdown_path = tmp_path / "curve.md"
    _write_complete_curve(input_path)

    rows = plotter.normalized_rows(plotter.load_curve_rows((input_path,)))

    assert len(rows) == 128
    assert sum(row["float32_status"] == "exact_zero" for row in rows) == 1
    assert sum(row["float64_status"] == "exact_zero" for row in rows) == 0
    plotter.render_plot(rows, output_path)
    plotter.write_markdown(rows, markdown_path)
    assert "ZERO" in markdown_path.read_text(encoding="utf-8")
    image = mpimg.imread(output_path)
    assert image.ndim == 3
    assert image.shape[0] > 100
    assert image.shape[1] > 100


def test_blank_nonzero_gradient_is_undefined_not_zero():
    row = {
        "float32_eqprop_vs_bptt_cosine": "",
        "float32_eqprop_l2": "2.0",
        "float32_exact_zero_fraction": "0.2",
    }
    assert plotter._status(row, "float32") == "undefined"
