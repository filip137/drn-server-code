from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import pytest
import torch

import experiments.mnist_conv.saturation_replay as saturation_replay
from experiments.mnist_conv.saturation_replay import (
    HardSigmoidLayerAccumulator,
    balanced_validation_positions,
    infer_rho_targets,
)


def test_hard_sigmoid_boundaries_are_inside_the_active_window() -> None:
    state = torch.tensor(
        [-5.0, -4.0, -3.0, 0.0, 3.0, 4.0, 5.0]
    ).reshape(1, 1, 1, 7)
    accumulator = HardSigmoidLayerAccumulator(
        v_off=4.0, layer_name="Hidden_0", layer_index=0
    )

    accumulator.update(state)
    layer, channels = accumulator.finalize()

    assert layer["low_count"] == 1
    assert layer["active_count"] == 5
    assert layer["high_count"] == 1
    assert layer["element_count"] == 7
    assert layer["saturation_fraction"] == pytest.approx(2 / 7)
    assert channels[0]["low_count"] == 1
    assert channels[0]["active_count"] == 5
    assert channels[0]["high_count"] == 1


def test_channel_axes_and_unequal_batch_merge_are_count_weighted() -> None:
    first = torch.tensor(
        [
            [
                [[-5.0, 0.0], [0.0, 0.0]],
                [[5.0, 5.0], [0.0, 0.0]],
            ],
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ],
        ]
    )
    second = torch.tensor(
        [
            [
                [[-5.0, -5.0], [-5.0, -5.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ]
        ]
    )
    merged = HardSigmoidLayerAccumulator(
        v_off=4.0, layer_name="Hidden_0", layer_index=0
    )
    merged.update(first)
    merged.update(second)
    concatenated = HardSigmoidLayerAccumulator(
        v_off=4.0, layer_name="Hidden_0", layer_index=0
    )
    concatenated.update(torch.cat([first, second], dim=0))

    layer_merged, channels_merged = merged.finalize()
    layer_concat, channels_concat = concatenated.finalize()

    for key in (
        "low_count",
        "active_count",
        "high_count",
        "element_count",
        "low_fraction",
        "active_fraction",
        "high_fraction",
        "saturation_fraction",
        "sample_saturation_p50",
        "sample_saturation_p90",
    ):
        assert layer_merged[key] == pytest.approx(layer_concat[key])
    assert channels_merged == pytest.approx(channels_concat)
    assert channels_merged[0]["low_count"] == 5
    assert channels_merged[1]["high_count"] == 2
    assert layer_merged["sample_count"] == 3


def test_accumulator_rejects_invalid_states_and_channel_drift() -> None:
    with pytest.raises(ValueError, match="finite positive"):
        HardSigmoidLayerAccumulator(
            v_off=0.0, layer_name="Hidden_0", layer_index=0
        )

    accumulator = HardSigmoidLayerAccumulator(
        v_off=4.0, layer_name="Hidden_0", layer_index=0
    )
    with pytest.raises(ValueError, match=r"\[B,C,H,W\]"):
        accumulator.update(torch.zeros(2, 3))
    with pytest.raises(ValueError, match="finite values"):
        accumulator.update(torch.full((1, 2, 1, 1), float("nan")))

    accumulator.update(torch.zeros(1, 2, 1, 1))
    with pytest.raises(ValueError, match="fixed hidden-channel count"):
        accumulator.update(torch.zeros(1, 3, 1, 1))


def test_infer_rho_targets_supports_v6_and_v5_provenance() -> None:
    direct = {
        "run": {
            "lr_provenance": {
                "rho_conv": 1e-2,
                "rho_dense": 3e-2,
            }
        }
    }
    relative = {
        "run": {
            "lr_provenance": {
                "alpha_arch": 1e-3,
                "target_multipliers_by_weight": {
                    "ConvWeight_0": 1.0,
                    "ConvWeight_1": 1.0,
                    "DenseWeight_0": 10.0,
                },
            }
        }
    }

    assert infer_rho_targets(direct) == pytest.approx((1e-2, 3e-2))
    assert infer_rho_targets(relative) == pytest.approx((1e-3, 1e-2))


def test_balanced_validation_positions_are_near_equal_and_round_robin() -> None:
    labels = tuple(label for label in range(10) for _ in range(20))

    positions = balanced_validation_positions(labels, sample_count=128)
    selected = [labels[position] for position in positions]

    assert len(positions) == 128
    assert selected[:10] == list(range(10))
    assert [selected.count(label) for label in range(10)] == [
        13,
        13,
        13,
        13,
        13,
        13,
        13,
        13,
        12,
        12,
    ]
    assert len(set(positions)) == len(positions)

    with pytest.raises(ValueError, match="at least one example per class"):
        balanced_validation_positions(labels, sample_count=9)


def test_replay_saturation_checks_hashes_reuses_cohort_and_labels_metrics(
    tmp_path, monkeypatch
) -> None:
    row = {
        "row_id": "conv1_ours_v4_c1",
        "architecture": "conv1",
        "scheme": "ours",
        "run_name": "mnist_bp_amp_v4_c1",
        "input_gain": 2.0,
        "voltage_amp": 4.0,
        "current_amp": 1.0,
        "inference_iterations": 4,
        "training_iterations": 4,
    }
    architecture = {
        "channels": [2],
        "kernel_sizes": [3],
        "strides": [2],
        "paddings": [1],
        "output_dim": 20,
        "pooling": "none",
    }
    affine = {
        "degrees": 0.0,
        "enabled": False,
        "fill": 0.0,
        "interpolation": "bilinear",
        "preset": "ordinary_identity",
        "scale": [1.0, 1.0],
        "seed": 1729,
        "shear": 0.0,
        "translate": [0.0, 0.0],
        "deterministic_by_original_index": True,
    }
    minimizer = {"adaptive_equilibrium": False, "settings": {"rel_tol": 1e-5}}
    study = {
        "schema_version": "mnist-conv-lr-study/v6",
        "dataset": {
            "variant": "ordinary",
            "name": "mnist",
            "normalization": {"mean": 0.1307, "std": 0.3081, "scale": 0.3},
            "affine": affine,
            "official_test": {"enabled": False, "read_allowed": False},
            "train": {"batch_size": 16, "shuffle_seed": 0},
            "validation": {
                "source": "mnist_train",
                "split_seed": 0,
                "size": 5000,
                "batch_size": 128,
            },
        },
        "model": {
            "architectures": {"conv1": architecture},
            "non_linearity": "hard_sigmoid",
            "quadratic_diode_param": {},
            "exponential_diode_param": {},
            "hard_sigmoid": {"g_off": 0.0, "g_on": 100.0, "v_off": 4.0},
            "conductance_bounds": [0.0, 100.0],
            "weight_initialization": "kaiming_uniform",
            "weight_gains": 1.0,
            "model_seed": 0,
        },
        "solver": {"energy_mode": "asynchronous", "minimizer": minimizer},
        "optimizer": {},
        "rows": [row],
    }
    study_path = tmp_path / "study.json"
    study_path.write_text(json.dumps(study))
    initialization_dir = tmp_path / "initialization"
    initialization_dir.mkdir()
    initialization_path = initialization_dir / "conv1.pt"
    initialization_path.write_bytes(b"init")
    entry = tmp_path / "entry"
    entry.mkdir()
    best_path = entry / "best_validation.pt"
    final_path = entry / "final.pt"
    best_path.write_bytes(b"best")
    final_path.write_bytes(b"final")
    run_spec = {
        "schema_version": "mnist-conv-run/v6",
        "seed": 0,
        "run": {
            "architecture": architecture,
            "dataset": {
                "name": "mnist",
                "normalization": study["dataset"]["normalization"],
                "affine": {
                    key: value
                    for key, value in affine.items()
                    if key != "deterministic_by_original_index"
                },
                "batch_size": 16,
                "validation": {
                    "source": "mnist_train",
                    "split_seed": 0,
                    "size": 5000,
                    "official_test_enabled": False,
                },
            },
            "initialization": {
                "checkpoint": {
                    "sha256": saturation_replay.sha256_file(initialization_path)
                }
            },
            "lr_provenance": {"rho_conv": 1e-2, "rho_dense": 1e-2},
            "model": {
                "non_linearity": "hard_sigmoid",
                "quadratic_diode_param": {},
                "exponential_diode_param": {},
                "hard_sigmoid_param": study["model"]["hard_sigmoid"],
                "input_gain": 2.0,
                "voltage_amp": 4.0,
                "current_amp": 1.0,
                "weight_min": 0.0,
                "weight_max": 100.0,
                "weight_init_mode": "kaiming_uniform",
                "weight_gains": [1.0, 1.0],
            },
            "solver": {
                "energy_mode": "asynchronous",
                "minimizer": minimizer,
                "inference_iterations": 4,
                "training_iterations": 4,
            },
        },
    }
    run_spec_path = entry / "run_spec.v6.json"
    run_spec_path.write_text(json.dumps(run_spec))
    summary = {
        "row": row,
        "rho_conv": 1e-2,
        "rho_dense": 1e-2,
        "best_validation_epoch": 5,
        "best_checkpoint_sha256": saturation_replay.sha256_file(best_path),
        "final_checkpoint_sha256": saturation_replay.sha256_file(final_path),
        "validation_indices_sha256": "split-hash",
        "validation_metrics": [
            {
                "epoch": 5,
                "sample_count": 5000,
                "loss": 0.1,
                "accuracy": 0.95,
            }
        ],
        "run_spec_v6_sha256": saturation_replay.sha256_file(run_spec_path),
    }
    (entry / "summary.json").write_text(json.dumps(summary))

    batches = [
        (
            torch.zeros(2, 1, 28, 28),
            torch.tensor([0, 1]),
            torch.tensor([10, 20]),
        ),
        (
            torch.zeros(2, 1, 28, 28),
            torch.tensor([2, 3]),
            torch.tensor([30, 40]),
        ),
    ]
    monkeypatch.setattr(
        saturation_replay,
        "build_loader_bundle",
        lambda *args, **kwargs: SimpleNamespace(
            validation_indices_hash="split-hash"
        ),
    )
    monkeypatch.setattr(
        saturation_replay,
        "_limited_validation_loader",
        lambda *args, **kwargs: (batches, (0, 1, 2, 3)),
    )
    built_iterations = []

    class FakeRuntime:
        def __init__(self, checkpoint_path, runtime_row):
            value = {
                "conv1.pt": 0.0,
                "best_validation.pt": 5.0,
                "final.pt": -5.0,
            }[checkpoint_path.name]
            self.device = torch.device("cpu")
            self.hidden = SimpleNamespace(state=None)
            self.parameters = []
            self.batch_size = 0
            self.network = SimpleNamespace(set_input=self.set_input)
            self.minimizer_inference = SimpleNamespace(
                compute_equilibrium=lambda: setattr(
                    self.hidden,
                    "state",
                    torch.full((self.batch_size, 2, 1, 1), value),
                )
            )
            self.energy_fn = SimpleNamespace(
                layers=lambda: [SimpleNamespace(), self.hidden, SimpleNamespace()]
            )
            self.cost_fn = SimpleNamespace(
                set_target=self.set_target,
                eval=lambda: torch.full((self.batch_size,), 0.25),
                error_fn=lambda: torch.zeros(self.batch_size, dtype=torch.bool),
            )
            built_iterations.append(int(runtime_row["inference_iterations"]))

        def set_input(self, images, reset):
            assert reset is True
            self.batch_size = int(images.shape[0])

        def set_target(self, labels):
            assert int(labels.shape[0]) == self.batch_size

    monkeypatch.setattr(
        saturation_replay,
        "build_model_runtime",
        lambda _study, runtime_row, *, initialization_checkpoint, **kwargs: FakeRuntime(
            initialization_checkpoint, runtime_row
        ),
    )
    monkeypatch.setattr(
        saturation_replay, "parameter_tensor_digest", lambda parameters: "digest"
    )
    monkeypatch.setattr(saturation_replay, "_plot_saturation", lambda *args: None)

    output = tmp_path / "output"
    result = saturation_replay.replay_saturation(
        study_path=study_path,
        entry_dirs=[entry],
        initialization_dir=initialization_dir,
        data_root=tmp_path,
        output_dir=output,
        device="cpu",
        batch_size=2,
        max_batches=2,
        reference_inference_iterations=64,
    )

    assert built_iterations == [64, 64, 64]
    assert result["cohort"]["source_indices"] == [10, 20, 30, 40]
    assert result["cohort"]["source_indices_sha256"]
    layers = {row["checkpoint_role"]: row for row in result["layers"]}
    assert layers["initialization"]["saturation_fraction"] == 0.0
    assert layers["best_validation"]["high_fraction"] == 1.0
    assert layers["final"]["low_fraction"] == 1.0
    with (output / "checkpoint_summary.csv").open(newline="") as handle:
        checkpoint_rows = list(csv.DictReader(handle))
    assert "cohort_accuracy" in checkpoint_rows[0]
    assert "source_operational_full_validation_accuracy" in checkpoint_rows[0]
    assert "accuracy" not in checkpoint_rows[0]
    assert (
        checkpoint_rows[1]["source_operational_full_validation_accuracy"] == "0.95"
    )
