from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import experiments.evaluate_conv_weight_write_noise as evaluator


def test_noise_cases_deduplicate_and_evaluate_clean_case_once():
    assert evaluator.noise_cases(
        [0.0, 0.0, 0.01, 0.01],
        [7, 7, 9],
    ) == [
        (0.0, 7),
        (0.01, 7),
        (0.01, 9),
    ]


@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf")])
def test_noise_cases_reject_invalid_std_fraction(value):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        evaluator.noise_cases([value], [0])


def test_noise_cases_require_programming_seed():
    with pytest.raises(ValueError, match="programming seed"):
        evaluator.noise_cases([0.0], [])


def test_noise_cases_require_clean_reference_and_valid_seeds():
    with pytest.raises(ValueError, match="eta=0"):
        evaluator.noise_cases([0.01], [0])
    with pytest.raises(ValueError, match="Programming seeds"):
        evaluator.noise_cases([0.0, 0.01], [-1])


def test_noise_cases_put_clean_reference_first():
    assert evaluator.noise_cases([0.01, 0.0], [2, 3]) == [
        (0.0, 2),
        (0.01, 2),
        (0.01, 3),
    ]


def test_fraction_label_is_path_safe_and_distinguishes_scales():
    labels = {
        evaluator._fraction_label(value)
        for value in (0.0, 0.0025, 0.01, 0.1)
    }
    assert len(labels) == 4
    assert all(set(label) <= set("abcdefghijklmnopqrstuvwxyz0123456789_") for label in labels)


def _fake_source_identity(checkpoint_hash="a" * 64):
    return {
        "manifest": {"arm_id": "conv1-baseline"},
        "run_dir": Path("/tmp/source-run"),
        "hashes": {
            "checkpoint": checkpoint_hash,
            "config": "b" * 64,
            "manifest": "c" * 64,
            "metrics": "d" * 64,
            "result": "e" * 64,
        },
    }


def test_case_identity_separates_smoke_full_and_source_content():
    source = _fake_source_identity()
    _, smoke = evaluator._case_identity(
        source,
        0.01,
        3,
        max_validation_batches=1,
        evaluation_device="cpu",
    )
    _, full = evaluator._case_identity(
        source,
        0.01,
        3,
        max_validation_batches=None,
        evaluation_device="cpu",
    )
    _, changed_source = evaluator._case_identity(
        _fake_source_identity("f" * 64),
        0.01,
        3,
        max_validation_batches=1,
        evaluation_device="cpu",
    )
    _, changed_device = evaluator._case_identity(
        source,
        0.01,
        3,
        max_validation_batches=1,
        evaluation_device="cuda",
    )

    assert len({smoke, full, changed_source, changed_device}) == 4
    assert "smoke_1_validation_batches" in smoke
    assert "full_validation" in full


def test_resume_requires_exact_case_contract():
    requested = {"configuration": {"case_contract_sha256": "a" * 64}}
    evaluator._require_matching_case_contract(requested, requested)
    with pytest.raises(RuntimeError, match="does not match"):
        evaluator._require_matching_case_contract(
            {"configuration": {"case_contract_sha256": "b" * 64}},
            requested,
        )


def test_active_source_scope_rejects_smoke_affine_and_nonperfect_diode():
    result = {"smoke": False}
    dataset = {
        "key": "mnist",
        "variant": "ordinary",
        "factory": "labs.datasets.MnistTrainValidationDataset",
        "evaluation_split": "validation",
    }
    model = {
        "non_linearity": "perfect_diode",
        "conv_pipeline": [{"mode": "convolution"}],
    }
    evaluator._validate_active_source_scope(
        result=result,
        source_dataset=dataset,
        model_config=model,
    )

    with pytest.raises(ValueError, match="non-smoke"):
        evaluator._validate_active_source_scope(
            result={"smoke": True},
            source_dataset=dataset,
            model_config=model,
        )
    with pytest.raises(ValueError, match="ordinary-MNIST"):
        evaluator._validate_active_source_scope(
            result=result,
            source_dataset={**dataset, "variant": "deterministic_medium_affine"},
            model_config=model,
        )
    with pytest.raises(ValueError, match="perfect_diode"):
        evaluator._validate_active_source_scope(
            result=result,
            source_dataset=dataset,
            model_config={**model, "non_linearity": "relu"},
        )


def test_paired_output_scores_subtract_negative_terminal():
    output = torch.tensor(
        [
            [
                1.0,
                0.25,
                2.0,
                0.5,
                3.0,
                0.75,
                4.0,
                1.0,
                5.0,
                1.25,
                6.0,
                1.5,
                7.0,
                1.75,
                8.0,
                2.0,
                9.0,
                2.25,
                10.0,
                2.5,
            ]
        ]
    )
    assert torch.equal(
        evaluator._scores(output),
        torch.tensor([[0.75, 1.5, 2.25, 3.0, 3.75, 4.5, 5.25, 6.0, 6.75, 7.5]]),
    )


def test_projected_kkt_residual_uses_one_sided_gradient_at_diode_rails():
    layer = SimpleNamespace(
        state=torch.tensor([[0.0, 2.0, 0.0, -3.0]]),
    )
    output = SimpleNamespace(state=torch.empty(1, 2))
    gradient = torch.tensor([[-2.0, -3.0, 4.0, 5.0]])

    residual = evaluator._perfect_diode_layer_residual(
        layer,
        gradient,
        output_layer=output,
        epsilon=1.0e-8,
    )

    # Positive rail at zero admits only -gradient; negative rail at zero admits
    # only +gradient. Interior nodes use the absolute gradient.
    assert torch.equal(residual, torch.tensor([5.0]))


def test_validation_replay_keeps_autograd_enabled_for_energy_residuals():
    output = SimpleNamespace(
        name="OutputLayer_0",
        state=torch.tensor(
            [[2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            dtype=torch.float64,
            requires_grad=True,
        ),
    )

    class _Network:
        def set_input(self, images, reset):
            assert reset is True

    class _Minimizer:
        def compute_equilibrium(self):
            return None

    class _Cost:
        def set_target(self, labels):
            self.labels = labels

        def eval(self):
            return output.state[:, 0].square()

        def error_fn(self):
            return torch.zeros(1, dtype=torch.bool)

    class _Energy:
        def grad_layer_fn(self, layer):
            def gradient():
                value = layer.state.square().sum()
                return torch.autograd.grad(value, layer.state)[0]

            return gradient

    runtime = {
        "validation_loader": [
            (
                torch.ones((1, 1), dtype=torch.float64),
                torch.zeros(1, dtype=torch.long),
            )
        ],
        "base_states": [torch.zeros(1, dtype=torch.float64)],
        "input_dtype": torch.float64,
        "network": _Network(),
        "minimizer": _Minimizer(),
        "cost_fn": _Cost(),
        "energy_fn": _Energy(),
        "free_layers": [output],
        "output_layer": output,
        "inference_iterations": 4,
    }

    metrics = evaluator._evaluate_validation(runtime, max_batches=1)

    assert metrics["examples"] == 1
    assert metrics["accuracy"] == 1.0
    assert metrics["projected_kkt_residual_by_layer"]["OutputLayer_0"][
        "maximum"
    ] == pytest.approx(4.0)


def test_nonfinite_validation_is_retained_as_scientific_outcome(monkeypatch):
    def fail_validation(runtime, *, max_batches):
        raise FloatingPointError("non-finite equilibrium state")

    monkeypatch.setattr(evaluator, "_evaluate_validation", fail_validation)
    outcome = evaluator._evaluate_validation_outcome(
        {"inference_iterations": 8},
        max_batches=None,
    )

    assert outcome == {
        "outcome": "nonfinite",
        "split": "validation",
        "official_test_read": False,
        "inference_iterations": 8,
        "max_validation_batches": None,
        "failure": {
            "type": "FloatingPointError",
            "message": "non-finite equilibrium state",
        },
    }


def test_clean_reference_gate_requires_finite_and_exact_full_replay():
    source = {"source_best_validation_accuracy": 0.9}
    finite = {"outcome": "finite", "accuracy": 0.9}
    evaluator._verify_clean_reference(
        source,
        finite,
        max_validation_batches=None,
    )
    # A smoke has different coverage and therefore checks finiteness only.
    evaluator._verify_clean_reference(
        source,
        {"outcome": "finite", "accuracy": 0.5},
        max_validation_batches=1,
    )
    with pytest.raises(RuntimeError, match="did not produce finite"):
        evaluator._verify_clean_reference(
            source,
            {"outcome": "nonfinite"},
            max_validation_batches=1,
        )
    with pytest.raises(RuntimeError, match="does not reproduce"):
        evaluator._verify_clean_reference(
            source,
            {"outcome": "finite", "accuracy": 0.89},
            max_validation_batches=None,
        )


def test_dry_run_does_not_create_output_directory(tmp_path: Path, monkeypatch):
    source_path = tmp_path / "source"
    output_root = tmp_path / "must-not-exist"
    fake_source = {
        "labels": {
            "architecture": "conv1",
            "scheme": "baseline",
            "optimizer": "Adam",
            "weight_contract": "wide_0_100",
            "training_algorithm": "EP",
        },
        "run_dir": source_path.resolve(),
        "source_checkpoint_sha256": "a" * 64,
        "manifest": {"arm_id": "conv1-baseline"},
        "hashes": _fake_source_identity()["hashes"],
    }
    monkeypatch.setattr(evaluator, "_source_inventory", lambda path: fake_source)

    assert (
        evaluator.main(
            [
                "--source-run",
                str(source_path),
                "--output-root",
                str(output_root),
                "--std-fractions",
                "0",
                "0.01",
                "--programming-seeds",
                "4",
                "5",
                "--dry-run",
            ]
        )
        == 0
    )
    assert not output_root.exists()
