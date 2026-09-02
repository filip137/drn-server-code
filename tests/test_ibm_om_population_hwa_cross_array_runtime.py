from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.artifacts import sha256_file
from experiments.mnist_relu_drn import ibm_om_population_hwa_cross_array_runtime as runtime
from experiments.mnist_relu_drn.ibm_om_population_hwa_cross_array_config import (
    HWA_CONDITION_KEY,
    HWA_ESTIMATOR_KEY,
)
from experiments.mnist_relu_drn.ibm_om_population_hwa_cross_array_runtime import (
    _array_comparisons,
    _load_hwa_inputs,
    _train_population_hwa_arm,
)
from training.ibm_reram_program_verify import PopulationStepEstimator
from training.ibm_reram_raw_active_program_verify import RAW_ACTIVE_COORDINATE


def _record(target: float) -> dict[str, object]:
    return {
        "target": target,
        "accepted_noncorrupt_residual": {
            "fit_count": 32,
            "bin_edges": [-0.02, 0.0, 0.02],
            "bin_probabilities": [0.5, 0.5],
        },
    }


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    nominal_dw_min = 0.0949
    trajectory_digest = "1" * 64
    population_digest = "2" * 64
    estimator = PopulationStepEstimator(
        bins=2,
        fallback_step=nominal_dw_min / 2.0,
    )
    estimator_path = tmp_path / "step_estimators.json"
    estimator_path.write_text(
        json.dumps(
            {
                "schema": "ebl.ibm_reram.step_estimators",
                "schema_version": 1,
                "calibration_partition_only": True,
                "metadata": {
                    "execution_profile": "hwa_production_cap128",
                    "preset": "reram_array_om",
                    "enable_published_corruption": False,
                    "nominal_dw_min": nominal_dw_min,
                    "coordinate": RAW_ACTIVE_COORDINATE,
                    "state_coordinate": "raw_active",
                    "trajectory_artifact_sha256": trajectory_digest,
                    "device_population_artifact_sha256": population_digest,
                },
                "estimators": {HWA_ESTIMATOR_KEY: estimator.to_mapping()},
            }
        ),
        encoding="utf-8",
    )
    endpoint_path = tmp_path / "bounded_uniform_model.json"
    endpoint_path.write_text(
        json.dumps(
            {
                "schema": (
                    "ebl.ibm_reram.bounded_piecewise_uniform_endpoint_model"
                ),
                "schema_version": 2,
                "coordinate": RAW_ACTIVE_COORDINATE,
                "metadata": {
                    "execution_profile": "hwa_production_cap128",
                    "preset": "reram_array_om",
                    "aihwkit_version": "1.1.0",
                    "enable_published_corruption": False,
                    "nominal_dw_min": nominal_dw_min,
                    "coordinate": RAW_ACTIVE_COORDINATE,
                    "state_coordinate": "raw_active",
                    "sampled_reference_consumed_by_plant": False,
                    "trajectory_artifact_sha256": trajectory_digest,
                    "device_population_artifact_sha256": population_digest,
                    "step_estimator_artifact_sha256": sha256_file(estimator_path),
                    "controller": {
                        "maximum_program_pulses": 128,
                        "adaptive": {
                            "eta": 0.75,
                            "maximum_batch": 32,
                            "epsilon": 1e-8,
                            "force_one_within_steps": 2.0,
                        },
                    },
                },
                "conditions": {
                    HWA_CONDITION_KEY: {
                        "fit_status": "fit",
                        "reachability_fit_status": "fit",
                        "adequate": True,
                        "validation": {
                            "per_target": [_record(0.0), _record(1.0)]
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return endpoint_path, estimator_path


def test_hwa_inputs_require_one_matched_healthy_raw_active_characterization(
    tmp_path: Path,
) -> None:
    endpoint_path, estimator_path = _write_inputs(tmp_path)

    (
        model,
        endpoint_digest,
        resolved_endpoint,
        estimator,
        estimator_digest,
        resolved_estimator,
        metadata,
    ) = _load_hwa_inputs(endpoint_path, estimator_path)

    assert model.coordinate == RAW_ACTIVE_COORDINATE
    assert endpoint_digest == sha256_file(endpoint_path)
    assert resolved_endpoint == endpoint_path.resolve()
    assert estimator.fallback_step == pytest.approx(0.04745)
    assert estimator_digest == sha256_file(estimator_path)
    assert resolved_estimator == estimator_path.resolve()
    assert metadata["sampled_reference_consumed_by_plant"] is False


def test_hwa_inputs_reject_reference_relative_estimator_metadata(
    tmp_path: Path,
) -> None:
    endpoint_path, estimator_path = _write_inputs(tmp_path)
    estimator = json.loads(estimator_path.read_text(encoding="utf-8"))
    estimator["metadata"]["coordinate"] = "x=(w+1)/2"
    estimator_path.write_text(json.dumps(estimator), encoding="utf-8")
    endpoint = json.loads(endpoint_path.read_text(encoding="utf-8"))
    endpoint["metadata"]["step_estimator_artifact_sha256"] = sha256_file(
        estimator_path
    )
    endpoint_path.write_text(json.dumps(endpoint), encoding="utf-8")

    with pytest.raises(ValueError, match="healthy raw-active calibration"):
        _load_hwa_inputs(endpoint_path, estimator_path)


def test_array_comparisons_separate_hwa_gain_from_corruption_penalty() -> None:
    def state(repaired: float, corrupt: float) -> dict[str, object]:
        return {
            "program_verify": {
                "counterfactual_repaired": {"mean_accuracy": repaired},
                "published_corrupt": {"mean_accuracy": corrupt},
            },
            "published_corruption_penalty_accuracy": repaired - corrupt,
        }

    result = _array_comparisons(
        {
            "raw_relu": state(0.80, 0.70),
            "population_hwa_continuous": state(0.90, 0.77),
            "population_hwa_one_delta_qat": state(0.88, 0.76),
        }
    )

    repaired = result["counterfactual_repaired"]
    assert repaired["continuous_hwa_minus_raw_relu"] == pytest.approx(0.10)
    assert repaired["one_delta_qat_minus_continuous_hwa"] == pytest.approx(
        -0.02
    )
    assert result["corruption_penalty"] == {
        "raw_relu": pytest.approx(0.10),
        "population_hwa_continuous": pytest.approx(0.13),
        "population_hwa_one_delta_qat": pytest.approx(0.12),
    }


def test_training_canary_uses_complete_positive_g_and_freezes_final_master(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint_path, estimator_path = _write_inputs(tmp_path)
    model, endpoint_digest, *_rest = _load_hwa_inputs(
        endpoint_path,
        estimator_path,
    )
    applied: list[tuple[torch.Tensor, ...]] = []
    state: dict[str, object] = {"batch": 0}

    def apply_full_g(_stack: object, values: object) -> None:
        selected = tuple(values)  # type: ignore[arg-type]
        assert len(selected) == 2
        assert all(bool(torch.all(value >= 0.0)) for value in selected)
        assert all(bool(torch.all(value <= 2.0)) for value in selected)
        applied.append(tuple(value.detach().clone() for value in selected))
        state["full"] = selected

    def evaluate_detailed(
        _stack: object,
        _teacher: object,
        loader: object,
        *,
        sample_limit: int | None,
    ) -> tuple[dict[str, float | int], torch.Tensor]:
        del loader, sample_limit
        return {
            "examples": 2,
            "student_accuracy": 0.5,
            "teacher_agreement": 0.5,
            "kl_teacher_student": 0.25,
        }, torch.tensor([0, 1])

    monkeypatch.setattr(runtime, "_apply_full_g", apply_full_g)
    monkeypatch.setattr(runtime, "_evaluate_detailed", evaluate_detailed)

    class Network:
        def set_input(self, inputs: torch.Tensor, *, reset: bool) -> None:
            assert reset
            state["batch"] = int(inputs.shape[0])

    class Minimizer:
        def compute_equilibrium(self) -> None:
            return None

    class Cost:
        def set_teacher(
            self,
            logits: torch.Tensor,
            labels: torch.Tensor,
        ) -> None:
            assert logits.shape[0] == labels.shape[0]

        def eval(self) -> torch.Tensor:
            return torch.ones(int(state["batch"]), dtype=torch.float32)

        def student_logits(self) -> torch.Tensor:
            return torch.zeros((int(state["batch"]), 2), dtype=torch.float32)

    class Differentiator:
        def compute_gradient(self) -> tuple[torch.Tensor, ...]:
            full = state["full"]
            assert isinstance(full, tuple)
            return tuple(torch.full_like(value, 0.01) for value in full)

    class Teacher:
        def logits(self, inputs: torch.Tensor) -> torch.Tensor:
            return torch.zeros((inputs.shape[0], 2), dtype=torch.float32)

    class Store:
        def __init__(self) -> None:
            self.records: list[dict[str, object]] = []

        def append_metric(self, value: dict[str, object]) -> None:
            self.records.append(value)

    stack = SimpleNamespace(
        device=torch.device("cpu"),
        network=Network(),
        minimizer=Minimizer(),
        cost=Cost(),
        differentiator=Differentiator(),
    )
    inputs = torch.zeros((2, 3), dtype=torch.float32)
    labels = torch.tensor([0, 1])
    loaders = SimpleNamespace(
        train=[(inputs, labels)],
        validation=[(inputs, labels)],
        test=[(inputs, labels)],
    )
    protocol = SimpleNamespace(
        evidence_tier="exploratory_noncanonical",
        hwa={
            "programming_error_seed": 9,
            "ramp_epochs": 2,
            "initial_strength": 0.0,
            "final_strength": 1.0,
            "learning_rates": [0.01, 0.01],
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
            "epochs": 2,
        },
        execution={
            "maximum_training_batches_per_epoch": 1,
            "evaluation_sample_limit": None,
        },
    )
    arm = SimpleNamespace(
        arm_id="population_hwa_continuous",
        role="programming_error_HWA_without_quantization",
        quantization_spacing_raw_x=None,
    )
    initial = (
        torch.tensor([[-0.75, 0.5]], dtype=torch.float32),
        torch.tensor([[0.25]], dtype=torch.float32),
    )
    store = Store()

    final, report, checkpoint = _train_population_hwa_arm(
        arm=arm,
        initial_master=initial,
        stack=stack,
        teacher=Teacher(),
        loaders=loaders,
        model=model,
        endpoint_digest=endpoint_digest,
        protocol=protocol,
        checkpoint_dir=tmp_path / "checkpoints",
        store=store,  # type: ignore[arg-type]
    )

    assert checkpoint.is_file()
    assert len(store.records) == 2
    assert report["fixed_final_epoch"] == 2
    assert report["sampler"]["draw_ordinal"] == 2
    assert len(applied) == 5
    assert all(value.device.type == "cpu" for value in final)
    assert all(bool(torch.all(value >= -1.0)) for value in final)
    assert all(bool(torch.all(value <= 1.0)) for value in final)
