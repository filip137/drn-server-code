from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from experiments.mnist_conv import lr_stages
from experiments.mnist_conv.identity import code_provenance
from experiments.mnist_conv.io import atomic_write_json
from experiments.mnist_conv.lr_stages import (
    candidate_run_spec_v7,
    execute_v7_core_selection,
    execute_v7_finalization,
    execute_v7_preflight_entry,
    two_rho_learning_rates,
)
from experiments.mnist_conv.lr_study import (
    _entries,
    create_study,
    publish_lr_stage_manifest,
    submit_slurm_stage,
)
from experiments.mnist_conv.lr_study_spec import LRStudySpec
from experiments.mnist_conv.lr_v7_routing import route_v7_candidate_entries


REPO_ROOT = Path(__file__).resolve().parents[2]
V7_CONFIG = (
    REPO_ROOT
    / "configs/conv/hardsigmoid_lr_conv3_scheme_two_rho_constant_sgd_bs16_v7.json"
)
V7_EXECUTOR = (
    REPO_ROOT / "configs/executors/jeanzay-v100-conv3-lr-v7.json"
)


def _probe_summary(scale: float = 1.0) -> dict[str, object]:
    return {
        "median_units_by_weight": {
            "ConvWeight_0": 0.1 * scale,
            "ConvWeight_1": 0.2 * scale,
            "ConvWeight_2": 0.4 * scale,
            "DenseWeight_0": 1.0 * scale,
        },
        "bias_weight_lr_groups": {
            "ConvWeight_0": ["ConvWeight_0", "Bias_0"],
            "ConvWeight_1": ["ConvWeight_1", "Bias_1"],
            "ConvWeight_2": ["ConvWeight_2", "Bias_2"],
            "DenseWeight_0": ["DenseWeight_0"],
        },
    }


def _write_probes(root: Path, study: LRStudySpec) -> None:
    for index, row in enumerate(study.rows, start=1):
        atomic_write_json(
            root
            / "stages/probe/entries"
            / row["row_id"]
            / "summary.json",
            _probe_summary(float(index)),
            canonical=True,
        )


def test_v7_core_manifest_has_27_independent_seven_rate_candidates(
    tmp_path: Path,
) -> None:
    study = LRStudySpec.from_path(V7_CONFIG)
    _write_probes(tmp_path, study)

    entries = _entries(study, tmp_path, "core_candidates")

    assert len(entries) == 27
    assert [entry["payload"]["scheme"] for entry in entries[::9]] == [
        "baseline",
        "ours",
        "legacy",
    ]
    for entry in entries:
        payload = entry["payload"]
        assert payload["candidate_stage"] == "core_grid"
        rates = payload["learning_rates_by_parameter"]
        assert set(rates) == {
            "ConvWeight_0",
            "Bias_0",
            "ConvWeight_1",
            "Bias_1",
            "ConvWeight_2",
            "Bias_2",
            "DenseWeight_0",
        }
        for index in range(3):
            assert rates[f"Bias_{index}"] == rates[f"ConvWeight_{index}"]
        assert "run_spec.v7.json" in {
            Path(path).name for path in entry["outputs"]
        }


def test_v7_extension_manifest_is_zero_work_or_only_new_cells(
    tmp_path: Path,
) -> None:
    study = LRStudySpec.from_path(V7_CONFIG)
    _write_probes(tmp_path, study)
    selection_path = (
        tmp_path / "stages/select_core/entries/selection/selection.json"
    )
    atomic_write_json(
        selection_path,
        {
            "rows": [
                {
                    "row_id": row["row_id"],
                    "expansion_axes": [],
                }
                for row in study.rows
            ]
        },
        canonical=True,
    )
    zero = _entries(study, tmp_path, "extension_candidates")
    assert len(zero) == 1
    assert zero[0]["payload"]["no_op"] is True

    atomic_write_json(
        selection_path,
        {
            "rows": [
                {
                    "row_id": study.rows[0]["row_id"],
                    "expansion_axes": ["rho_conv"],
                },
                {
                    "row_id": study.rows[1]["row_id"],
                    "expansion_axes": ["rho_dense"],
                },
                {
                    "row_id": study.rows[2]["row_id"],
                    "expansion_axes": ["rho_conv", "rho_dense"],
                },
            ]
        },
        canonical=True,
    )
    extension = _entries(study, tmp_path, "extension_candidates")
    assert len(extension) == 3 + 3 + 7
    assert len({entry["entry_id"] for entry in extension}) == len(extension)
    assert all(
        entry["payload"]["candidate_stage"] == "extension_grid"
        for entry in extension
    )
    assert all(
        entry["payload"]["rho_conv"] == 0.03
        or entry["payload"]["rho_dense"] == 0.1
        for entry in extension
    )

    atomic_write_json(
        selection_path,
        {
            "rows": [
                {
                    "row_id": row["row_id"],
                    "expansion_axes": ["rho_conv", "rho_dense"],
                }
                for row in study.rows
            ]
        },
        canonical=True,
    )
    capped = _entries(study, tmp_path, "extension_candidates")
    assert len(capped) == 21
    assert 27 + len(capped) == 48


class _FakeBundle:
    validation_indices_hash = "a" * 64

    def train_batch_indices(self, *, num_epochs: int = 1):
        return tuple(((epoch,),) for epoch in range(num_epochs))


def test_v7_run_spec_freezes_three_epochs_validation64_and_conv3_vector(
    tmp_path: Path,
) -> None:
    study = LRStudySpec.from_path(V7_CONFIG)
    row = study.rows[0]
    root = (
        tmp_path
        / "results/lr_studies"
        / f"{study.data['name']}--{study.study_id}"
    )
    checkpoint = root / "initialization/conv3.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"conv3-v7-checkpoint")
    atomic_write_json(
        root / "initialization/conv3.json",
        {"parameter_tensor_sha256": "b" * 64},
        canonical=True,
    )
    atomic_write_json(
        root / "stages/probe/entries" / row["row_id"] / "summary.json",
        _probe_summary(),
        canonical=True,
    )
    probe = _probe_summary()
    rates = two_rho_learning_rates(
        probe["median_units_by_weight"],
        rho_conv=0.003,
        rho_dense=0.01,
        bias_weight_lr_groups=probe["bias_weight_lr_groups"],
    )

    run = candidate_run_spec_v7(
        study.data,
        root,
        row,
        "core-c01-d01",
        candidate_stage="core_grid",
        rho_conv=0.003,
        rho_dense=0.01,
        median_units_by_weight=probe["median_units_by_weight"],
        learning_rates_by_parameter=rates,
        bundle=_FakeBundle(),
    ).data

    assert run["schema_version"] == "mnist-conv-run/v7"
    assert run["run"]["dataset"]["validation"]["batch_size"] == 64
    assert run["run"]["training"]["epochs"] == 3
    assert run["run"]["training"]["schedule"]["total_steps"] == 10314
    assert len(run["run"]["training"]["learning_rates_by_parameter"]) == 7
    provenance = run["run"]["lr_provenance"]
    assert provenance["gain_calibration_dataset"] == (
        "deterministic_medium_affine_mnist"
    )
    assert provenance["optimization_dataset"] == "ordinary_mnist"
    assert provenance["official_test_read"] is False


def test_v7_preflight_rejects_noncanonical_inputs_before_training(
    tmp_path: Path,
) -> None:
    study = LRStudySpec.from_path(V7_CONFIG)
    payload = {
        "row_id": "conv3_ours_v4_c1",
        "rho_conv": 0.003,
        "rho_dense": 0.01,
    }

    with pytest.raises(ValueError, match="frozen v7 preflight row"):
        execute_v7_preflight_entry(
            study.data,
            tmp_path,
            candidate_payload={**payload, "rho_dense": 0.03},
            data_root=tmp_path / "mnist",
            download=False,
            device="cuda",
        )
    with pytest.raises(ValueError, match="already materialized MNIST"):
        execute_v7_preflight_entry(
            study.data,
            tmp_path,
            candidate_payload=payload,
            data_root=tmp_path / "mnist",
            download=True,
            device="cuda",
        )
    with pytest.raises(RuntimeError, match="V100 CUDA device"):
        execute_v7_preflight_entry(
            study.data,
            tmp_path,
            candidate_payload=payload,
            data_root=tmp_path / "mnist",
            download=False,
            device="cpu",
        )


def _candidate_summary(
    *,
    row: dict[str, object],
    rho_conv: float,
    rho_dense: float,
    loss: float,
    accuracy: float,
) -> dict[str, object]:
    weights = {
        "ConvWeight_0": 0.01,
        "ConvWeight_1": 0.02,
        "ConvWeight_2": 0.03,
        "DenseWeight_0": 0.04,
    }
    return {
        "row": row,
        "admissible": True,
        "inadmissible_reason": None,
        "final_validation_loss": loss,
        "final_validation_accuracy": accuracy,
        "median_projection_efficiency": 0.9,
        "learning_rates_by_weight": weights,
        "learning_rates_by_parameter": {
            "ConvWeight_0": 0.01,
            "Bias_0": 0.01,
            "ConvWeight_1": 0.02,
            "Bias_1": 0.02,
            "ConvWeight_2": 0.03,
            "Bias_2": 0.03,
            "DenseWeight_0": 0.04,
        },
        "observed_rho_relative_q90_complete_run_by_parameter": {
            name: rho_conv if name.startswith("Conv") else rho_dense
            for name in weights
        },
        "observed_rho_span_q90_complete_run_by_parameter": {
            name: 1e-4 for name in weights
        },
        "maximum_bound_occupancy_by_parameter": {
            name: 0.1 for name in weights
        },
        "median_projection_efficiency_by_parameter": {
            name: 0.9 for name in weights
        },
    }


def test_v7_core_selection_is_independent_and_requests_only_needed_axis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = LRStudySpec.from_path(V7_CONFIG)
    entries: list[dict[str, object]] = []
    core = study.data["rho_grid"]["core"]
    for row in study.rows:
        for conv_index, rho_conv in enumerate(core["rho_conv"]):
            for dense_index, rho_dense in enumerate(core["rho_dense"]):
                entry_id = (
                    f"{row['row_id']}--core-c{conv_index:02d}-d{dense_index:02d}"
                )
                entries.append(
                    {
                        "entry_id": entry_id,
                        "payload": {
                            "row_id": row["row_id"],
                            "scheme": row["scheme"],
                            "rho_conv": rho_conv,
                            "rho_dense": rho_dense,
                        },
                    }
                )
                if row["scheme"] == "baseline":
                    best = (rho_conv, rho_dense) == (0.003, 0.01)
                    accuracy = 0.90
                elif row["scheme"] == "ours":
                    best = (rho_conv, rho_dense) == (0.01, 0.01)
                    accuracy = 0.90
                else:
                    best = False
                    accuracy = 0.899
                atomic_write_json(
                    tmp_path
                    / "stages/core_candidates/entries"
                    / entry_id
                    / "summary.json",
                    _candidate_summary(
                        row=row,
                        rho_conv=rho_conv,
                        rho_dense=rho_dense,
                        loss=0.1 if best else 1.0,
                        accuracy=accuracy,
                    ),
                    canonical=True,
                )
    monkeypatch.setattr(
        lr_stages,
        "load_stage_manifest",
        lambda *_args, **_kwargs: {"entries": entries},
    )
    monkeypatch.setattr(
        lr_stages, "_plot_v7_final_metrics", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        lr_stages,
        "_plot_v7_parameter_heatmaps",
        lambda *_args, **_kwargs: None,
    )

    result = execute_v7_core_selection(study.data, tmp_path)

    by_scheme = {row["scheme"]: row for row in result["rows"]}
    assert by_scheme["baseline"]["status"] == "selected_core"
    assert by_scheme["baseline"]["selected_rho_conv"] == 0.003
    assert by_scheme["ours"]["status"] == "expansion_required"
    assert by_scheme["ours"]["expansion_axes"] == ["rho_conv"]
    assert by_scheme["legacy"]["status"] == "unresolved"
    assert by_scheme["legacy"]["reason"] == "no_passing_candidate"


def test_v7_routing_checks_all_local_targets_uses_trex_then_r3() -> None:
    availability = {
        "main": {"checked": True, "reachable": True, "free_gpu_slots": 1},
        "akibscomputer": {
            "checked": True,
            "reachable": True,
            "free_gpu_slots": 1,
        },
        "trex": {"checked": True, "reachable": True, "free_gpu_slots": 2},
    }

    result = route_v7_candidate_entries(
        [f"candidate-{index}" for index in range(5)],
        local_availability=availability,
    )

    assert result["processes_per_gpu"] == 1
    assert result["routes"][0] == {
        "target": "trex",
        "entry_ids": ["candidate-0", "candidate-1"],
        "entry_count": 2,
    }
    assert result["routes"][1]["target"] == "jean_zay_r3"
    assert result["routes"][1]["account"] == "fmu@v100"
    assert result["routes"][1]["entry_count"] == 3

    with pytest.raises(ValueError, match="memory and throughput"):
        route_v7_candidate_entries(
            ["candidate-0"],
            local_availability=availability,
            packing_benchmark={
                "processes_per_gpu": 2,
                "memory_passed": True,
                "throughput_passed": False,
            },
        )


def test_v7_slurm_dry_run_freezes_r3_v100_worker_contract(
    tmp_path: Path,
) -> None:
    study = LRStudySpec.from_path(V7_CONFIG)
    root, _contract = create_study(study, tmp_path / "results")
    _manifest, manifest_path = publish_lr_stage_manifest(
        root,
        study,
        "audit",
        code_provenance(REPO_ROOT),
    )

    result = submit_slurm_stage(
        study_dir=root,
        manifest_path=manifest_path,
        data_root=tmp_path / "mnist",
        device="cuda",
        profile_path=V7_EXECUTOR,
        dry_run=True,
    )

    worker = result["worker"]
    joined = "\n".join(worker)
    assert result["submitted"] is False
    assert "--account=fmu@v100" in worker
    assert "--partition=gpu_p13" in worker
    assert "--qos=qos_gpu-t3" in worker
    assert "--constraint=v100" in worker
    assert "--gres=gpu:1" in worker
    assert "--hint=nomultithread" in worker
    assert "--array=0-0%27" in worker
    assert "MNIST_CONV_R3_ALLOCATION=AD010913993R3" in joined
    assert "MNIST_CONV_EXPECTED_SLURM_CONSTRAINT=v100" in joined
    assert "MNIST_CONV_THREADS_PER_PROCESS=1" in joined
    assert "OMP_NUM_THREADS=1" in joined
    assert "MKL_NUM_THREADS=1" in joined
    assert "OPENBLAS_NUM_THREADS=1" in joined
    assert "NUMEXPR_NUM_THREADS=1" in joined
    wrapper = (
        REPO_ROOT / "experiments/run_mnist_conv_lr_stage_slurm.sh"
    ).read_text()
    assert "--format=Constraints" in wrapper
    assert "export OMP_NUM_THREADS=1" in wrapper
    assert "export MKL_NUM_THREADS=1" in wrapper
    assert "export OPENBLAS_NUM_THREADS=1" in wrapper
    assert "export NUMEXPR_NUM_THREADS=1" in wrapper


def test_v7_slurm_wrapper_resolves_an_unset_constraint_from_sacct(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_sacct = fake_bin / "sacct"
    fake_sacct.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'v100|\\n'\n",
        encoding="utf-8",
    )
    fake_sacct.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "MNIST_CONV_REPO_ROOT": str(REPO_ROOT),
        "MNIST_CONV_MODULE": "",
        "MNIST_CONV_IDRENV_PROJECT": "",
        "MNIST_CONV_R3_ALLOCATION": "AD010913993R3",
        "MNIST_CONV_THREADS_PER_PROCESS": "1",
        "MNIST_CONV_EXPECTED_SLURM_ACCOUNT": "fmu@v100",
        "MNIST_CONV_EXPECTED_SLURM_PARTITION": "gpu_p13",
        "MNIST_CONV_EXPECTED_SLURM_QOS": "qos_gpu-t3",
        "MNIST_CONV_EXPECTED_SLURM_CONSTRAINT": "v100",
        "SLURM_JOB_ACCOUNT": "fmu@v100",
        "SLURM_JOB_PARTITION": "gpu_p13",
        "SLURM_JOB_QOS": "qos_gpu-t3",
        "SLURM_JOB_ID": "123",
        "SLURM_ARRAY_TASK_ID": "0",
        "MNIST_CONV_PYTHON": "/bin/true",
        "MNIST_CONV_LR_STAGE": "audit",
        "MNIST_CONV_LR_STUDY": str(tmp_path / "study"),
        "MNIST_CONV_LR_MANIFEST": str(tmp_path / "manifest.json"),
        "MNIST_CONV_DATASET_ROOT": str(tmp_path / "data"),
        "MNIST_CONV_DEVICE": "cuda",
    }
    environment.pop("SLURM_JOB_CONSTRAINTS", None)

    completed = subprocess.run(
        [str(REPO_ROOT / "experiments/run_mnist_conv_lr_stage_slurm.sh")],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def _v7_record(
    *,
    stage: str,
    entry_id: str,
    row: dict[str, object],
    rho_conv: float,
    rho_dense: float,
    loss: float,
    accuracy: float = 0.91,
) -> dict[str, object]:
    summary = _candidate_summary(
        row=row,
        rho_conv=rho_conv,
        rho_dense=rho_dense,
        loss=loss,
        accuracy=accuracy,
    )
    return {
        "stage": stage,
        "entry_id": entry_id,
        "row_id": row["row_id"],
        "scheme": row["scheme"],
        "rho_conv": rho_conv,
        "rho_dense": rho_dense,
        "admissible": summary["admissible"],
        "inadmissible_reason": summary["inadmissible_reason"],
        "final_validation_loss": summary["final_validation_loss"],
        "final_validation_accuracy": summary["final_validation_accuracy"],
        "median_projection_efficiency": summary[
            "median_projection_efficiency"
        ],
        "learning_rates_by_weight": summary["learning_rates_by_weight"],
        "learning_rates_by_parameter": summary[
            "learning_rates_by_parameter"
        ],
        "achieved_relative_updates_by_weight": summary[
            "observed_rho_relative_q90_complete_run_by_parameter"
        ],
        "achieved_span_updates_by_weight": summary[
            "observed_rho_span_q90_complete_run_by_parameter"
        ],
        "maximum_bound_occupancy_by_parameter": summary[
            "maximum_bound_occupancy_by_parameter"
        ],
        "median_projection_efficiency_by_parameter": summary[
            "median_projection_efficiency_by_parameter"
        ],
    }


def _write_selected_candidate_artifacts(
    root: Path,
    record: dict[str, object],
) -> None:
    candidate_dir = (
        root
        / "stages"
        / str(record["stage"])
        / "entries"
        / str(record["entry_id"])
    )
    row_id = str(record["row_id"])
    atomic_write_json(
        candidate_dir / "summary.json",
        {
            "validation_metrics": [
                {
                    "epoch": 3,
                    "loss": record["final_validation_loss"],
                    "accuracy": record["final_validation_accuracy"],
                }
            ],
            "inadmissible_reason": None,
            "safety_gate_failure": None,
            "accuracy_gate_passed": True,
            "checkpoint_sha256": "a" * 64,
            "initial_parameter_tensor_sha256": "b" * 64,
            "validation_indices_sha256": "c" * 64,
            "minibatch_order_sha256": "d" * 64,
        },
        canonical=True,
    )
    atomic_write_json(
        candidate_dir / "run_spec.v7.json",
        {"schema_version": "mnist-conv-run/v7", "entry_id": record["entry_id"]},
        canonical=True,
    )
    atomic_write_json(
        root / "stages/probe/entries" / row_id / "summary.json",
        {"row_id": row_id, "median_units_by_weight": {}},
        canonical=True,
    )


def test_v7_finalization_freezes_partial_rows_and_stops_after_one_expansion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = LRStudySpec.from_path(V7_CONFIG)
    atomic_write_json(
        tmp_path / "study.resolved.json", study.data, canonical=True
    )
    core = study.data["rho_grid"]["core"]
    baseline, ours, legacy = study.rows
    core_records: list[dict[str, object]] = []
    for row in study.rows:
        for conv_index, rho_conv in enumerate(core["rho_conv"]):
            for dense_index, rho_dense in enumerate(core["rho_dense"]):
                if row["scheme"] == "baseline":
                    best = (rho_conv, rho_dense) == (0.003, 0.01)
                elif row["scheme"] == "ours":
                    best = (rho_conv, rho_dense) == (0.01, 0.01)
                else:
                    best = (rho_conv, rho_dense) == (0.003, 0.03)
                core_records.append(
                    _v7_record(
                        stage="core_candidates",
                        entry_id=(
                            f"{row['row_id']}--core-"
                            f"c{conv_index:02d}-d{dense_index:02d}"
                        ),
                        row=row,
                        rho_conv=float(rho_conv),
                        rho_dense=float(rho_dense),
                        loss=0.1 if best else 1.0,
                    )
                )

    extension_records: list[dict[str, object]] = []
    for dense_index, rho_dense in enumerate(core["rho_dense"]):
        extension_records.append(
            _v7_record(
                stage="extension_candidates",
                entry_id=f"{ours['row_id']}--extension-c03-d{dense_index:02d}",
                row=ours,
                rho_conv=0.03,
                rho_dense=float(rho_dense),
                loss=1.0,
            )
        )
    for conv_index, rho_conv in enumerate(core["rho_conv"]):
        extension_records.append(
            _v7_record(
                stage="extension_candidates",
                entry_id=f"{legacy['row_id']}--extension-c{conv_index:02d}-d03",
                row=legacy,
                rho_conv=float(rho_conv),
                rho_dense=0.1,
                loss=0.05 if rho_conv == 0.003 else 1.0,
            )
        )

    baseline_id = f"{baseline['row_id']}--core-c01-d01"
    ours_id = f"{ours['row_id']}--core-c02-d01"
    atomic_write_json(
        tmp_path / "stages/select_core/entries/selection/selection.json",
        {
            "rows": [
                {
                    "row_id": baseline["row_id"],
                    "status": "selected_core",
                    "reason": None,
                    "expansion_axes": [],
                    "selected_entry_id": baseline_id,
                },
                {
                    "row_id": ours["row_id"],
                    "status": "expansion_required",
                    "reason": "passing_plateau_confined_to_outer_boundary",
                    "expansion_axes": ["rho_conv"],
                    "selected_entry_id": None,
                },
                {
                    "row_id": legacy["row_id"],
                    "status": "expansion_required",
                    "reason": "passing_plateau_confined_to_outer_boundary",
                    "expansion_axes": ["rho_dense"],
                    "selected_entry_id": None,
                },
            ]
        },
        canonical=True,
    )

    by_stage = {
        "core_candidates": core_records,
        "extension_candidates": extension_records,
    }
    monkeypatch.setattr(
        lr_stages,
        "_v7_candidate_records",
        lambda _root, *, stage, row_id=None: [
            record
            for record in by_stage[stage]
            if row_id is None or record["row_id"] == row_id
        ],
    )
    selected_records = {
        baseline_id: next(
            record
            for record in core_records
            if record["entry_id"] == baseline_id
        ),
        ours_id: next(
            record
            for record in core_records
            if record["entry_id"] == ours_id
        ),
    }
    for record in selected_records.values():
        _write_selected_candidate_artifacts(tmp_path, record)

    result = execute_v7_finalization(study.data, tmp_path)

    rows = {row["scheme"]: row for row in result["rows"]}
    assert result["status"] == "partial_seed0_ordinary_mnist_layerwise_screen"
    assert result["frozen_row_count"] == 2
    assert result["unresolved_row_count"] == 1
    assert result["candidate_training_run_count"] == 33
    assert result["candidate_epochs"] == 99
    assert rows["baseline"]["status"] == (
        "frozen_seed0_ordinary_mnist_layerwise_screen"
    )
    assert rows["ours"]["status"] == (
        "frozen_seed0_ordinary_mnist_layerwise_screen"
    )
    assert rows["ours"]["selected_from_stage"] == "core_candidates"
    assert rows["ours"]["selection_stage"] == "post_expansion"
    assert rows["ours"]["expansion_axes"] == ["rho_conv"]
    assert rows["ours"]["selection_status"] == "selected"
    assert rows["legacy"]["status"] == "unresolved"
    assert rows["legacy"]["reason"] == "post_expansion_plateau_on_boundary"
    assert rows["legacy"]["remaining_upper_boundary_axes"] == ["rho_dense"]
    assert len(rows["baseline"]["four_weight_learning_rates"]) == 4
    assert len(rows["baseline"]["seven_parameter_learning_rate_vector"]) == 7
    assert all(
        len(value) == 64
        for value in rows["baseline"]["provenance_hashes"].values()
    )
    assert result["ordinary_mnist_only"] is True
    assert result["medium_affine_conv3_handoff_replaced"] is False
    assert result["heatmap_scope"] == "per_scheme_final_adaptive_grid"
    output_dir = tmp_path / "stages/finalize/entries/finalization"
    for filename in (
        "epoch3_accuracy_loss_heatmap.png",
        "raw_learning_rates_heatmap.png",
        "achieved_relative_updates_heatmap.png",
        "occupancy_heatmap.png",
        "projection_efficiency_heatmap.png",
    ):
        assert (output_dir / filename).stat().st_size > 0


def test_v7_finalization_can_publish_zero_selected_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = LRStudySpec.from_path(V7_CONFIG)
    atomic_write_json(
        tmp_path / "study.resolved.json", study.data, canonical=True
    )
    atomic_write_json(
        tmp_path / "stages/select_core/entries/selection/selection.json",
        {
            "rows": [
                {
                    "row_id": row["row_id"],
                    "status": "unresolved",
                    "reason": "no_passing_candidate",
                    "expansion_axes": [],
                    "selected_entry_id": None,
                    "plateau_entry_ids": [],
                    "diagnostic_best_entry_id": None,
                    "minimum_final_validation_loss": None,
                }
                for row in study.rows
            ]
        },
        canonical=True,
    )
    core = study.data["rho_grid"]["core"]
    records = [
        _v7_record(
            stage="core_candidates",
            entry_id=(
                f"{row['row_id']}--core-c{conv_index:02d}-d{dense_index:02d}"
            ),
            row=row,
            rho_conv=float(rho_conv),
            rho_dense=float(rho_dense),
            loss=1.0,
            accuracy=0.899,
        )
        for row in study.rows
        for conv_index, rho_conv in enumerate(core["rho_conv"])
        for dense_index, rho_dense in enumerate(core["rho_dense"])
    ]
    monkeypatch.setattr(
        lr_stages,
        "_v7_candidate_records",
        lambda _root, *, stage, row_id=None: (
            [
                record
                for record in records
                if row_id is None or record["row_id"] == row_id
            ]
            if stage == "core_candidates"
            else []
        ),
    )
    monkeypatch.setattr(
        lr_stages, "_plot_v7_final_metrics", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        lr_stages,
        "_plot_v7_parameter_heatmaps",
        lambda *_args, **_kwargs: None,
    )

    result = execute_v7_finalization(study.data, tmp_path)

    assert result["status"] == "unresolved"
    assert result["frozen_row_count"] == 0
    assert result["unresolved_row_count"] == 3
    assert result["partial_handoff"] is False
    assert result["candidate_training_run_count"] == 27
    assert result["candidate_epochs"] == 81
    assert all(row["provenance_hashes"] is None for row in result["rows"])
    assert len(result["study_contract_sha256"]) == 64
    assert len(result["core_selection_sha256"]) == 64
