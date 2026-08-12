from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from experiments.analyze_paper_gradient_trace_epochs import (
    EXPECTED_ARM_IDS,
    _aggregate_rows,
    _batch_payload_sha256,
    _discover_runs,
    _gradient_metrics,
    _arm_identity,
    _load_checkpoint_inventory,
    _plot_run,
    _source_config_provenance,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_expected_surface_is_conv1_all_optimizers_and_deeper_sgd() -> None:
    assert len(EXPECTED_ARM_IDS) == 12
    assert len(set(EXPECTED_ARM_IDS)) == 12
    assert sum(arm.startswith("conv1_") for arm in EXPECTED_ARM_IDS) == 6
    assert all(
        "_sgd_" in arm
        for arm in EXPECTED_ARM_IDS
        if arm.startswith(("conv2_", "conv3_"))
    )


def test_discover_runs_ignores_smoke_and_returns_frozen_order(tmp_path: Path) -> None:
    for position, arm_id in enumerate(reversed(EXPECTED_ARM_IDS)):
        architecture = arm_id.split("_", 1)[0]
        run = tmp_path / "runs" / architecture / f"{position:03d}_{arm_id}"
        run.mkdir(parents=True)
        (run / "result.json").write_text(
            json.dumps({"completion": {"criteria_met": True}}),
            encoding="utf-8",
        )
        (run / "config.used.json").write_text(
            json.dumps({"arm_id": arm_id}), encoding="utf-8"
        )
        (run / "epoch_checkpoint_index.jsonl").write_text("\n", encoding="utf-8")

    smoke = tmp_path / "runs" / "conv1" / "smoke" / "000_smoke"
    smoke.mkdir(parents=True)
    (smoke / "result.json").write_text(
        json.dumps({"completion": {"criteria_met": True}}),
        encoding="utf-8",
    )
    (smoke / "config.used.json").write_text(
        json.dumps({"arm_id": EXPECTED_ARM_IDS[0]}), encoding="utf-8"
    )
    (smoke / "epoch_checkpoint_index.jsonl").write_text("\n", encoding="utf-8")

    runs = _discover_runs(tmp_path)
    discovered = [
        json.loads((run / "config.used.json").read_text(encoding="utf-8"))["arm_id"]
        for run in runs
    ]
    assert discovered == list(EXPECTED_ARM_IDS)


def test_checkpoint_inventory_verifies_paths_sizes_hashes_and_epochs(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    records = []
    for epoch in (0, 1):
        model = checkpoint_dir / f"epoch_{epoch:03d}_model.pt"
        optimizer = checkpoint_dir / f"epoch_{epoch:03d}_optimizer.pt"
        model.write_bytes(f"model-{epoch}".encode())
        optimizer.write_bytes(f"optimizer-{epoch}".encode())
        records.append(
            {
                "schema_version": "mnist-conv-diagnostic-epoch-checkpoint/v1",
                "epoch": epoch,
                "model_path": str(model.relative_to(tmp_path)),
                "model_size_bytes": model.stat().st_size,
                "model_sha256": _sha256(model),
                "optimizer_path": str(optimizer.relative_to(tmp_path)),
                "optimizer_size_bytes": optimizer.stat().st_size,
                "optimizer_sha256": _sha256(optimizer),
            }
        )
    (tmp_path / "epoch_checkpoint_index.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    inventory = _load_checkpoint_inventory(tmp_path, expected_epochs=(0, 1))
    assert [item["epoch"] for item in inventory] == [0, 1]
    assert inventory[1]["model_path"] == checkpoint_dir / "epoch_001_model.pt"

    (checkpoint_dir / "epoch_001_model.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Checkpoint size mismatch|SHA-256 mismatch"):
        _load_checkpoint_inventory(tmp_path, expected_epochs=(0, 1))


def test_source_config_provenance_allows_only_recorded_dataset_root_override(
    tmp_path: Path,
) -> None:
    original = {
        "arm_id": "conv1_baseline_sgd_seed0",
        "lab": {"dataset_key": "mnist"},
        "datasets": {"mnist": {"params": {"root": "/remote", "batch_size": 16}}},
    }
    runtime = json.loads(json.dumps(original))
    runtime["datasets"]["mnist"]["params"]["root"] = "/local"
    (tmp_path / "config.used.json").write_text(json.dumps(runtime), encoding="utf-8")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "configuration": {
                    "path": "/source/config.json",
                    "sha256": "abc",
                    "resolved": original,
                },
                "transport_overrides": {"dataset_root": "/local"},
            }
        ),
        encoding="utf-8",
    )

    provenance = _source_config_provenance(tmp_path, runtime)

    assert provenance["resolved_config_science_matches_manifest"] is True
    assert provenance["manifest_dataset_root"] == "/remote"
    assert provenance["runtime_dataset_root"] == "/local"

    runtime["datasets"]["mnist"]["params"]["batch_size"] = 32
    with pytest.raises(ValueError, match="differs scientifically"):
        _source_config_provenance(tmp_path, runtime)


def test_gradient_metrics_compare_vectors_only_to_same_batch_epoch_zero() -> None:
    initial = torch.tensor([3.0, 4.0, 0.0])
    current = torch.tensor([0.0, 5.0, 0.0])

    metrics = _gradient_metrics(current, initial)

    assert metrics["gradient_l2"] == pytest.approx(5.0)
    assert metrics["gradient_rms_relative_to_epoch0"] == pytest.approx(1.0)
    assert metrics["gradient_cosine_vs_epoch0_same_batch"] == pytest.approx(0.8)
    assert metrics[
        "gradient_relative_vector_change_vs_epoch0_same_batch"
    ] == pytest.approx((10.0**0.5) / 5.0)
    assert metrics["gradient_zero_fraction"] == pytest.approx(2.0 / 3.0)


def test_gradient_metrics_mark_zero_reference_direction_undefined() -> None:
    metrics = _gradient_metrics(torch.ones(4), torch.zeros(4))
    assert metrics["gradient_cosine_vs_epoch0_same_batch"] is None
    assert metrics["gradient_rms_relative_to_epoch0"] is None
    assert metrics["gradient_relative_vector_change_vs_epoch0_same_batch"] is None


def test_arm_identity_uses_canonical_optimizer_labels() -> None:
    adam = _arm_identity(
        {
            "arm_id": "conv1_baseline_adam_seed0",
            "optimizer": {"name": "Adam"},
        }
    )
    sgd = _arm_identity(
        {
            "arm_id": "conv2_ours_sgd_seed0",
            "optimizer": {"name": "SGD"},
        }
    )
    assert adam["optimizer"] == "Adam"
    assert sgd["optimizer"] == "SGD"


def test_batch_payload_hash_binds_images_labels_and_source_order() -> None:
    images = torch.arange(8, dtype=torch.float32).reshape(2, 1, 2, 2)
    labels = torch.tensor([1, 2])
    original = _batch_payload_sha256(images, labels, (10, 11))

    assert original == _batch_payload_sha256(images.clone(), labels.clone(), (10, 11))
    assert original != _batch_payload_sha256(images, labels, (11, 10))
    assert original != _batch_payload_sha256(images, torch.tensor([2, 1]), (10, 11))


def test_aggregate_rows_reports_median_and_spread(tmp_path: Path) -> None:
    common = {
        "architecture": "conv1",
        "scheme": "baseline",
        "optimizer": "SGD",
        "seed": 0,
        "arm_id": "conv1_baseline_sgd_seed0",
        "run_dir": "/run",
        "config_sha256": "c",
        "epoch": 1,
        "checkpoint_role": "epoch",
        "checkpoint_path": "/run/checkpoint.pt",
        "checkpoint_sha256": "d",
        "parameter_name": "ConvWeight_0",
        "parameter_type": "ConvWeight",
        "element_count": 2,
        "learning_rate": 0.1,
        "cohort_sha256": "e",
        "inference_iterations": 8,
        "gradient_iterations": 8,
        "parameter_rms": 1.0,
        "gradient_zero_fraction": 0.0,
        "gradient_exact_zero_fraction": 0.0,
        "gradient_rms_relative_to_epoch0": 1.0,
        "gradient_cosine_vs_epoch0_same_batch": 0.5,
        "gradient_relative_vector_change_vs_epoch0_same_batch": 0.5,
        "accuracy": 0.75,
    }
    rows = [
        {**common, "loss": 2.0, "gradient_l2": 2.0, "gradient_rms": 2.0},
        {**common, "loss": 4.0, "gradient_l2": 4.0, "gradient_rms": 4.0},
    ]

    summary = _aggregate_rows(rows)

    assert len(summary) == 1
    assert summary[0]["batch_count"] == 2
    assert summary[0]["loss_median"] == pytest.approx(3.0)
    assert summary[0]["gradient_rms_q10"] == pytest.approx(2.2)
    assert summary[0]["gradient_rms_q90"] == pytest.approx(3.8)

    plot = _plot_run(
        summary,
        arm_id="conv1_baseline_sgd_seed0",
        output_dir=tmp_path,
    )
    assert plot.is_file()
    assert plot.stat().st_size > 0
