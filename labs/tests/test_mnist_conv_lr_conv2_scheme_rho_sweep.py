from __future__ import annotations

from pathlib import Path

import pytest

from experiments.mnist_conv.io import atomic_write_bytes, atomic_write_json
from experiments.mnist_conv.lr_stages import (
    candidate_run_spec_v6,
    two_rho_learning_rates,
)
from experiments.mnist_conv.lr_study_spec import LRStudySpec
from experiments.run_mnist_conv_lr_conv2_scheme_rho_sweep import (
    RHO_CONV_GRID,
    RHO_DENSE_GRID,
    TRANSFER_REUSE_PAIR,
    V5_REUSE_BY_PAIR,
    scheme_entries,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
V5_CONFIG = (
    REPO_ROOT
    / "configs/conv/hardsigmoid_lr_architecture_relative_rho_constant_sgd_bs16_v5.json"
)
SLURM_WRAPPER = (
    REPO_ROOT
    / "experiments/run_mnist_conv_lr_conv2_scheme_rho_sweep_jeanzay.slurm"
)


class _FakeBundle:
    validation_indices_hash = "a" * 64

    def train_batch_indices(self, *, num_epochs: int = 1):
        return tuple(((epoch, epoch + 1),) for epoch in range(num_epochs))


def test_conv2_grid_reuses_four_cells_per_scheme() -> None:
    pairs = {
        (rho_conv, rho_dense)
        for rho_conv in RHO_CONV_GRID
        for rho_dense in RHO_DENSE_GRID
    }
    reused = set(V5_REUSE_BY_PAIR) | {TRANSFER_REUSE_PAIR}

    assert len(pairs) == 12
    assert reused <= pairs
    assert len(reused) == 4
    assert len(pairs - reused) == 8


def test_scheme_entries_are_disjoint_eight_run_packs() -> None:
    entries = []
    for scheme_index, scheme in enumerate(("ours", "legacy")):
        for local_index in range(12):
            entries.append(
                {
                    "entry_index": 12 * scheme_index + local_index,
                    "scheme": scheme,
                    "mode": (
                        "new_five_epoch_training" if local_index < 8 else "reuse"
                    ),
                }
            )
    manifest = {"entries": entries}

    ours = scheme_entries(manifest, "ours")
    legacy = scheme_entries(manifest, "legacy")

    assert len(ours) == len(legacy) == 8
    assert {entry["entry_index"] for entry in ours}.isdisjoint(
        {entry["entry_index"] for entry in legacy}
    )


def test_direct_conv2_run_spec_maps_two_conv_biases(tmp_path: Path) -> None:
    study = LRStudySpec.from_path(V5_CONFIG)
    row = next(item for item in study.rows if item["row_id"] == "conv2_ours_v4_c1")
    root = tmp_path / "study"
    atomic_write_bytes(root / "initialization/conv2.pt", b"checkpoint")
    atomic_write_json(
        root / "initialization/conv2.json",
        {"parameter_tensor_sha256": "b" * 64},
        canonical=True,
    )
    atomic_write_json(
        root / f"stages/probe/entries/{row['row_id']}/summary.json",
        {"probe": True},
        canonical=True,
    )
    units = {
        "ConvWeight_0": 0.2,
        "ConvWeight_1": 0.4,
        "DenseWeight_0": 1.0,
    }
    rates = two_rho_learning_rates(
        units,
        rho_conv=1.5e-3,
        rho_dense=1e-2,
        bias_weight_lr_groups={
            "ConvWeight_0": ["ConvWeight_0", "Bias_0"],
            "ConvWeight_1": ["ConvWeight_1", "Bias_1"],
            "DenseWeight_0": ["DenseWeight_0"],
        },
    )
    result = candidate_run_spec_v6(
        study.data,
        root,
        row,
        "scheme-rho--c01-d01",
        rho_conv=1.5e-3,
        rho_dense=1e-2,
        median_units_by_weight=units,
        learning_rates_by_parameter=rates,
        bundle=_FakeBundle(),
    )

    assert result.data["schema_version"] == "mnist-conv-run/v6"
    assert "conv2-amplified-scheme-two-rho" in result.data["protocol_id"]
    learning_rates = result.data["run"]["training"]["learning_rates_by_parameter"]
    assert learning_rates["Bias_0"] == pytest.approx(learning_rates["ConvWeight_0"])
    assert learning_rates["Bias_1"] == pytest.approx(learning_rates["ConvWeight_1"])
    assert learning_rates["ConvWeight_0"] != learning_rates["ConvWeight_1"]


def test_conv2_slurm_wrapper_requires_frozen_v6_source_authorization() -> None:
    source = SLURM_WRAPPER.read_text()

    assert "MNIST_CONV_V6_R3_AUTHORIZATION" in source
    assert (
        "AD010913993R3:fmu@v100:gpu_p13:qos_gpu-t3:v100-32g:gpu1"
        in source
    )
    assert '"source_v6_authorization": os.environ[' in source
