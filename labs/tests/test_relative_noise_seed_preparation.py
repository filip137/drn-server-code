import copy
import json

import pytest

from experiments.prepare_relative_noise_seeds import ETAS, SCHEMES, make_config, source_path


@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("seed", [1, 2])
def test_only_declared_noise_and_metadata_change(depth, scheme, seed):
    original = json.loads(source_path(depth, scheme, seed).read_text())
    for tag, eta in ETAS:
        generated = make_config(depth, scheme, seed, tag, eta)
        for field in ("model_base", "model_overrides", "optimizer", "lr", "datasets",
                      "initialization", "init_checkpoint_path", "beta", "lab",
                      "runtime_dtype", "energy_minimizer", "parameter_order",
                      "learning_rates_by_parameter", "training_algorithm", "seed"):
            assert generated[field] == original[field], field
        expected_noise = copy.deepcopy(original["eqprop"])
        expected_noise.update(endpoint_read_noise_eta=eta, endpoint_read_noise_seed=2026081601 + seed)
        assert generated["eqprop"] == expected_noise
        assert generated["evaluation"]["official_test"]["policy"] == "disabled"
        assert generated["read_noise_seed_replication"]["checkpoint_recovery"] == "full_state_epoch_boundary_only"
        assert not generated["read_noise_seed_replication"]["optimizer_reset_at_epoch10"]
        assert "tests_and_canaries_skipped_by_user" not in generated["relative_noise_training"]


def test_all_72_cells_pair_weights_shuffle_and_noise():
    arms = []
    for depth in [1, 2, 3]:
        identities = []
        for seed in [1, 2]:
            cells = [make_config(depth, scheme, seed, tag, eta) for scheme in SCHEMES for tag, eta in ETAS]
            assert len({c["initialization"]["checkpoint_sha256"] for c in cells}) == 1
            assert {c["datasets"]["mnist"]["params"]["shuffle_seed"] for c in cells} == {seed}
            assert {c["eqprop"]["endpoint_read_noise_seed"] for c in cells} == {2026081601 + seed}
            identities.append(cells[0]["initialization"]["checkpoint_sha256"])
            arms.extend(c["arm_id"] for c in cells)
        assert len(set(identities)) == 2
    assert len(arms) == len(set(arms)) == 72
