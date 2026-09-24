import pytest

from experiments.summarize_paper_training_completion import COMPLETE, summarize_group


def group():
    return [{"block": "T3_EP", "architecture": "conv1", "scheme": "ours", "G_max": ".001",
             "algorithm": "EP", "optimizer": "Adam", "epochs": "10", "T": "4", "K": "4",
             "G_min": ".00001", "voltage_amp": "4", "current_amp": "1", "split_seed": "0",
             "model_seed": str(seed), "shuffle_seed": str(seed), "training_status": COMPLETE,
             "official_test_read": "false", "best_validation_accuracy_percent": str(90 + 2 * seed),
             "final_validation_accuracy_percent": str(80 + seed)} for seed in range(3)]


def test_uses_all_three_seeds_and_sample_standard_deviation():
    summary = summarize_group(group())
    assert summary["mean_best_validation_percent"] == 92
    assert summary["sample_sd_best_validation_pp"] == 2
    assert summary["mean_final_validation_percent"] == 81
    assert summary["sample_sd_final_validation_pp"] == 1


def test_partial_group_keeps_individual_scores_without_aggregate():
    rows = group()
    rows[2].update(training_status="running", best_validation_accuracy_percent="",
                   final_validation_accuracy_percent="")
    summary = summarize_group(rows)
    assert summary["collected_count"] == 2 and summary["collected_seeds"] == "0,1"
    assert summary["mean_best_validation_percent"] is None
    assert summary["sample_sd_best_validation_pp"] is None
    assert summary["seed0_best_validation_percent"] == 90
    assert summary["seed2_best_validation_percent"] is None


def test_poor_replication_is_retained_in_the_three_seed_estimate():
    rows = group()
    rows[0].update(best_validation_accuracy_percent="10", final_validation_accuracy_percent="5")
    assert summarize_group(rows)["mean_best_validation_percent"] == pytest.approx((10 + 92 + 94) / 3)


@pytest.mark.parametrize("field,value,match", [
    ("model_seed", "1", "exactly once"),
    ("shuffle_seed", "2", "shuffle"),
    ("T", "8", "mixed scientific"),
    ("official_test_read", "true", "validation-only"),
    ("best_validation_accuracy_percent", "nan", "Invalid validation"),
])
def test_rejects_invalid_group_evidence(field, value, match):
    rows = group()
    rows[0][field] = value
    with pytest.raises(ValueError, match=match):
        summarize_group(rows)
