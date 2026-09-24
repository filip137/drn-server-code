"""A recorded data path override must not permit changes to the training contract."""
import copy

from experiments.collect_eqprop_read_noise import config_matches_recorded_transport


def test_dataset_override_requires_its_recorded_path():
    expected = {"lab": {"dataset_key": "mnist"},
                "datasets": {"mnist": {"params": {"root": "/source/mnist", "split_seed": 0}}},
                "beta": .001, "lr": [.01, .002]}
    actual = copy.deepcopy(expected)
    actual["datasets"]["mnist"]["params"]["root"] = "/akib/mnist"
    assert config_matches_recorded_transport(actual, expected, "/akib/mnist")
    assert not config_matches_recorded_transport(actual, expected, None)
    assert not config_matches_recorded_transport(actual, expected, "/different/mnist")
    assert expected["datasets"]["mnist"]["params"]["root"] == "/source/mnist"


def test_dataset_override_does_not_allow_a_changed_beta_or_cohort():
    expected = {"lab": {"dataset_key": "mnist"},
                "datasets": {"mnist": {"params": {"root": "/source/mnist", "split_seed": 0}}},
                "beta": .001}
    actual = copy.deepcopy(expected)
    actual["datasets"]["mnist"]["params"]["root"] = "/akib/mnist"
    actual["beta"] = .01
    assert not config_matches_recorded_transport(actual, expected, "/akib/mnist")
    actual["beta"] = expected["beta"]
    actual["datasets"]["mnist"]["params"]["split_seed"] = 1
    assert not config_matches_recorded_transport(actual, expected, "/akib/mnist")
