"""Check oracle separation, physical budgets and dataset isolation in training."""

from itertools import combinations

import numpy as np
import pytest
from sklearn.datasets import load_digits

from labs.adjoint_estimators import estimate_adjoint, make_probes
from labs.random_nudge_hopfield import Hopfield
from labs.recurrent_eqprop import cost_gradient, error_pair, flatten_gradient, make_classifier, parameter_gradient, settle
from labs.tools import train_recurrent_eqprop_digits as digits


@pytest.mark.parametrize(
    "method,asymmetry",
    [("contrastive_ep", 0.0), ("known_skew_asymep", 1.0), ("mc8", 1.0),
     ("kaczmarz8", 1.0), ("lstsq8", 1.0), ("ridge8", 1.0), ("orthogonal8", 1.0),
     ("orthogonal_mc8", 1.0)],
)
def test_physical_gradient_steps_avoid_oracle_and_match_declared_probe_budget(monkeypatch, method, asymmetry):
    model = make_classifier(20, size=8, outputs=2, input_size=3, asymmetry=asymmetry)
    x = np.random.default_rng(21).normal(scale=0.5, size=(3, 3))
    labels = np.array([0, 1, 0])
    net = model.network()
    free = settle(net, model.drive(x), tolerance=1e-13).state
    c = cost_gradient(free, labels, model.hidden, model.logit_scale)
    reference = parameter_gradient(model, x, free, digits.oracle_adjoint(net, free, c))
    beta = 5e-5

    def forbidden(*args, **kwargs):
        raise AssertionError("A physical gradient_step called an exact-adjoint oracle")

    monkeypatch.setattr(digits, "oracle_adjoint", forbidden)
    monkeypatch.setattr(Hopfield, "jacobian", forbidden)
    monkeypatch.setattr(np.linalg, "solve", forbidden)
    gradient, metadata = digits.gradient_step(
        model, x, labels, method, np.random.default_rng(22),
        beta=beta, sigma=0.0, tolerance=1e-13, audit=False,
    )
    assert np.isfinite(flatten_gradient(gradient)).all()
    if method in ("contrastive_ep", "known_skew_asymep", "orthogonal8", "orthogonal_mc8"):
        for actual, exact in zip(gradient, reference):
            np.testing.assert_allclose(actual, exact, atol=3e-8, rtol=3e-7)
    probe_count = 0 if method in ("contrastive_ep", "known_skew_asymep") else 8
    assert metadata["probe_count"] == probe_count
    assert metadata["equilibrations"] == len(x)*(3+2*probe_count)
    assert metadata["state_reads"] == metadata["equilibrations"]
    assert metadata["total_probe_excitation_sq"] == pytest.approx(2*len(x)*probe_count*beta**2)
    norm = np.linalg.norm(c, axis=-1)
    effective = beta/np.maximum(norm, 1.0)
    assert metadata["total_error_excitation_sq"] == pytest.approx(2*np.sum((effective*norm)**2))
    assert metadata["total_error_excitation_sq"] <= 2*len(x)*beta**2*(1+1e-14)
    assert metadata["mean_free_error_force_norm"] == pytest.approx(np.mean(norm))
    assert metadata["max_residual"] <= 1e-13
    assert "gradient_relative_error" not in metadata
    assert "realized_jacobian_asymmetry" not in metadata


def test_orthogonal_mc_dispatch_uses_same_measurements_with_n_over_m_correction():
    model = make_classifier(30, size=16, outputs=2, input_size=3, asymmetry=1.0)
    x = np.random.default_rng(31).normal(size=(3, 3))
    labels = np.array([0, 1, 0])
    net = model.network()
    free = settle(net, model.drive(x), tolerance=1e-12).state
    pair = error_pair(model, net, x, labels, free, 0.01, tolerance=1e-12)
    baseline = parameter_gradient(model, x, free, pair.response)
    results = []
    for method in ("orthogonal8", "orthogonal_mc8"):
        results.append(digits.gradient_step(
            model, x, labels, method, np.random.default_rng(32), beta=0.01,
            sigma=0.0, tolerance=1e-12, audit=False,
        ))
    for base, projection, mc in zip(baseline, results[0][0], results[1][0]):
        np.testing.assert_allclose(mc, base + 2*(projection-base), atol=1e-13, rtol=1e-13)
    for field in ("probe_count", "equilibrations", "state_reads", "total_probe_excitation_sq"):
        assert results[0][1][field] == results[1][1][field]
    assert "orthogonal_mc8" in digits.METHOD_CHOICES
    assert "orthogonal_mc8" not in digits.METHODS


def test_random_orthogonal_subset_mc_has_exact_expectation():
    # Average every size-two subset of an orthonormal basis. Random ordering
    # in make_probes draws these subspaces uniformly; n/m removes their bias.
    basis = make_probes(np.random.default_rng(33), 4, 4, "hadamard")
    truth, baseline = np.random.default_rng(34).normal(size=(2, 4))
    estimates = []
    for indices in combinations(range(4), 2):
        probes = basis[list(indices)]
        estimates.append(estimate_adjoint(baseline, probes, probes @ truth, "mc"))
    np.testing.assert_allclose(np.mean(estimates, axis=0), truth, atol=1e-14)
    np.testing.assert_allclose(np.mean(np.sum((np.array(estimates)-truth)**2, axis=-1)),
                               (4/2-1)*np.sum((truth-baseline)**2), atol=1e-14)


def test_auditing_is_post_estimation_and_does_not_change_noisy_gradient():
    model = make_classifier(23, size=8, outputs=2, input_size=3, asymmetry=1.0)
    x = np.random.default_rng(24).normal(size=(3, 3))
    labels = np.array([0, 1, 1])
    gradients = []
    for audit in (False, True):
        gradient, metadata = digits.gradient_step(
            model, x, labels, "ridge8", np.random.default_rng(25),
            beta=0.01, sigma=1e-5, audit=audit,
        )
        gradients.append(gradient)
    for unaudited, audited in zip(*gradients):
        np.testing.assert_array_equal(unaudited, audited)
    assert np.isfinite(metadata["gradient_cosine"])
    assert metadata["gradient_relative_error"] >= 0


@pytest.mark.parametrize("quick", [False, True])
def test_digits_split_is_disjoint_and_normalization_uses_training_only(quick):
    data = digits.dataset(quick)
    raw, labels = load_digits(return_X_y=True)
    raw = raw/16.0
    groups = [data[f"{name}_indices"] for name in ("train", "val", "test")]
    for i, indices in enumerate(groups):
        assert len(np.unique(indices)) == len(indices)
        for other in groups[i+1:]:
            assert not np.intersect1d(indices, other).size
    expected_mean = raw[groups[0]].mean(axis=0)
    expected_std = np.maximum(raw[groups[0]].std(axis=0), 0.1)
    np.testing.assert_array_equal(data["feature_mean"], expected_mean)
    np.testing.assert_array_equal(data["feature_std"], expected_std)
    for name, indices in zip(("train", "val", "test"), groups):
        np.testing.assert_array_equal(data[f"{name}_y"], labels[indices])
        np.testing.assert_allclose(data[f"{name}_x"], (raw[indices]-expected_mean)/expected_std, atol=1e-15)
        assert len(np.unique(data[f"{name}_y"])) == 10
    np.testing.assert_allclose(data["train_x"].mean(axis=0), 0.0, atol=1e-14)
    if quick:
        assert [len(indices) for indices in groups] == [192, 64, 64]
    else:
        assert sum(map(len, groups)) == len(raw)


def test_learning_rate_calibration_never_evaluates_held_out_test(monkeypatch, tmp_path):
    rng = np.random.default_rng(26)
    data = dict(train_x=rng.normal(size=(4, 3)), train_y=np.array([0, 1, 0, 1]),
                val_x=rng.normal(size=(2, 3)), val_y=np.array([0, 1]),
                test_x=rng.normal(size=(2, 3)), test_y=np.array([1, 0]))
    monkeypatch.setattr(digits, "make_classifier", lambda seed, asymmetry, **kwargs:
                        make_classifier(seed, size=8, outputs=2, input_size=3, asymmetry=asymmetry))
    original_evaluate = digits.evaluate
    calls = []

    def checked_evaluate(model, x, labels):
        assert x is not data["test_x"], "Calibration accessed held-out test inputs"
        assert labels is not data["test_y"], "Calibration accessed held-out test labels"
        calls.append(x)
        return original_evaluate(model, x, labels)

    monkeypatch.setattr(digits, "evaluate", checked_evaluate)
    config = dict(asymmetry=1.0, epochs=1, learning_rate=0.01, batch_size=4,
                  beta=0.01, read_noise=0.0, audit=False)
    rows = digits.train_one(data, "adjoint", 0, config, tmp_path,
                            lambda detail, advance: None, calibration=True)
    assert len(calls) == 4  # train/validation before and after the one update
    assert len(rows) == 2
    assert all("test_loss" not in row and "test_accuracy" not in row for row in rows)


@pytest.mark.parametrize("method", ["mc0", "mc-1", "mc1.5", "unknown8", "orthogonal17"])
def test_scaling_rejects_invalid_probe_budgets(method):
    with pytest.raises(ValueError, match="Expected"):
        digits.probe_specification(method, size=16)


def test_scaling_honors_configured_width_and_probe_count(tmp_path):
    data = digits.dataset(quick=True)
    data["train_x"], data["train_y"] = data["train_x"][:4], data["train_y"][:4]
    data["val_x"], data["val_y"] = data["val_x"][:4], data["val_y"][:4]
    config = dict(size=16, outputs=10, asymmetry=1.0, epochs=1,
                  learning_rate=0.01, batch_size=4, beta=0.01, read_noise=0.0, audit=False)
    rows = digits.train_one(data, "mc16", 0, config, tmp_path,
                            lambda detail, advance: None, calibration=True)
    with np.load(tmp_path/"mc16_seed0_lr0.01.npz") as values:
        assert values["symmetric"].shape == (16,16)
        assert values["inputs"].shape == (6,64)
        assert np.linalg.norm(values["symmetric"]-values["initial_symmetric"]) > 0
    assert rows[-1]["equilibrations"] == 4*(3+2*16)
    assert rows[-1]["gradient_seconds"] > 0
    assert rows[-1]["update_seconds"] > 0
