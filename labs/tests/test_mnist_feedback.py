"""Scientific scaling gates: identical measurements, no oracle leakage, IDX split."""

import copy
import gzip
import struct

import numpy as np
import pytest
import torch

from labs.adjoint_baselines import AveragedMomentum, MeasuredBaseline
from labs.mnist_eqprop_data import read_idx, split_training
from labs.random_nudge_hopfield import Hopfield
from labs.recurrent_eqprop import make_classifier, flatten_gradient
from labs.tools.train_recurrent_eqprop_digits import gradient_step
from labs.torch_recurrent_eqprop import PhysicalBackend


@pytest.mark.parametrize("method", ["contrastive_ep", "known_skew_asymep", "dc_asymep", "noise_asymep", "learned_mc4"])
def test_physical_backend_matches_noisy_numpy_trajectory_without_oracle(monkeypatch, method):
    torch.set_num_threads(1)
    base = make_classifier(1, size=8, outputs=2, input_size=3)
    models = [copy.deepcopy(base), copy.deepcopy(base)]
    learners = [MeasuredBaseline.zeros(8, 2) if method == "learned_mc4" else None for _ in range(2)]
    rngs = [np.random.default_rng(33), np.random.default_rng(33)]
    optimizers = [AveragedMomentum(.9), AveragedMomentum(.9)]
    controller = -.96*base.skew if method in ("dc_asymep", "noise_asymep") else None
    cpu_method = "circulation_asymep" if controller is not None else method
    backend = PhysicalBackend(tolerance=1e-12)

    def forbidden(*args, **kwargs):
        raise AssertionError("physical estimator accessed a Jacobian/adjoint/autograd")

    monkeypatch.setattr(Hopfield, "jacobian", forbidden)
    monkeypatch.setattr(np.linalg, "solve", forbidden)
    monkeypatch.setattr(torch.linalg, "solve", forbidden)
    monkeypatch.setattr(torch.autograd, "grad", forbidden)
    x = np.random.default_rng(55).normal(size=(5, 3))
    labels = np.array([0, 1, 0, 0, 1])
    for _ in range(3):
        g0, m0 = gradient_step(models[0], x, labels, cpu_method, rngs[0], beta=.01,
            sigma=1e-5, tolerance=1e-12, learner=learners[0], feedback_controller=controller)
        g1, m1, _ = backend.gradient(models[1], x, labels, method, rngs[1], beta=.01,
                                    sigma=1e-5, learner=learners[1], controller=controller)
        np.testing.assert_allclose(flatten_gradient(g0), flatten_gradient(g1), atol=2e-9, rtol=2e-7)
        expected = len(x)*(9 if method == "learned_mc4" else 3)
        assert m0["equilibrations"] == m1["equilibrations"] == expected
        assert m1["max_residual"] <= 1e-12
        for model, opt, g in zip(models, optimizers, (g0, g1)):
            model.update(opt.direction(g), .1)
    np.testing.assert_allclose(models[0].inputs, models[1].inputs, atol=2e-9, rtol=2e-7)
    if learners[0] is not None:
        assert learners[0].observations == learners[1].observations == 60
        np.testing.assert_allclose(learners[0].matrix, learners[1].matrix, atol=2e-9, rtol=2e-7)


def test_idx_reader_raw_gzip_and_bad_length(tmp_path):
    images = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
    raw = struct.pack(">IIII", 2051, *images.shape)+images.tobytes()
    path = tmp_path/"images"
    path.write_bytes(raw)
    first, checksum = read_idx(path)
    np.testing.assert_array_equal(first, images)
    path.unlink()
    with gzip.open(str(path)+".gz", "wb") as stream:
        stream.write(raw)
    second, other_checksum = read_idx(path)
    np.testing.assert_array_equal(second, first)
    assert checksum == other_checksum
    path.write_bytes(raw[:-1])
    with pytest.raises(ValueError, match="Expected 24 IDX entries"):
        read_idx(path)


def test_mnist_validation_and_operational_subsets_are_disjoint_reproducible():
    labels = np.repeat(np.arange(10), 100)
    train, val = split_training(labels, validation_size=200)
    assert len(train) == 800 and len(val) == 200
    assert not np.intersect1d(train, val).size
    smaller, validation = split_training(labels, validation_size=200, train_limit=100, val_limit=50)
    assert set(smaller) <= set(train)
    assert set(validation) <= set(val)
    assert len(smaller) == 100 and len(validation) == 50
    assert not np.intersect1d(smaller, validation).size
    np.testing.assert_array_equal(split_training(labels, validation_size=200)[0], train)
