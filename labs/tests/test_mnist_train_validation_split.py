from itertools import islice

import pytest
import torch

from labs import datasets as dataset_module


def _balanced_targets():
    return torch.arange(10, dtype=torch.long).repeat_interleave(6_000)


class _FakeMnist:
    calls = []

    def __init__(self, *, root, train, download, transform):
        del root, download
        self.calls.append(bool(train))
        if not train:
            raise AssertionError("the official MNIST test split was instantiated")
        self.targets = _balanced_targets()
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        # The fake intentionally returns a tensor directly; transform behavior is
        # covered by the existing image-dataset path and is not part of this test.
        return torch.tensor([int(index)], dtype=torch.long), int(self.targets[index])


def _build(monkeypatch, *, return_source_indices=True):
    _FakeMnist.calls = []
    monkeypatch.setattr(dataset_module, "_TrainOnlyMNIST", _FakeMnist)
    return dataset_module.build_mnist_train_validation_loaders(
        batch_size=16,
        root="unused",
        download=False,
        normalize=False,
        normalize_std=1.0,
        split_seed=0,
        shuffle_seed=0,
        return_source_indices=return_source_indices,
    )


def test_stratified_split_is_exact_disjoint_and_deterministic():
    targets = _balanced_targets()

    train_indices, validation_indices = (
        dataset_module.stratified_mnist_train_validation_indices(
            targets, split_seed=0
        )
    )
    train_again, validation_again = (
        dataset_module.stratified_mnist_train_validation_indices(
            targets, split_seed=0
        )
    )

    assert len(train_indices) == 55_000
    assert len(validation_indices) == 5_000
    assert train_indices == train_again
    assert validation_indices == validation_again
    assert train_indices == tuple(sorted(train_indices))
    assert validation_indices == tuple(sorted(validation_indices))
    assert set(train_indices).isdisjoint(validation_indices)
    assert set(train_indices) | set(validation_indices) == set(range(60_000))

    validation_targets = targets[list(validation_indices)]
    assert torch.bincount(validation_targets, minlength=10).tolist() == [500] * 10

    _, validation_other_seed = (
        dataset_module.stratified_mnist_train_validation_indices(
            targets, split_seed=1
        )
    )
    assert validation_other_seed != validation_indices


def test_validation_only_builder_never_instantiates_official_test(monkeypatch):
    loaders = _build(monkeypatch)

    assert _FakeMnist.calls == [True]
    assert len(loaders.train_loader.dataset) == 55_000
    assert len(loaders.validation_loader.dataset) == 5_000
    assert loaders.train_loader.batch_size == 16
    assert loaders.validation_loader.batch_size == 128

    validation_batch = next(iter(loaders.validation_loader))
    images, labels, original_indices = validation_batch
    assert images[:, 0].tolist() == original_indices.tolist()
    assert labels.tolist() == _balanced_targets()[original_indices].tolist()
    assert original_indices.tolist() == list(loaders.validation_indices[:128])


def test_train_only_mnist_resource_contract_excludes_official_test_files():
    resource_names = [
        url.rsplit("/", 1)[-1]
        for url, _digest in dataset_module._TrainOnlyMNIST.resources
    ]

    assert resource_names == [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
    ]
    assert all("t10k" not in name for name in resource_names)


def test_train_only_mnist_bypasses_legacy_test_file_probe(
    tmp_path, monkeypatch
):
    from torchvision.datasets import mnist as torchvision_mnist

    # Force the condition under which torchvision's inherited implementation
    # would hash both training.pt and the prohibited official-test test.pt.
    processed = tmp_path / "MNIST" / "processed"
    processed.mkdir(parents=True)
    (processed / "training.pt").write_bytes(b"legacy-train")
    (processed / "test.pt").write_bytes(b"legacy-test")

    probed_paths = []

    def record_integrity(path, *unused_args, **unused_kwargs):
        probed_paths.append(str(path))
        return True

    monkeypatch.setattr(torchvision_mnist, "check_integrity", record_integrity)
    monkeypatch.setattr(
        dataset_module._TrainOnlyMNIST,
        "_load_data",
        lambda self: (torch.zeros(1, 28, 28), torch.zeros(1, dtype=torch.long)),
    )

    dataset = dataset_module._TrainOnlyMNIST(
        root=tmp_path,
        train=True,
        download=False,
        transform=None,
    )

    assert dataset.train is True
    assert len(probed_paths) == 2
    assert all("train-" in path for path in probed_paths)
    assert all("test.pt" not in path and "t10k" not in path for path in probed_paths)
    with pytest.raises(ValueError, match="train=True"):
        dataset_module._TrainOnlyMNIST(
            root=tmp_path,
            train=False,
            download=False,
            transform=None,
        )


def test_original_index_subset_does_not_reindex_affine_source():
    class RecordingDataset:
        def __init__(self):
            self.requests = []

        def __getitem__(self, index):
            self.requests.append(index)
            return torch.tensor(index), index % 10

    source = RecordingDataset()
    subset = dataset_module.OriginalIndexSubset(
        source,
        [17, 4, 29],
        return_source_index=True,
    )

    image, label, original_index = subset[1]

    assert source.requests == [4]
    assert image.item() == 4
    assert label == 4
    assert original_index == 4


def test_train_shuffle_can_be_reset_and_matches_recorded_batch_hash(monkeypatch):
    loaders = _build(monkeypatch)
    expected_epoch = loaders.train_batch_indices(num_epochs=1)[0]

    observed = []
    for _, _, original_indices in islice(loaders.train_loader, 3):
        observed.append(tuple(original_indices.tolist()))
    assert tuple(observed) == expected_epoch[:3]

    loaders.reset_train_shuffle()
    replayed = []
    for _, _, original_indices in islice(loaders.train_loader, 3):
        replayed.append(tuple(original_indices.tolist()))
    assert replayed == observed

    epoch_hashes = loaders.train_batch_order_hashes(num_epochs=2)
    assert loaders.first_epoch_batch_order_hash == epoch_hashes[0]
    assert epoch_hashes[0] != epoch_hashes[1]
    assert epoch_hashes == loaders.train_batch_order_hashes(num_epochs=2)


def test_provenance_exposes_exact_indices_and_stable_hashes(monkeypatch):
    loaders = _build(monkeypatch, return_source_indices=False)

    provenance = loaders.provenance(num_epochs=2)

    assert provenance["schema"] == "mnist-train-validation-split/v1"
    assert provenance["source_split"] == "train"
    assert provenance["split_seed"] == 0
    assert provenance["shuffle_seed"] == 0
    assert provenance["train_indices"] == list(loaders.train_indices)
    assert provenance["validation_indices"] == list(loaders.validation_indices)
    assert provenance["train_indices_sha256"] == (
        dataset_module.stable_index_sequence_hash(loaders.train_indices)
    )
    assert provenance["validation_indices_sha256"] == (
        dataset_module.stable_index_sequence_hash(loaders.validation_indices)
    )
    assert provenance["train_batch_order_sha256"] == list(
        loaders.train_batch_order_hashes(num_epochs=2)
    )

    # The compatibility mode continues to yield the conventional (image, label)
    # pair even though exact source indices remain available in the bundle.
    assert len(next(iter(loaders.validation_loader))) == 2
