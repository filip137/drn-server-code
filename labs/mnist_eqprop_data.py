"""Official MNIST IDX data, with a held-out training-only validation split."""

import gzip
import hashlib
import struct
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


def read_idx(path):
    path = Path(path)
    if not path.exists():
        path = path.with_name(path.name+".gz")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        raw = stream.read()
    if len(raw) < 4 or raw[:3] != b"\x00\x00\x08" or raw[3] not in (1, 3):
        raise ValueError(f"Expected uint8 IDX labels/images; got {path}")
    dimensions = raw[3]
    shape = struct.unpack(">"+"I"*dimensions, raw[4:4+4*dimensions])
    data = np.frombuffer(raw, dtype=np.uint8, offset=4+4*dimensions)
    if data.size != int(np.prod(shape)):
        raise ValueError(f"Expected {np.prod(shape)} IDX entries; got {data.size} in {path}")
    return data.reshape(shape), hashlib.sha256(raw).hexdigest()


def split_training(labels, validation_size=5000, train_limit=None, val_limit=None):
    train, val = train_test_split(np.arange(len(labels)), test_size=validation_size,
                                stratify=labels, random_state=20260914)
    if train_limit is not None and train_limit < len(train):
        train, _ = train_test_split(train, train_size=train_limit,
                                   stratify=labels[train], random_state=20260915)
    if val_limit is not None and val_limit < len(val):
        val, _ = train_test_split(val, train_size=val_limit,
                                 stratify=labels[val], random_state=20260916)
    return train, val


def load_mnist(raw_dir, *, train_limit=None, val_limit=None):
    root = Path(raw_dir)
    files = ["train-images-idx3-ubyte", "train-labels-idx1-ubyte",
             "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"]
    loaded = [read_idx(root/name) for name in files]
    images, labels, test_images, test_labels = [item[0] for item in loaded]
    shapes = [array.shape for array in (images, labels, test_images, test_labels)]
    expected = [(60000, 28, 28), (60000,), (10000, 28, 28), (10000,)]
    if shapes != expected or labels.max() > 9 or test_labels.max() > 9:
        raise ValueError(f"Expected official MNIST shapes {expected} and labels 0..9; got {shapes}")
    train, val = split_training(labels, train_limit=train_limit, val_limit=val_limit)
    x = images.reshape(60000, 784).astype(np.float64)/255
    mean = x[train].mean(axis=0)
    std = np.maximum(x[train].std(axis=0), .1)
    result = dict(train_x=(x[train]-mean)/std, train_y=labels[train].astype(np.int64),
                  val_x=(x[val]-mean)/std, val_y=labels[val].astype(np.int64),
                  test_x=(test_images.reshape(10000, 784)/255.-mean)/std,
                  test_y=test_labels.astype(np.int64), train_indices=train,
                  val_indices=val, test_indices=np.arange(10000),
                  feature_mean=mean, feature_std=std)
    manifest = dict(dataset="official MNIST, original 28x28 pixels; no PCA or resizing",
        raw_sha256={name:item[1] for name, item in zip(files, loaded)},
        train_examples=len(train), validation_examples=len(val), test_examples=10000,
        features=784, split_seed=20260914, train_subset_seed=20260915,
        validation_subset_seed=20260916,
        normalization="divide by 255; per-pixel mean/std from selected training images only; std floor 0.1",
        test_index_namespace="official test file, separate from train/validation index namespace",
        test_used_for_normalization_or_selection=False)
    return result, manifest
