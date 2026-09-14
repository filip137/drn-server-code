"""Verified Fashion-MNIST IDX data with the homeostasis paper's preprocessing."""

from pathlib import Path

import numpy as np

from labs.mnist_eqprop_data import read_idx


# Raw hashes derived from archives matching the publisher's gzip MD5 checksums:
# https://github.com/zalandoresearch/fashion-mnist#Get-the-Data
RAW_SHA256 = {
    "train-images-idx3-ubyte": "c59f468a2f672dc815687fe0f83887768d799fd8a3f3276145d20f83aa44d888",
    "train-labels-idx1-ubyte": "bad3541b69d912435c50bb6ba87bec294ff4f6a2e1246121d8633921760443d9",
    "t10k-images-idx3-ubyte": "5b4141f0afbad91edebe8549f8fcffe087ea10ca49f1dbef5c9a5cd8815ce37b",
    "t10k-labels-idx1-ubyte": "0402a96d92fd2663957122ceb108a494c5af83dab82d92729df917d7dec38c34",
}


def load_fashion_mnist(raw_dir, *, train_limit=None, dtype="float32"):
    arrays = []
    for name, expected in RAW_SHA256.items():
        array, digest = read_idx(Path(raw_dir)/name)
        if digest != expected:
            raise ValueError(f"Expected verified Fashion-MNIST {name} SHA256 {expected}; got {digest}")
        arrays.append(array)
    images, labels, test_images, test_labels = arrays
    indices = np.arange(len(labels))
    if train_limit is not None:
        if not 1 <= train_limit <= len(labels):
            raise ValueError(f"Expected train_limit in 1..60000; got {train_limit}")
        indices = np.random.default_rng(20260914).permutation(indices)[:train_limit]
    data = dict(train_x=images[indices].reshape(-1, 784).astype(dtype)/255,
                train_y=labels[indices].astype(np.int64),
                test_x=test_images.reshape(-1, 784).astype(dtype)/255,
                test_y=test_labels.astype(np.int64), train_indices=indices)
    manifest = dict(dataset="Fashion-MNIST", raw_sha256=RAW_SHA256,
        train_examples=len(indices), test_examples=len(test_labels),
        normalization="divide pixels by 255 only; no augmentation, centering, or standardization",
        split="official 60000 training / 10000 test, matching authors' utils/data.py",
        evaluation="official test set is the paper's reported validation set; no checkpoint or hyperparameter selection",
        train_subset_seed=20260914 if train_limit is not None else None, dtype=dtype)
    return data, manifest
