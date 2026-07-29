from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import hashlib
import numpy as np
import os
import random
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
import torch
from torch.utils.data import random_split, DataLoader, Sampler, TensorDataset


def _make_moons(*args, **kwargs):
    from sklearn.datasets import make_moons

    return make_moons(*args, **kwargs)


def _load_digits(*args, **kwargs):
    from sklearn.datasets import load_digits

    return load_digits(*args, **kwargs)


AFFINE_PRESETS = {
    "none": {
        "degrees": 0.0,
        "translate": [0.0, 0.0],
        "scale": [1.0, 1.0],
        "shear": 0.0,
    },
    "medium": {
        "degrees": 25.0,
        "translate": [0.20, 0.20],
        "scale": [0.80, 1.20],
        "shear": 0.0,
    },
    "mnist_affine": {
        "degrees": 60.0,
        "translate": [0.15, 0.15],
        "scale": [0.80, 1.20],
        "shear": 15.0,
    },
}


MNIST_TRAIN_SIZE = 60_000
MNIST_VALIDATION_PER_CLASS = 500
MNIST_VALIDATION_SIZE = 5_000
MNIST_LR_STUDY_TRAIN_SIZE = 55_000
MNIST_LR_STUDY_VALIDATION_BATCH_SIZE = 128


class _TrainOnlyMNIST(datasets.MNIST):
    """MNIST reader whose existence/download contract contains train files only.

    ``torchvision.datasets.MNIST(train=True)`` loads only the training tensors,
    but its inherited ``_check_exists`` hashes all four raw resources, including
    the official test images and labels.  The Conv LR studies prohibit even
    that incidental read, so their train/validation-only path narrows the
    resource list before the torchvision constructor performs its checks.
    """

    resources = tuple(datasets.MNIST.resources[:2])

    def __init__(self, *args, train=True, **kwargs):
        if train is not True:
            raise ValueError(
                "Expected the train-only MNIST reader to receive train=True. "
                f"Provided value: {train!r}."
            )
        super().__init__(*args, train=True, **kwargs)

    def _check_legacy_exist(self):
        """Bypass torchvision's legacy train+test processed-file probe.

        ``MNIST._check_legacy_exist`` hashes both ``training.pt`` and
        ``test.pt`` even when ``train=True``.  This study must never touch the
        official test artifact, so it always uses the raw train-only path.
        """

        return False

    @property
    def raw_folder(self):
        return os.path.join(self.root, "MNIST", "raw")

    @property
    def processed_folder(self):
        return os.path.join(self.root, "MNIST", "processed")


def stable_index_sequence_hash(indices):
    """Return a platform-independent SHA-256 for an ordered index sequence."""

    values = tuple(int(index) for index in indices)
    digest = hashlib.sha256()
    digest.update(b"mnist-index-sequence/v1\0")
    digest.update(len(values).to_bytes(8, byteorder="big", signed=False))
    for value in values:
        if value < 0:
            raise ValueError(f"Expected non-negative dataset indices, got {value}.")
        digest.update(value.to_bytes(8, byteorder="big", signed=False))
    return digest.hexdigest()


def stable_batch_order_hash(batches):
    """Hash ordered batches while retaining batch-boundary information."""

    digest = hashlib.sha256()
    digest.update(b"mnist-batch-order/v1\0")
    for batch in batches:
        values = tuple(int(index) for index in batch)
        digest.update(b"B")
        digest.update(len(values).to_bytes(8, byteorder="big", signed=False))
        for value in values:
            if value < 0:
                raise ValueError(f"Expected non-negative dataset indices, got {value}.")
            digest.update(value.to_bytes(8, byteorder="big", signed=False))
    return digest.hexdigest()


def stratified_mnist_train_validation_indices(targets, *, split_seed=0):
    """Select 500 validation examples per MNIST class using ``torch.randperm``.

    A generator dedicated to splitting is seeded once and used class by class in
    label order.  Returned indices are sorted into original MNIST order so that
    validation traversal is canonical and independent of the selection draws.
    """

    labels = torch.as_tensor(targets, dtype=torch.long).reshape(-1).cpu()
    if labels.numel() != MNIST_TRAIN_SIZE:
        raise ValueError(
            f"Expected {MNIST_TRAIN_SIZE} MNIST training labels, got {labels.numel()}."
        )
    classes = torch.unique(labels, sorted=True).tolist()
    if classes != list(range(10)):
        raise ValueError(f"Expected MNIST classes 0 through 9, got {classes}.")

    generator = torch.Generator().manual_seed(int(split_seed))
    train_indices = []
    validation_indices = []
    for label in classes:
        class_indices = torch.nonzero(labels == label, as_tuple=False).flatten()
        if class_indices.numel() < MNIST_VALIDATION_PER_CLASS:
            raise ValueError(
                "Expected at least "
                f"{MNIST_VALIDATION_PER_CLASS} MNIST samples for class {label}, "
                f"got {class_indices.numel()}."
            )
        permutation = torch.randperm(class_indices.numel(), generator=generator)
        validation_indices.extend(
            class_indices[permutation[:MNIST_VALIDATION_PER_CLASS]].tolist()
        )
        train_indices.extend(
            class_indices[permutation[MNIST_VALIDATION_PER_CLASS:]].tolist()
        )

    train_indices = tuple(sorted(int(index) for index in train_indices))
    validation_indices = tuple(sorted(int(index) for index in validation_indices))
    if len(train_indices) != MNIST_LR_STUDY_TRAIN_SIZE:
        raise RuntimeError(
            f"Expected {MNIST_LR_STUDY_TRAIN_SIZE} training indices, "
            f"got {len(train_indices)}."
        )
    if len(validation_indices) != MNIST_VALIDATION_SIZE:
        raise RuntimeError(
            f"Expected {MNIST_VALIDATION_SIZE} validation indices, "
            f"got {len(validation_indices)}."
        )
    return train_indices, validation_indices


class OriginalIndexSubset(torch.utils.data.Dataset):
    """Subset that indexes its source with, and can return, original indices."""

    def __init__(self, dataset, indices, *, return_source_index=False):
        self.dataset = dataset
        self.indices = tuple(int(index) for index in indices)
        self.return_source_index = bool(return_source_index)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        source_index = self.indices[int(position)]
        item = self.dataset[source_index]
        if not self.return_source_index:
            return item
        if isinstance(item, tuple):
            return (*item, source_index)
        return item, source_index


class ResettableRandomSampler(Sampler):
    """Random sampler whose RNG can be restored to its candidate-start state."""

    def __init__(self, data_source, *, seed):
        self.data_source = data_source
        self.seed = int(seed)
        self.generator = torch.Generator()
        self.reset()

    def __len__(self):
        return len(self.data_source)

    def __iter__(self):
        order = torch.randperm(len(self.data_source), generator=self.generator)
        return iter(order.tolist())

    def reset(self):
        self.generator.manual_seed(self.seed)


def _batch_original_indices(
    source_indices,
    *,
    batch_size,
    shuffle_seed,
    num_epochs,
):
    if int(batch_size) <= 0:
        raise ValueError(f"Expected a positive train batch size, got {batch_size}.")
    if int(num_epochs) <= 0:
        raise ValueError(f"Expected a positive number of epochs, got {num_epochs}.")

    source_indices = tuple(int(index) for index in source_indices)
    generator = torch.Generator().manual_seed(int(shuffle_seed))
    epochs = []
    for _ in range(int(num_epochs)):
        positions = torch.randperm(len(source_indices), generator=generator).tolist()
        ordered = [source_indices[position] for position in positions]
        epochs.append(
            tuple(
                tuple(ordered[start : start + int(batch_size)])
                for start in range(0, len(ordered), int(batch_size))
            )
        )
    return tuple(epochs)


@dataclass
class MnistTrainValidationLoaders:
    """Loaders and exact provenance for the LR-study train/validation split."""

    train_loader: DataLoader
    validation_loader: DataLoader
    train_indices: tuple
    validation_indices: tuple
    split_seed: int
    shuffle_seed: int
    batch_size: int
    validation_batch_size: int
    train_indices_hash: str
    validation_indices_hash: str
    first_epoch_batch_order_hash: str
    _train_sampler: ResettableRandomSampler = field(repr=False)
    _worker_generator: torch.Generator = field(repr=False)

    def reset_train_shuffle(self):
        """Reset sampler and worker RNGs so another candidate sees the same order."""

        self._train_sampler.reset()
        self._worker_generator.manual_seed(self.shuffle_seed)
        return self.train_loader

    def reset_train_loader(self):
        """Compatibility alias for callers that reset between LR candidates."""

        return self.reset_train_shuffle()

    def train_batch_indices(self, *, num_epochs=1):
        """Return original MNIST indices grouped exactly as candidate batches."""

        return _batch_original_indices(
            self.train_indices,
            batch_size=self.batch_size,
            shuffle_seed=self.shuffle_seed,
            num_epochs=num_epochs,
        )

    def train_batch_order_hashes(self, *, num_epochs=1):
        """Return one original-index batch-order hash per epoch."""

        return tuple(
            stable_batch_order_hash(epoch_batches)
            for epoch_batches in self.train_batch_indices(num_epochs=num_epochs)
        )

    def provenance(self, *, num_epochs=1):
        """Return JSON-serializable split and shuffle provenance, including indices."""

        return {
            "schema": "mnist-train-validation-split/v1",
            "source_split": "train",
            "split_method": "class-wise-torch-randperm-500-per-class",
            "split_seed": self.split_seed,
            "shuffle_seed": self.shuffle_seed,
            "batch_size": self.batch_size,
            "validation_batch_size": self.validation_batch_size,
            "train_indices": list(self.train_indices),
            "validation_indices": list(self.validation_indices),
            "train_indices_sha256": self.train_indices_hash,
            "validation_indices_sha256": self.validation_indices_hash,
            "train_batch_order_sha256": list(
                self.train_batch_order_hashes(num_epochs=num_epochs)
            ),
        }


def affine_config_from_preset(
    preset="none",
    *,
    degrees=None,
    translate=None,
    scale=None,
    shear=None,
    seed=1729,
):
    if preset not in AFFINE_PRESETS:
        raise ValueError(f"Expected affine preset in {sorted(AFFINE_PRESETS)}, got {preset!r}.")
    config = dict(AFFINE_PRESETS[preset])
    if degrees is not None:
        config["degrees"] = float(degrees)
    if translate is not None:
        values = [float(value) for value in translate]
        if len(values) == 1:
            values = [values[0], values[0]]
        if len(values) != 2:
            raise ValueError(f"Expected affine translate to contain 1 or 2 values, got {values}.")
        config["translate"] = values
    if scale is not None:
        values = [float(value) for value in scale]
        if len(values) != 2:
            raise ValueError(f"Expected affine scale to contain 2 values, got {values}.")
        config["scale"] = values
    if shear is not None:
        config["shear"] = float(shear)

    scale_min, scale_max = [float(value) for value in config["scale"]]
    if scale_min <= 0.0 or scale_max <= 0.0 or scale_min > scale_max:
        raise ValueError(f"Expected positive ordered affine scale, got {config['scale']}.")
    translate_values = [float(value) for value in config["translate"]]
    if any(value < 0.0 for value in translate_values):
        raise ValueError(f"Expected non-negative affine translate, got {translate_values}.")

    enabled = (
        abs(float(config["degrees"])) > 0.0
        or any(abs(value) > 0.0 for value in translate_values)
        or abs(scale_min - 1.0) > 0.0
        or abs(scale_max - 1.0) > 0.0
        or abs(float(config.get("shear", 0.0))) > 0.0
    )
    config["preset"] = preset
    config["seed"] = int(seed)
    config["enabled"] = enabled
    return config


class DeterministicAffineImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, *, transform, affine_config, split):
        self.dataset = dataset
        self.transform = transform
        self.affine_config = dict(affine_config)
        self.split_offset = 0 if split == "train" else 10_000_000

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        if bool(self.affine_config.get("enabled", False)):
            image = self._apply_affine(image, int(index))
        return self.transform(image), label

    def _apply_affine(self, image, index):
        seed = int(self.affine_config.get("seed", 1729))
        rng = random.Random(seed + self.split_offset + 104_729 * index)
        degrees = float(self.affine_config["degrees"])
        translate_frac = [float(value) for value in self.affine_config["translate"]]
        scale_range = [float(value) for value in self.affine_config["scale"]]
        shear = float(self.affine_config.get("shear", 0.0))

        angle = rng.uniform(-degrees, degrees) if degrees else 0.0
        scale = rng.uniform(scale_range[0], scale_range[1])
        shear_x = rng.uniform(-shear, shear) if shear else 0.0
        width, height = image.size
        max_dx = translate_frac[0] * width
        max_dy = translate_frac[1] * height
        translate = (
            int(round(rng.uniform(-max_dx, max_dx))),
            int(round(rng.uniform(-max_dy, max_dy))),
        )
        return TF.affine(
            image,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=[shear_x, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        )



class IndexedDataset(torch.utils.data.Dataset):
    """
    Wrapper class that turns a (labeled) datadet into an `indexed dataset'.

    Each item of an `indexed dataset' consists of an image, its label, and its index in the dataset.
    In contrast, the standard Dataset class does not return the index of the image

    Attributes
    ----------
    _dataset (Dataset): the original dataset that we wrap into an `indexed dataset'
    """

    def __init__(self, dataset):
        """Creates an instance of IndexedDataset

        Args:
            dataset (Dataset): the original dataset that we wrap into an `indexed dataset'
        """
        self._dataset = dataset
        
    def __getitem__(self, index):
        """Returns the image, its label, and its index."""
        data, target = self._dataset[index]
        
        return data, target, index

    def __len__(self):
        return len(self._dataset)


class Datasets(ABC):
    def __init__(self, name, batch_size, device):
        self.name = name
        self.batch_size = batch_size
        self.device = device

    @abstractmethod
    def build(self):
        pass

class MoonsDataset(Datasets):
    def __init__(self, name, batch_size, device, num_samples, input_dim=None):
        super().__init__(name, batch_size, device)
        self.num_samples = num_samples
        self.noise = 0.1
        self.input_dim = int(input_dim) if input_dim is not None else None

    @staticmethod
    def _expand_xy_features(x, target_dim):
        if x.shape[1] == target_dim:
            return x
        if x.shape[1] != 2:
            raise ValueError(f"Expected 2D inputs for moons expansion, got shape {tuple(x.shape)}.")
        if target_dim < 2 or target_dim % 2 != 0:
            raise ValueError(f"Moons input_dim must be an even number >= 2; got {target_dim}.")
        half = target_dim // 2
        x1 = x[:, [0]].repeat(1, half)
        x2 = x[:, [1]].repeat(1, half)
        return torch.cat([x1, x2], dim=1)
    def build(self):
        x, y = _make_moons(n_samples=self.num_samples, noise=self.noise, random_state=42)
        # Keep features in float32 to match model weights; labels stay int for targets.
        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        y = torch.tensor(y, dtype=torch.long, device=self.device)
        # Normalize each feature to [-1, 1] based on global min/max.
        eps = 1e-12
        x_min = x.min(dim=0).values
        x_max = x.max(dim=0).values
        scale = (x_max - x_min).clamp_min(eps)
        x = (x - x_min) / scale * 2.0 - 1.0
        if self.input_dim is not None and self.input_dim != x.shape[1]:
            x = self._expand_xy_features(x, self.input_dim)
        dataset = TensorDataset(x, y)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_ds, test_ds = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(0))
        test_ds = IndexedDataset(test_ds)
        train_data_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=False)
        test_data_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)
        return train_data_loader, test_data_loader


class YinYangDataset(Datasets):
    """
    Balanced 2D Yin-Yang classification dataset with 3 classes:
    yin (0), yang (1), and dots (2).
    """

    def __init__(
        self,
        name,
        batch_size,
        device,
        num_samples,
        *,
        r_small=0.1,
        r_big=0.5,
        seed=42,
    ):
        super().__init__(name, batch_size, device)
        self.num_samples = int(num_samples)
        self.r_small = float(r_small)
        self.r_big = float(r_big)
        self.seed = int(seed)

    def _dist_to_right_dot(self, x, y):
        return np.sqrt((x - 1.5 * self.r_big) ** 2 + (y - self.r_big) ** 2)

    def _dist_to_left_dot(self, x, y):
        return np.sqrt((x - 0.5 * self.r_big) ** 2 + (y - self.r_big) ** 2)

    def _which_class(self, x, y):
        d_right = self._dist_to_right_dot(x, y)
        d_left = self._dist_to_left_dot(x, y)

        criterion1 = d_right <= self.r_small
        criterion2 = d_left > self.r_small and d_left <= 0.5 * self.r_big
        criterion3 = y > self.r_big and d_right > 0.5 * self.r_big
        is_yin = criterion1 or criterion2 or criterion3

        is_dot = d_right < self.r_small or d_left < self.r_small
        if is_dot:
            return 2
        return int(is_yin)

    def _sample_xy_for_class(self, rng, goal_class):
        while True:
            x, y = rng.rand(2) * 2.0 * self.r_big
            if np.sqrt((x - self.r_big) ** 2 + (y - self.r_big) ** 2) > self.r_big:
                continue
            c = self._which_class(x, y)
            if c == goal_class:
                return x, y

    def build(self):
        rng = np.random.RandomState(self.seed)
        xs = []
        ys = []

        # Keep classes balanced by target-class rejection sampling.
        for _ in range(self.num_samples):
            goal_class = int(rng.randint(3))
            x, y = self._sample_xy_for_class(rng, goal_class)
            xs.append([x, y])
            ys.append(goal_class)

        x = torch.tensor(np.asarray(xs), dtype=torch.float32, device=self.device)
        y = torch.tensor(np.asarray(ys), dtype=torch.long, device=self.device)

        # Match moons preprocessing range.
        x = x * 2.0 - 1.0

        dataset = TensorDataset(x, y)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_ds, test_ds = random_split(
            dataset,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(0),
        )
        test_ds = IndexedDataset(test_ds)
        train_data_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=False)
        test_data_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)
        return train_data_loader, test_data_loader


class LinSpaceDataset(Datasets):
    def __init__(self, name, batch_size, device, num_inputs, min, max, num_samples=100):
        super().__init__(name, batch_size, device)
        self.min = min
        self.max = max
        self.num_inputs = num_inputs
        self.num_samples = num_samples

    def build(self):
        grid = torch.linspace(self.min, self.max, self.num_samples, device=self.device)
        x = torch.zeros((self.num_samples, self.num_inputs), device=self.device)
        x[:, 0] = grid
        if self.num_inputs > 1:
            x[:, 1] = torch.sin(grid)
        labels = torch.zeros(self.num_samples, dtype=torch.long, device=self.device)
        dataset = TensorDataset(x, labels)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

    def build_mesh(self, xmin, xmax, ymin, ymax):
        x = torch.linspace(xmin, xmax, self.num_samples, device = self.device)
        y = torch.linspace(ymin, ymax, self.num_samples, device = self.device)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        points = torch.stack([X.flatten(), Y.flatten()], dim=1)  # (N*N, 2)
        labels = torch.zeros(points.shape[0], dtype=torch.long, device=self.device)  # dummy labels if needed
        dataset = TensorDataset(points)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)


class DigitsDataset(Datasets):
    """Sklearn digits (8x8) dataset flattened to 64 features."""

    def __init__(self, name, batch_size, device, num_samples=None, seed=0):
        super().__init__(name, batch_size, device)
        self.num_samples = int(num_samples) if num_samples is not None else None
        self.seed = int(seed)

    def build(self):
        digits = _load_digits()
        x = digits.data  # (N, 64), values in [0, 16]
        y = digits.target
        if self.num_samples is not None and self.num_samples < x.shape[0]:
            rng = np.random.RandomState(self.seed)
            idx = rng.permutation(x.shape[0])[: self.num_samples]
            x = x[idx]
            y = y[idx]
        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        y = torch.tensor(y, dtype=torch.long, device=self.device)
        # Normalize to [-1, 1] (digits are in [0, 16]).
        x = x / 16.0 * 2.0 - 1.0

        dataset = TensorDataset(x, y)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_ds, test_ds = random_split(
            dataset,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(self.seed),
        )
        test_ds = IndexedDataset(test_ds)
        train_data_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=False)
        test_data_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)
        return train_data_loader, test_data_loader
def _build_torchvision_image_loaders(
    *,
    dataset_cls,
    batch_size,
    root,
    download,
    normalize,
    normalize_mean,
    normalize_std,
    normalize_scale=1.0,
    affine_config=None,
    shuffle_seed=None,
):
    transforms_list = [transforms.ToTensor()]
    if normalize:
        transforms_list.append(
            transforms.Normalize(
                mean=(float(normalize_mean),), std=(float(normalize_std),)
            )
        )
        scale = float(normalize_scale)
        if abs(scale - 1.0) > 1e-12:
            transforms_list.append(transforms.Lambda(lambda tensor, s=scale: tensor * s))
    transform = transforms.Compose(transforms_list)

    train_dataset = dataset_cls(
        root=root,
        train=True,
        download=download,
        transform=None if affine_config is not None else transform,
    )
    test_dataset = dataset_cls(
        root=root,
        train=False,
        download=download,
        transform=None if affine_config is not None else transform,
    )
    if affine_config is not None:
        train_dataset = DeterministicAffineImageDataset(
            train_dataset,
            transform=transform,
            affine_config=affine_config,
            split="train",
        )
        test_dataset = DeterministicAffineImageDataset(
            test_dataset,
            transform=transform,
            affine_config=affine_config,
            split="test",
        )

    shuffle_generator = (
        None
        if shuffle_seed is None
        else torch.Generator().manual_seed(int(shuffle_seed))
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=shuffle_generator,
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def build_mnist_train_validation_loaders(
    *,
    batch_size,
    root,
    download,
    normalize,
    normalize_std,
    normalize_mean=0.1307,
    normalize_scale=1.0,
    affine_config=None,
    split_seed=0,
    shuffle_seed=0,
    validation_batch_size=MNIST_LR_STUDY_VALIDATION_BATCH_SIZE,
    return_source_indices=False,
    num_workers=0,
    pin_memory=False,
):
    """Build deterministic MNIST-train loaders without touching the test split.

    This is an opt-in path for the Conv LR study.  The legacy MNIST builders keep
    returning their existing train/test loader pair.  Reusing an LR-study bundle
    across candidates requires ``reset_train_shuffle()`` before each candidate.
    """

    if int(batch_size) <= 0:
        raise ValueError(f"Expected a positive train batch size, got {batch_size}.")
    if int(validation_batch_size) <= 0:
        raise ValueError(
            "Expected a positive validation batch size, "
            f"got {validation_batch_size}."
        )
    if int(num_workers) < 0:
        raise ValueError(f"Expected a non-negative worker count, got {num_workers}.")

    transforms_list = [transforms.ToTensor()]
    if normalize:
        transforms_list.append(
            transforms.Normalize(
                mean=(float(normalize_mean),), std=(float(normalize_std),)
            )
        )
        scale = float(normalize_scale)
        if abs(scale - 1.0) > 1e-12:
            transforms_list.append(
                transforms.Lambda(lambda tensor, s=scale: tensor * s)
            )
    transform = transforms.Compose(transforms_list)

    # Deliberately instantiate only MNIST's training split.  Validation is carved
    # from it, so the official test split cannot be read accidentally here.
    full_train_dataset = _TrainOnlyMNIST(
        root=root,
        train=True,
        download=download,
        transform=None if affine_config is not None else transform,
    )
    train_indices, validation_indices = stratified_mnist_train_validation_indices(
        full_train_dataset.targets,
        split_seed=split_seed,
    )
    if affine_config is not None:
        full_train_dataset = DeterministicAffineImageDataset(
            full_train_dataset,
            transform=transform,
            affine_config=affine_config,
            split="train",
        )

    train_dataset = OriginalIndexSubset(
        full_train_dataset,
        train_indices,
        return_source_index=return_source_indices,
    )
    validation_dataset = OriginalIndexSubset(
        full_train_dataset,
        validation_indices,
        return_source_index=return_source_indices,
    )

    train_sampler = ResettableRandomSampler(train_dataset, seed=shuffle_seed)
    # Keep worker seeding separate from shuffle seeding.  DataLoader consumes its
    # generator for worker base seeds; sharing it with the sampler would perturb
    # the scientifically recorded example order.
    worker_generator = torch.Generator().manual_seed(int(shuffle_seed))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(batch_size),
        sampler=train_sampler,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        generator=worker_generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(validation_batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
    )

    first_epoch_batches = _batch_original_indices(
        train_indices,
        batch_size=int(batch_size),
        shuffle_seed=int(shuffle_seed),
        num_epochs=1,
    )[0]
    return MnistTrainValidationLoaders(
        train_loader=train_loader,
        validation_loader=validation_loader,
        train_indices=train_indices,
        validation_indices=validation_indices,
        split_seed=int(split_seed),
        shuffle_seed=int(shuffle_seed),
        batch_size=int(batch_size),
        validation_batch_size=int(validation_batch_size),
        train_indices_hash=stable_index_sequence_hash(train_indices),
        validation_indices_hash=stable_index_sequence_hash(validation_indices),
        first_epoch_batch_order_hash=stable_batch_order_hash(first_epoch_batches),
        _train_sampler=train_sampler,
        _worker_generator=worker_generator,
    )


class MnistTrainValidationDataset(Datasets):
    """Adapter exposing the deterministic 55k/5k MNIST split to trainers."""

    def __init__(
        self,
        name,
        batch_size,
        device,
        root,
        train,
        download,
        normalize,
        normalize_std,
        normalize_mean=0.1307,
        normalize_scale=1.0,
        affine_config=None,
        split_seed=0,
        shuffle_seed=0,
        validation_batch_size=MNIST_LR_STUDY_VALIDATION_BATCH_SIZE,
        num_workers=0,
        pin_memory=False,
    ):
        super().__init__(name, batch_size, device)
        self.params = {
            "batch_size": batch_size,
            "root": root,
            "download": download,
            "normalize": normalize,
            "normalize_std": normalize_std,
            "normalize_mean": normalize_mean,
            "normalize_scale": normalize_scale,
            "affine_config": affine_config,
            "split_seed": split_seed,
            "shuffle_seed": shuffle_seed,
            "validation_batch_size": validation_batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }

    def build(self):
        return build_mnist_train_validation_loaders(**self.params)


class MnistDataset(Datasets):

    def __init__(
        self,
        name,
        batch_size,
        device,
        root,
        train,
        download,
        normalize,
        normalize_std,
        normalize_mean=0.1307,
        normalize_scale=1.0,
        affine_config=None,
        shuffle_seed=None,
    ):
        super().__init__(name, batch_size, device)
        self.root = root
        self.train = train
        self.download = download
        self.normalize = normalize
        self.normalize_std = normalize_std
        self.normalize_mean = normalize_mean
        self.normalize_scale = normalize_scale
        self.affine_config = affine_config
        self.shuffle_seed = shuffle_seed

    def build(self):
        return _build_torchvision_image_loaders(
            dataset_cls=datasets.MNIST,
            batch_size=self.batch_size,
            root=self.root,
            download=self.download,
            normalize=self.normalize,
            normalize_mean=self.normalize_mean,
            normalize_std=self.normalize_std,
            normalize_scale=self.normalize_scale,
            affine_config=self.affine_config,
            shuffle_seed=self.shuffle_seed,
        )


class AffineMnistDataset(MnistDataset):
    def __init__(
        self,
        name,
        batch_size,
        device,
        root,
        train,
        download,
        normalize,
        normalize_std,
        normalize_mean=0.1307,
        normalize_scale=1.0,
        affine_config=None,
        affine_preset="medium",
        affine_seed=1729,
        affine_degrees=None,
        affine_translate=None,
        affine_scale=None,
        affine_shear=None,
        shuffle_seed=None,
    ):
        if affine_config is None:
            affine_config = affine_config_from_preset(
                affine_preset,
                degrees=affine_degrees,
                translate=affine_translate,
                scale=affine_scale,
                shear=affine_shear,
                seed=affine_seed,
            )
        super().__init__(
            name=name,
            batch_size=batch_size,
            device=device,
            root=root,
            train=train,
            download=download,
            normalize=normalize,
            normalize_std=normalize_std,
            normalize_mean=normalize_mean,
            normalize_scale=normalize_scale,
            affine_config=affine_config,
            shuffle_seed=shuffle_seed,
        )


class FashionMnistDataset(Datasets):

    def __init__(
        self,
        name,
        batch_size,
        device,
        root,
        train,
        download,
        normalize,
        normalize_std=0.3530,
        normalize_mean=0.2860,
        normalize_scale=1.0,
    ):
        super().__init__(name, batch_size, device)
        self.root = root
        self.train = train
        self.download = download
        self.normalize = normalize
        self.normalize_std = normalize_std
        self.normalize_mean = normalize_mean
        self.normalize_scale = normalize_scale

    def build(self):
        return _build_torchvision_image_loaders(
            dataset_cls=datasets.FashionMNIST,
            batch_size=self.batch_size,
            root=self.root,
            download=self.download,
            normalize=self.normalize,
            normalize_mean=self.normalize_mean,
            normalize_std=self.normalize_std,
            normalize_scale=self.normalize_scale,
        )


class TinyGridDataset(Datasets):
    """Synthetic dataset of tiny 2D grids for quick conv debugging."""

    def __init__(
        self,
        name,
        batch_size,
        device,
        patterns,
        samples_per_pattern=32,
        noise_std=0.0,
    ):
        super().__init__(name, batch_size, device)
        if not patterns:
            raise ValueError("TinyGridDataset requires at least one pattern.")
        self.patterns = patterns
        self.samples_per_pattern = samples_per_pattern
        self.noise_std = noise_std

    def build(self):
        xs = []
        ys = []

        for label, pattern in enumerate(self.patterns):
            base = torch.tensor(pattern, dtype=torch.float32)
            if base.ndim == 2:
                base = base.unsqueeze(0)  # add channel dimension
            elif base.ndim != 3:
                raise ValueError(f"Pattern must be 2D or 3D (got shape {tuple(base.shape)})")
            base = base.unsqueeze(0)  # add batch dimension → (1, C, H, W)

            for _ in range(self.samples_per_pattern):
                sample = base.clone()
                if self.noise_std > 0.0:
                    sample += self.noise_std * torch.randn_like(sample)
                xs.append(sample)
                ys.append(label)

        x_tensor = torch.cat(xs, dim=0).to(self.device)
        y_tensor = torch.tensor(ys, dtype=torch.long, device=self.device)
        dataset = TensorDataset(x_tensor, y_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
