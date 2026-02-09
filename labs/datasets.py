from abc import ABC, abstractmethod
from sklearn.datasets import make_moons
from torchvision import datasets, transforms
import torch
from torch.utils.data import random_split, DataLoader, TensorDataset



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
    def __init__(self, name, batch_size, device, num_samples):
        super().__init__(name, batch_size, device)
        self.num_samples = num_samples
        self.noise = 0.1
    def build(self):
        x, y = make_moons(n_samples=self.num_samples, noise=self.noise, random_state=42)
        # Keep features in float32 to match model weights; labels stay int for targets.
        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        y = torch.tensor(y, dtype=torch.long, device=self.device)
        # Normalize each feature to [-10, 10] based on global min/max.
        eps = 1e-12
        x_min = x.min(dim=0).values
        x_max = x.max(dim=0).values
        scale = (x_max - x_min).clamp_min(eps)
        x = (x - x_min) / scale * 20.0 - 10.0
        dataset = TensorDataset(x, y)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_ds, test_ds = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(0))
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
class MnistDataset(Datasets):

    def __init__(self, name, batch_size, device, root, train, download, normalize, normalize_std):
        super().__init__(name, batch_size, device)
        self.root = root
        self.train = train
        self.download = download
        self.normalize = normalize
        self.normalize_std = normalize_std

    def build(self):
        transforms_list = [transforms.ToTensor()]
        if self.normalize:
            transforms_list.append(
                transforms.Normalize(
                    mean=(0.1307,), std=(self.normalize_std,)
                )
            )
        transform = transforms.Compose(transforms_list)

        train_dataset = datasets.MNIST(
            root=self.root,
            train=True,
            download=self.download,
            transform=transform,
        )
        test_dataset = datasets.MNIST(
            root=self.root,
            train=False,
            download=self.download,
            transform=transform,
        )

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader, test_loader


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
