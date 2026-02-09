from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import DataLoader, TensorDataset


def compute_pca_from_loader(train_loader, *, device: Optional[torch.device] = None):
    """
    Compute top-2 PCA components from a loader yielding image tensors.

    Returns:
        mu, eigvals_2, eigvecs_2, n_total
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sum_x = None
    sum_xx_t = None
    n_total = 0

    with torch.no_grad():
        for x, _ in train_loader:
            x = x.to(device)
            batch = x.shape[0]
            x_flat = x.view(batch, -1)

            if sum_x is None:
                dim = x_flat.shape[1]
                sum_x = torch.zeros(dim, device=device)
                sum_xx_t = torch.zeros(dim, dim, device=device)

            sum_x += x_flat.sum(dim=0)
            sum_xx_t += x_flat.T @ x_flat
            n_total += batch

    mu = sum_x / n_total
    mu_col = mu.unsqueeze(1)
    cov = (sum_xx_t - n_total * (mu_col @ mu_col.T)) / (n_total - 1)

    eigvals, eigvecs = torch.linalg.eigh(cov)
    idx = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    eigvals_2 = eigvals[:2]
    eigvecs_2 = eigvecs[:, :2]
    return mu, eigvals_2, eigvecs_2, n_total


def prepare_pca_grid(
    eigvals,
    eigvecs,
    mu,
    n_total=None,
    *,
    steps: int = 30,
    n_sigma: float = 3.0,
    image_shape=(1, 28, 28),
    batch_size: int = 1,
):
    # Keep the grid on CPU (DataLoader + CUDA tensors is painful); model moves it to GPU later.
    mu = mu.detach().flatten().cpu()
    eigvals = eigvals.detach().flatten().cpu()[:2]
    eigvecs = eigvecs.detach().cpu()

    if eigvecs.shape[0] != mu.numel():
        eigvecs = eigvecs.T
    eigvecs = eigvecs[:, :2]

    sigma = eigvals.clamp_min(0).sqrt()
    a = torch.linspace(-n_sigma * sigma[0], n_sigma * sigma[0], steps)
    b = torch.linspace(-n_sigma * sigma[1], n_sigma * sigma[1], steps)
    aa, bb = torch.meshgrid(a, b, indexing="ij")
    points = torch.stack([aa.flatten(), bb.flatten()], dim=1)

    x_flat = mu.unsqueeze(0) + points @ eigvecs.T
    x = x_flat.view(points.shape[0], *image_shape)
    y = torch.zeros(points.shape[0], dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)

