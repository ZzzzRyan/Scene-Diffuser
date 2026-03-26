import os

import torch


def _nearest_neighbor_distance(
    source: torch.Tensor,
    target: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute squared nearest-neighbor distances from `source` to `target`."""
    if source.ndim != 3 or target.ndim != 3:
        raise ValueError("ChamferDistance expects tensors with shape (B, N, C).")
    if source.shape[0] != target.shape[0]:
        raise ValueError("ChamferDistance expects matching batch dimensions.")
    if source.shape[-1] != target.shape[-1]:
        raise ValueError("ChamferDistance expects matching point dimensions.")
    if target.shape[1] == 0:
        raise ValueError("ChamferDistance target point cloud must be non-empty.")

    batch_size, num_source, _ = source.shape
    best_dist = torch.full(
        (batch_size, num_source),
        torch.inf,
        device=source.device,
        dtype=source.dtype,
    )
    best_idx = torch.zeros(
        (batch_size, num_source),
        device=source.device,
        dtype=torch.long,
    )

    for start in range(0, target.shape[1], chunk_size):
        end = min(start + chunk_size, target.shape[1])
        # Match the legacy otaheri/pytorch3d-based API, which returns squared L2 distances.
        chunk_dist = torch.cdist(source, target[:, start:end, :], p=2).square()
        chunk_best_dist, chunk_best_idx = torch.min(chunk_dist, dim=2)
        update_mask = chunk_best_dist < best_dist
        best_dist = torch.where(update_mask, chunk_best_dist, best_dist)
        best_idx = torch.where(update_mask, chunk_best_idx + start, best_idx)

    return best_dist, best_idx


def ChamferDistance(
    xyz1: torch.Tensor,
    xyz2: torch.Tensor,
    *,
    chunk_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """PyTorch fallback for the legacy `chamfer_distance` API used in this repo.

    Returns `(dist1, dist2, idx1, idx2)` with squared L2 nearest-neighbor distances,
    matching the original dependency's `knn_points(...).dists` behavior.
    """
    if chunk_size is None:
        chunk_size = int(os.environ.get("SCENEDIFFUSER_CHAMFER_CHUNK", "512"))

    dist1, idx1 = _nearest_neighbor_distance(xyz1, xyz2, chunk_size)
    dist2, idx2 = _nearest_neighbor_distance(xyz2, xyz1, chunk_size)
    return dist1, dist2, idx1, idx2
