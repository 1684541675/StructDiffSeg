"""One-vs-rest, truncated 2D signed Euclidean distances in pixel units."""

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


@torch.no_grad()
def multiclass_sdf(mask, num_classes=4, radius=16, dtype=torch.float32):
    """Return [B,C,H,W] on mask.device, negative inside and positive outside.

    SciPy EDT runs on CPU, including a synchronous copy for CUDA inputs.
    Distances are between pixel centres; adjacent boundary pixels have distance
    +/-1 before normalization. Empty/full classes are explicitly +/-1.
    """
    if mask.ndim == 4 and mask.shape[1] == 1:
        mask = mask.squeeze(1)
    if mask.ndim != 3 or any(size == 0 for size in mask.shape):
        raise ValueError("mask must be nonempty [B,H,W] or [B,1,H,W]")
    if num_classes < 1 or not np.isfinite(radius) or radius <= 0:
        raise ValueError("num_classes and radius must be positive")
    if not dtype.is_floating_point:
        raise ValueError("SDF dtype must be floating point")
    labels = mask.detach().cpu().numpy()
    if not np.isfinite(labels).all() or not np.equal(labels, np.floor(labels)).all():
        raise ValueError("mask must contain finite integer class indices")
    if labels.min() < 0 or labels.max() >= num_classes:
        raise ValueError("mask class indices out of range")
    sdf = np.empty((labels.shape[0], num_classes, *labels.shape[1:]), dtype=np.float32)
    for b, label in enumerate(labels):
        for c in range(num_classes):
            inside = label == c
            if not inside.any():
                sdf[b, c].fill(1)
            elif inside.all():
                sdf[b, c].fill(-1)
            else:
                distance = distance_transform_edt(~inside) - distance_transform_edt(inside)
                sdf[b, c] = np.clip(distance, -radius, radius) / radius
    return torch.from_numpy(sdf).to(device=mask.device, dtype=dtype)
