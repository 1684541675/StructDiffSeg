"""Binary 2D signed-distance supervision targets.

Foreground is positive, background is negative. Distances are measured in
resized/augmented image pixels and clipped to [-1, 1]. This utility is for
ground-truth supervision; it does not alter diffusion-label embeddings.
"""

import math

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


@torch.no_grad()
def binary_sdf_target(mask: torch.Tensor, radius: float = 16.0,
                      dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Return [B,1,H,W] signed Euclidean distances from binary masks.

    Args:
        mask: [B,H,W] or [B,1,H,W] containing only 0 (background) and
            1 (foreground). Compute after spatial augmentation / resizing.
        radius: Positive clipping radius in pixels, not physical units.
        dtype: Floating-point output tensor dtype.

    Convention: foreground positive, background negative, boundary-adjacent
    pixel-center values +/-1/radius. All-background masks return -1 and
    all-foreground masks +1. EDT runs on CPU; CUDA inputs are copied to CPU
    before calculation and the result is copied back to mask.device.
    """
    if not isinstance(mask, torch.Tensor):
        raise TypeError("mask must be a torch.Tensor")
    if mask.ndim == 4 and mask.shape[1] == 1:
        mask = mask[:, 0]
    if mask.ndim != 3 or any(dim == 0 for dim in mask.shape):
        raise ValueError("mask must be nonempty [B,H,W] or [B,1,H,W]")
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be finite and positive")
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise ValueError("dtype must be a floating-point torch dtype")

    labels = mask.detach().cpu().numpy()
    if not np.isfinite(labels).all() or not np.isin(labels, (0, 1)).all():
        raise ValueError("mask must contain only finite binary values 0 and 1")

    targets = np.empty((labels.shape[0], 1, *labels.shape[1:]), dtype=np.float32)
    for batch_index, label in enumerate(labels):
        foreground = label == 1
        if not foreground.any():
            targets[batch_index, 0].fill(-1.0)
        elif foreground.all():
            targets[batch_index, 0].fill(1.0)
        else:
            # Positive inside the object, negative outside.
            distance = (distance_transform_edt(foreground)
                        - distance_transform_edt(~foreground))
            targets[batch_index, 0] = np.clip(distance, -radius, radius) / radius

    return torch.from_numpy(targets).to(device=mask.device, dtype=dtype)
