"""Shape-checked Smooth L1 loss for supervised binary signed distance fields."""

import math

import torch
import torch.nn.functional as F


def sdf_smooth_l1_loss(predicted_sdf: torch.Tensor, target_sdf: torch.Tensor,
                       beta: float = 1.0) -> torch.Tensor:
    """Compare [B,1,H,W] predicted/target SDF; return a scalar mean loss.

    Targets can be produced by binary_sdf_target() from augmented binary masks.
    The function deliberately accepts SDF targets rather than masks so future
    data loaders may precompute EDT on CPU, avoiding per-step GPU synchronization.
    """
    if not isinstance(predicted_sdf, torch.Tensor) or not isinstance(target_sdf, torch.Tensor):
        raise TypeError("predicted_sdf and target_sdf must be torch.Tensor")
    if predicted_sdf.ndim != 4 or predicted_sdf.shape[1] != 1:
        raise ValueError("predicted_sdf must have shape [B,1,H,W]")
    if predicted_sdf.shape != target_sdf.shape:
        raise ValueError("predicted_sdf and target_sdf must have identical shapes")
    if predicted_sdf.device != target_sdf.device:
        raise ValueError("predicted_sdf and target_sdf must be on the same device")
    if not predicted_sdf.is_floating_point() or not target_sdf.is_floating_point():
        raise TypeError("predicted_sdf and target_sdf must be floating point")
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")

    return F.smooth_l1_loss(predicted_sdf, target_sdf.to(predicted_sdf.dtype),
                            beta=beta, reduction="mean")
