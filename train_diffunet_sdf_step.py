"""One supervised update for binary DiffUNet + SDF; not a training CLI.

This reuses the original DiffUNet's CE/Dice auxiliary and uncertainty-weighted
diffusion losses and adds lambda_sdf * SmoothL1(predicted_sdf, target_sdf).
GCF and real-dataset loading are intentionally outside this step.
"""

import math

import torch

from models.DiffUNet3201.diffunet.sdf_loss import sdf_smooth_l1_loss
from models.DiffUNet3201.diffunet.sdf_target import binary_sdf_target


def train_step_sdf(batch, model, criterion, optimizer, *,
                   epoch=1, epochs=100, lambda_sdf=0.25,
                   sdf_radius=16.0, beta=1.0):
    """Run exactly one optimizer update on a binary RGB batch.

    batch: {'image': [B,3,H,W], 'label': [B,H,W]}, binary label values 0/1.
    model: DiffUNet(..., out_channels=2, enable_sdf=True).
    criterion: original CeDiceLoss(2, [1, 1]) or compatible interface.
    Return detached scalar diagnostics, not tensors that retain a graph.

    This function deliberately does not load datasets, change schedulers,
    checkpoint models, or run multi-epoch training.
    """
    if not isinstance(epoch, int) or not isinstance(epochs, int) or not (1 <= epoch <= epochs):
        raise ValueError("epoch and epochs must satisfy 1 <= epoch <= epochs")
    if not math.isfinite(lambda_sdf) or lambda_sdf < 0:
        raise ValueError("lambda_sdf must be finite and nonnegative")
    if not isinstance(batch, dict) or "image" not in batch or "label" not in batch:
        raise ValueError("batch requires image and label tensors")

    images, labels = batch["image"], batch["label"]
    if not isinstance(images, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise TypeError("image and label must be torch.Tensor")
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("image must have shape [B,3,H,W]")
    if labels.ndim != 3 or labels.shape != (images.shape[0], *images.shape[2:]):
        raise ValueError("label must have shape [B,H,W] matching image")
    if not torch.isfinite(images).all().item():
        raise ValueError("image must contain only finite values")
    if not torch.isfinite(labels).all().item() or not ((labels == 0) | (labels == 1)).all().item():
        raise ValueError("label must contain only binary values 0 and 1")

    device = next(model.parameters()).device
    images = images.to(device=device, dtype=torch.float32)
    labels = labels.to(device=device, dtype=torch.long)
    # Ground truth is built after spatial augmentation and uses no gradients.
    target_sdf = binary_sdf_target(labels, radius=sdf_radius, dtype=images.dtype)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    pred_diff, pred_geo, uncertainty, pred_sdf = model(
        images, gt=labels, return_sdf=True
    )
    if pred_sdf.shape != target_sdf.shape:
        raise ValueError("predicted SDF and target SDF shapes disagree")

    # Preserve the original uncertainty schedule used by engine_synapse.func.
    uncertainty_scale = math.exp(-10.0 * (1.0 - epoch / epochs) ** 2)
    loss_diff = criterion(pred_diff, labels, uncertainty.clamp(0, 1), uncertainty_scale)
    loss_geo = criterion(pred_geo, labels)
    loss_sdf = sdf_smooth_l1_loss(pred_sdf, target_sdf, beta=beta)
    total_loss = loss_diff + loss_geo + lambda_sdf * loss_sdf

    if not torch.isfinite(total_loss).item():
        raise FloatingPointError("Non-finite total training loss")
    total_loss.backward()

    sdf_head = model.edge_model.sdf_head
    if sdf_head is None or sdf_head[0].weight.grad is None:
        raise RuntimeError("SDF head must be enabled and receive gradients")
    denoiser_param = next(model.denoise_model.parameters())
    if denoiser_param.grad is None:
        raise RuntimeError("Denoising branch did not receive gradients")
    grad_sdf = sdf_head[0].weight.grad.norm().item()
    grad_diff = denoiser_param.grad.norm().item()

    optimizer.step()
    return {
        "loss_total": total_loss.detach().item(),
        "loss_diffusion": loss_diff.detach().item(),
        "loss_geometry": loss_geo.detach().item(),
        "loss_sdf": loss_sdf.detach().item(),
        "lambda_sdf": float(lambda_sdf),
        "grad_sdf_head": grad_sdf,
        "grad_denoiser": grad_diff,
    }
