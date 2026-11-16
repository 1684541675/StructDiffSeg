"""Adapter for the existing 2D UNETR++ implementation in this folder.

The model implementation itself is kept unchanged. The original training
configuration in models/UNETRPP256/train.py is retained:
feature_size=16, hidden_size=256, num_heads=4, depths=[3,3,3,3],
dims=[32,64,128,256], do_ds=False.

This local model is 2D (Conv2d), despite the upstream UNETR++ project being
primarily a 3D segmentation framework. Its encoder has a total downsampling
factor of 32, so input dimensions must be divisible by 32. For batch-size-one
training with the BatchNorm blocks, use at least 64x64.
"""
from torch import nn

try:
    from .unetr_plus_plus.unetr_pp.network_architecture.tumor.unetr_pp_tumor import (
        UNETR_PP,
    )
except ModuleNotFoundError as exc:
    required = {"monai", "timm", "einops", "batchgenerators", "scipy"}
    if exc.name in required:
        raise ModuleNotFoundError(
            "UNETRPP256 requires the existing environment dependencies "
            "monai, timm, einops, batchgenerators and scipy. "
            "Check environment_record/requirements_local_windows.txt."
        ) from exc
    raise


def build_unetrpp256(image_size):
    image_size = int(image_size)
    if image_size < 64:
        raise ValueError("UNETRPP256 image size must be at least 64")
    if image_size % 32:
        raise ValueError("UNETRPP256 image size must be divisible by 32")

    return UNETR_PP(
        in_channels=3,
        out_channels=2,
        img_size=image_size,
        feature_size=16,
        hidden_size=256,
        num_heads=4,
        depths=[3, 3, 3, 3],
        dims=[32, 64, 128, 256],
        do_ds=False,
    )
