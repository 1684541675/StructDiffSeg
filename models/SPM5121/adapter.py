"""Input-size-aware construction for the existing SPM5121 network.

The original SPM5121 model source (SPM.py, unet.py and resnet_skip.py) is
left unchanged. This adapter supplies the shape-dependent config expected by
network(), without requiring ml_collections.

The SPM self-update attention uses 8 heads, so n_patches must be divisible by
8. For square inputs with a 16-pixel patch grid, this means sizes such as
64, 128, 192, 256, ... are supported; 32, 80, 96, ... are not.
"""
from torch import nn

from .unet import network


class _ConfigNode(dict):
    """Small ConfigDict-compatible mapping with attribute access."""
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


def make_spm5121_config(image_size):
    image_size = int(image_size)
    if image_size < 64 or image_size % 16:
        raise ValueError("SPM5121 image size must be >=64 and divisible by 16")
    grid = image_size // 16
    n_patches = grid * grid
    num_heads = 8
    if n_patches % num_heads:
        raise ValueError(
            f"SPM5121 requires (image_size/16)^2 divisible by {num_heads}; "
            f"got image_size={image_size}, n_patches={n_patches}"
        )

    transformer = _ConfigNode(
        num_heads=num_heads,
        attention_dropout_rate=0.0,
        dropout_rate=0.1,
    )
    return _ConfigNode(
        n_classes=2,
        n_patches=n_patches,
        h=grid,
        w=grid,
        hidden_size=768,
        transformer=transformer,
    )


def build_spm5121(image_size):
    config = make_spm5121_config(image_size)
    return network(in_channel=3, out_channel=2, training=True, config=config)
