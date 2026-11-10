"""Single-scale geometry-conditioned fusion for the 2D DiffUNet.

Uses the image-predicted SDF (never ground truth at inference) and an existing
geometry skip feature. Designed for an opt-in replacement of one old fusion.
"""
import torch
from torch import nn
from torch.nn import functional as F


class GeometryConditionedFusion(nn.Module):
    """Fused = F_diff + gamma * sigmoid(Conv([F_diff, H_g]) + W_t e_t) * H_g.

    H_g = phi([F_geometry, resize(predicted_SDF)]). Channel/spatial dimensions
    must match the denoiser feature after any optional batch repetition.
    """

    def __init__(self, channels: int, time_dim: int = 32, gamma_init: float = 0.1):
        super().__init__()
        if channels <= 0 or time_dim <= 0:
            raise ValueError("channels and time_dim must be positive")
        self.geometry_projection = nn.Sequential(
            nn.Conv2d(channels + 1, channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(channels, affine=True),
            nn.GELU(),
        )
        self.gate_conv = nn.Conv2d(2 * channels, channels, kernel_size=1)
        self.time_projection = nn.Linear(time_dim, channels)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    @staticmethod
    def _expand_batch(feature: torch.Tensor, batch: int, name: str) -> torch.Tensor:
        if feature.shape[0] == batch:
            return feature
        if feature.shape[0] <= 0 or batch % feature.shape[0] != 0:
            raise ValueError(f"{name} batch must divide diffusion feature batch")
        # Matches the original DiffUNet MC/DDIM repetition order: (r b).
        return feature.repeat(batch // feature.shape[0], 1, 1, 1)

    def forward(self, diffusion_feature: torch.Tensor, geometry_feature: torch.Tensor,
                predicted_sdf: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        if diffusion_feature.ndim != 4 or geometry_feature.ndim != 4:
            raise ValueError("diffusion and geometry features must be BCHW tensors")
        if predicted_sdf.ndim != 4 or predicted_sdf.shape[1] != 1:
            raise ValueError("predicted_sdf must have shape [B,1,H,W]")
        batch, channels, height, width = diffusion_feature.shape
        if geometry_feature.shape[1] != channels or geometry_feature.shape[2:] != (height, width):
            raise ValueError("geometry feature channels and spatial shape must match diffusion feature")
        if time_embedding.ndim != 2 or time_embedding.shape[0] != batch:
            raise ValueError("time_embedding must have shape [B,time_dim]")

        geometry_feature = self._expand_batch(geometry_feature, batch, "geometry")
        predicted_sdf = self._expand_batch(predicted_sdf, batch, "SDF")
        sdf_scaled = F.interpolate(
            predicted_sdf.to(dtype=geometry_feature.dtype),
            size=(height, width), mode="bilinear", align_corners=False
        )
        geometry_condition = self.geometry_projection(
            torch.cat((geometry_feature, sdf_scaled), dim=1)
        )
        time_bias = self.time_projection(time_embedding).unsqueeze(-1).unsqueeze(-1)
        gate = torch.sigmoid(
            self.gate_conv(torch.cat((diffusion_feature, geometry_condition), dim=1))
            + time_bias
        )
        return diffusion_feature + self.gamma * gate * geometry_condition
