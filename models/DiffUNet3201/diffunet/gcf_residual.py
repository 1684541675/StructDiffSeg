"""Additive SDF conditioning that preserves every existing fusion operation.

The original SDF model is composed with a new encoder wrapper; legacy classes
and files are not patched. New checkpoints require the new entry points.
"""
import torch
from torch import nn

from .encoder_denoise import affinity_fusion
from .gcf import GeometryConditionedFusion


VARIANTS = ("gcf_residual", "gcf_residual_no_sdf", "gcf_residual_no_time")


class ResidualGeometryConditioning(GeometryConditionedFusion):
    """F_out = F_old_fusion + gamma * gate(F_old_fusion, geometry, t) * H_g.

    Gamma starts at zero: the initial forward is exactly the SDF baseline.
    Its first update opens the residual path; other residual parameters receive
    useful gradients after gamma becomes nonzero. SDF supervision is active
    from the first step. Ablations zero inputs but keep parameter counts equal.
    """

    def __init__(self, channels, *, use_sdf=True, use_time=True):
        super().__init__(channels, time_dim=32, gamma_init=0.0)
        self.use_sdf = use_sdf
        self.use_time = use_time

    def forward(self, fused, geometry, sdf, time_embedding):
        if sdf is None:
            raise ValueError("Predicted SDF is required, never a ground-truth SDF")
        if not self.use_sdf:
            sdf = torch.zeros_like(sdf)
        if not self.use_time:
            time_embedding = torch.zeros_like(time_embedding)
        return super().forward(fused, geometry, sdf, time_embedding)


class ResidualFusionEncoder(nn.Module):
    """Keep original concatenation transforms AND bottleneck affinity fusion."""

    def __init__(self, original, *, stage=2, use_sdf=True, use_time=True):
        super().__init__()
        if original.gcf is not None or not original.bta:
            raise ValueError("Expected an unmodified BTA encoder without legacy GCF")
        if not 0 <= stage < len(original.stages) - 1:
            raise ValueError("Residual stage must precede the bottleneck")
        self.original = original
        self.stage = stage
        self.residual = ResidualGeometryConditioning(
            original.output_channels[stage], use_sdf=use_sdf, use_time=use_time
        )

    def forward(self, x, temb, embeddings=None, sdf=None):
        if embeddings is None or sdf is None:
            raise ValueError("Residual fusion requires geometry features and predicted SDF")
        skips = []
        for index, layer in enumerate(self.original.stages):
            x = layer([x, temb])
            geometry = GeometryConditionedFusion._expand_batch(
                embeddings[index], x.shape[0], "geometry"
            )
            if index == len(self.original.stages) - 1:
                x = affinity_fusion(x, geometry)
            else:
                # Crucially, execute the old fusion BEFORE the new residual.
                x = self.original.transforms[index](torch.cat((x, geometry), dim=1))
            if index == self.stage:
                x = self.residual(x, geometry, sdf, temb)
            skips.append(x)
        return skips if self.original.return_skips else skips[-1]


def build_model(variant="gcf_residual", ddim_steps=3, shared_seed=None):
    """Reuse legacy shared initialization, including the same SDF head weights."""
    from train_sdf_2d import build_model as build_legacy

    if variant not in VARIANTS:
        raise ValueError(f"Expected one of {VARIANTS}, got {variant!r}")
    model = build_legacy("sdf", ddim_steps, shared_seed=shared_seed)

    def attach():
        encoder = ResidualFusionEncoder(
            model.denoise_model.encoder,
            use_sdf=variant != "gcf_residual_no_sdf",
            use_time=variant != "gcf_residual_no_time",
        )
        model.denoise_model.encoder = encoder
        # Decoder registers an alias to the encoder: keep it the SAME object.
        model.denoise_model.decoder.encoder = encoder
        # Existing DiffUNet.forward now supplies the image-predicted SDF in
        # both training and label-free DDIM inference.
        model.enable_gcf = True

    if shared_seed is None:
        attach()
    else:
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(shared_seed + 2)
            attach()
    return model
