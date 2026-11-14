import torch
from torch import nn
import torch.nn.functional as F
import time

#---------------------------------- Basic Building Blocks ----------------------------------#

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, 
            stride=stride, padding=padding, groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ECB(nn.Module):
    """Efficient Convolution Block (ECB) - DSConv + BN + SiLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = DepthwiseSeparableConv(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
        
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


#---------------------------------- Rolling-Convolution (R-Conv) ----------------------------------#

class RollingConv(nn.Module):
    """Rolling-Convolution (R-Conv) - Efficient multi-orientation feature extraction using grouped convolution"""
    def __init__(self, channels):
        super().__init__()
        self.conv_group = nn.Conv2d(
            channels, channels * 4, 3, padding=1, 
            groups=channels, bias=False
        )
        self.bn = nn.BatchNorm2d(channels)
        
        # Initialize kernels for 4 rotation angles (0°, 90°, 180°, 270°)
        self._init_rotated_kernels()
        
    def _init_rotated_kernels(self):
        with torch.no_grad():
            for i in range(self.conv_group.in_channels):
                base_kernel = torch.randn(3, 3) * 0.1
                self.conv_group.weight[i*4 + 0, 0] = base_kernel
                self.conv_group.weight[i*4 + 1, 0] = torch.rot90(base_kernel, 1)
                self.conv_group.weight[i*4 + 2, 0] = torch.rot90(base_kernel, 2)
                self.conv_group.weight[i*4 + 3, 0] = torch.rot90(base_kernel, 3)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        x_all = self.conv_group(x) 
        
        x_all = x_all.view(B, C, 4, H, W)
        fused = torch.max(x_all, dim=2)[0]
        
        return self.bn(fused)


#---------------------------------- RGDS Internal Components ----------------------------------#

class RenormalizationGroupConv(nn.Module):
    """RG transformation inspired by renormalization group theory
    Implements discrete RG flow equation for scale transformation"""
    def __init__(self, channels):
        super().__init__()
        
        # Fused network computes both beta and gamma functions
        self.rg_transform = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels * 2, 1, bias=False),
            nn.BatchNorm2d(channels * 2)
        )
        
        # Learnable RG flow step size
        self.scale_step = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, x):
        """Simplified RG flow equation update"""
        rg_feat = self.rg_transform(x)
        beta, gamma = rg_feat.chunk(2, dim=1)
        
        beta = torch.tanh(beta)
        gamma = torch.sigmoid(gamma)
        
        x_new = x + self.scale_step * (beta + gamma * x)
        return x_new


class CriticalPointDetection(nn.Module):
    def __init__(self, channels):
        super().__init__()
        
        self.hessian_approx = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels // 4, 1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True)
        )
        
        self.criticality = nn.Sequential(
            nn.Conv2d(channels // 4, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        hessian_feat = self.hessian_approx(x)
        criticality_map = self.criticality(hessian_feat)
        
        return x * (1 + criticality_map)


class RGDS(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.rg_critical_fused = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Tanh()
        )
        
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
        
        self.high_freq_path = nn.Sequential(
            nn.AvgPool2d(2, 2),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Learnable fusion weight (λ in paper)
        self.fusion_weight = nn.Parameter(torch.tensor(0.3))
        
    def forward(self, x):
        x_enhanced = x + 0.1 * self.rg_critical_fused(x)
        
        x_low = self.downsample(x_enhanced)
        x_high = self.high_freq_path(x_enhanced)

        return x_low + self.fusion_weight * x_high


#---------------------------------- Enhanced Symplectic Convolution (ESConv) ----------------------------------#

class GradientAugmentation(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.fused_gradient = nn.Conv2d(
            channels, channels * 2, 3, padding=1, 
            groups=channels, bias=False
        )

        self._init_sobel_kernels()
        
        self.gradient_enhance = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels)
        )
    
    def _init_sobel_kernels(self):
        with torch.no_grad():
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                   dtype=torch.float32) / 4.0
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                   dtype=torch.float32) / 4.0
            for i in range(self.fused_gradient.in_channels):
                self.fused_gradient.weight[i*2, 0] = sobel_x
                self.fused_gradient.weight[i*2 + 1, 0] = sobel_y
    
    def forward(self, x):
        grad_xy = self.fused_gradient(x)
        grad_x, grad_y = grad_xy.chunk(2, dim=1)
        gradient_magnitude = (grad_x.abs() + grad_y.abs()) * 0.707
        gradient_features = self.gradient_enhance(gradient_magnitude)
        return torch.cat([x, gradient_features], dim=1)


class SimplifiedSymplecticCoupling(nn.Module):
    """Symplectic coupling from Hamiltonian mechanics
    Implements structure-preserving transformations on position-momentum pairs"""
    def __init__(self, channels):
        super().__init__()
        assert channels % 2 == 0

        self.coupling = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Tanh()
        )

        self.coupling_strength = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, x):
        q, p = x.chunk(2, dim=1) 
        coupled = self.coupling(x)
        delta_q, delta_p = coupled.chunk(2, dim=1)
        
        # Hamiltonian-inspired coupled updates
        q_new = q + self.coupling_strength * delta_p
        p_new = p - self.coupling_strength * delta_q
        
        return torch.cat([q_new, p_new], dim=1)


class BoundaryExtraction(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.boundary_detector = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.GELU(),
            nn.Conv2d(in_channels // 4, 1, 1),
            nn.Sigmoid()
        )
        
        self.boundary_features = nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        
    def forward(self, x):
        boundary_map = self.boundary_detector(x)
        features = self.boundary_features(x)
        boundary_enhanced = features * (1 + boundary_map)
        return boundary_enhanced


class ESConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        
        # Main patH
        self.main_path = nn.Sequential(
            DepthwiseSeparableConv(channels, channels, kernel_size=3),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )
        
        # Symplectic boundary path
        self.symplectic_boundary_module = nn.Sequential(
            GradientAugmentation(channels),
            SimplifiedSymplecticCoupling(channels * 2),
            BoundaryExtraction(channels * 2, channels)
        )
        
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )
        
        # Learnable residual weight (β in paper)
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x):
        f_main = self.main_path(x)
        f_boundary = self.symplectic_boundary_module(x)
        fused = self.fusion(torch.cat([f_main, f_boundary], dim=1))
        output = fused + self.residual_weight * x
        return output


#---------------------------------- Pseudo-Global Receptive Field (PGRF) ----------------------------------#

class PGRF(nn.Module):
    """Pseudo-Global Receptive Field (PGRF) via entropy-driven diffusion"""
    def __init__(self, channels, propagation_steps=5):
        super().__init__()
        self.channels = channels
        self.propagation_steps = propagation_steps
        
        self.entropy_propagator = EntropyDiffusionPath(channels, propagation_steps)
        
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1, bias=False)
        )
        
    def forward(self, x):
        global_info = self.global_context(x)
        global_info = global_info.expand_as(x)
        
        # Entropy-driven diffusion with global context injection
        entropy_field = self.entropy_propagator(x, global_info)
        
        return entropy_field + x


class EntropyDiffusionPath(nn.Module):
    def __init__(self, channels, steps=3):
        super().__init__()
        self.steps = steps
        
        self.diffusion_kernels = nn.ModuleList([
            nn.Sequential(
                DepthwiseSeparableConv(channels, channels, kernel_size=3),
                nn.BatchNorm2d(channels),
                nn.SiLU(inplace=True)
            ) for _ in range(steps)
        ])
        
        self.diffusion_modulator = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels // 8, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x, global_info):
        mean_x = F.avg_pool2d(x, 3, stride=1, padding=1)
        variance = F.avg_pool2d((x - mean_x) ** 2, 3, stride=1, padding=1)
        diffusion_rate = self.diffusion_modulator(variance)
        
        u = x
        for step, diffusion_kernel in enumerate(self.diffusion_kernels):
            laplacian = diffusion_kernel(u) - u 
            u = u + diffusion_rate * laplacian * (0.5 ** step) 
            u = u + global_info * 0.1 
        
        return u


#---------------------------------- Core Modules ----------------------------------#

class LMB(nn.Module):
    """Lightweight Memory Bank (LMB)"""
    def __init__(self, channels, num_prototypes=16):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, channels))
        
        reduced_dim = max(channels // 4, 16)
        self.query_proj = nn.Conv2d(channels, reduced_dim, 1, bias=False)
        self.value_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.1))  # Residual weight
        
    def forward(self, x):
        B, C, H, W = x.shape
        q = self.query_proj(x).view(B, -1, H*W).permute(0, 2, 1)
        q_norm = F.normalize(q, dim=2)

        proto_reduced = self.prototypes[:, :q.size(2)]
        proto_norm = F.normalize(proto_reduced, dim=1)

        similarity = torch.matmul(q_norm, proto_norm.t())
        attn = F.softmax(similarity * 10, dim=-1)
        retrieved = torch.matmul(attn, self.prototypes)
        retrieved = retrieved.permute(0, 2, 1).view(B, C, H, W)
        
        return x + self.alpha * self.value_proj(retrieved)


class AS(nn.Module):
    """Adaptive Scale Selector (AS)"""
    def __init__(self, channels):
        super().__init__()
        self.scale_small = RollingConv(channels)
        self.scale_medium = RollingConv(channels)
        self.scale_large = RollingConv(channels)
        
        self.scale_predictor = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1, bias=False),
            nn.BatchNorm2d(channels // 8),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels // 8, 3, 1, bias=False),
            nn.Softmax(dim=1)
        )
        
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )
        
    def forward(self, x):
        scale_weights = self.scale_predictor(x)
        f_small = self.scale_small(x) * scale_weights[:, 0:1]
        f_medium = self.scale_medium(x) * scale_weights[:, 1:2]
        f_large = self.scale_large(x) * scale_weights[:, 2:3]
        fused = f_small + f_medium + f_large
        return F.silu(self.fusion(fused) + x)


class PR(nn.Module):
    """Progressive Refinement Unit (PR)"""
    def __init__(self, channels, num_iterations=3):
        super().__init__()
        self.num_iterations = num_iterations

        self.refine_conv = nn.ModuleList([
            nn.Sequential(
                DepthwiseSeparableConv(channels, channels, kernel_size=3),
                nn.BatchNorm2d(channels),
                nn.SiLU(inplace=True),
                DepthwiseSeparableConv(channels, channels, kernel_size=3),
                nn.BatchNorm2d(channels)
            ) for _ in range(num_iterations)
        ])

        self.attention = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, channels // 16, 1, bias=False),
                nn.SiLU(inplace=True),
                nn.Conv2d(channels // 16, channels, 1, bias=False),
                nn.Sigmoid()
            ) for _ in range(num_iterations)
        ])
        
    def forward(self, x):
        for refine_block, attn_block in zip(self.refine_conv, self.attention):
            residual = refine_block(x)
            attn = attn_block(x)
            x = x + residual * attn
        return x


class BF(nn.Module):
    """Boundary-aware Fusion (BF)"""
    def __init__(self, channels):
        super().__init__()
        self.boundary_detector = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1, bias=False),
            nn.BatchNorm2d(channels // 8),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels // 8, 1, 3, padding=1, bias=False),
            nn.Sigmoid()
        )
        
        self.region_fusion = nn.Sequential(
            DepthwiseSeparableConv(channels * 2, channels, kernel_size=3),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )
        
        self.boundary_enhance = nn.Sequential(
            DepthwiseSeparableConv(channels, channels, kernel_size=3),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )
        
    def forward(self, dec_feat, enc_feat):
        if dec_feat.shape[2:] != enc_feat.shape[2:]:
            dec_feat = F.interpolate(
                dec_feat, size=enc_feat.shape[2:], 
                mode='bilinear', align_corners=False
            )

        boundary_map = self.boundary_detector(enc_feat)
        region_feat = self.region_fusion(torch.cat([dec_feat, enc_feat], dim=1))
        boundary_feat = self.boundary_enhance(enc_feat)
        output = region_feat * (1 - boundary_map) + boundary_feat * boundary_map
        
        return output


#---------------------------------- Main Network: PMRNet ----------------------------------#

class PMRNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super().__init__()
        
        chs = [24, 48, 96, 192]
        
        # Stem
        self.stem = nn.Sequential(
            DepthwiseSeparableConv(in_channels, chs[0], kernel_size=3),
            nn.BatchNorm2d(chs[0]),
            nn.SiLU(inplace=True)
        )
        
        # Encoder Stage 1: ESConv + AS + LMB
        self.enc1 = nn.Sequential(
            ESConv(chs[0]),
            AS(chs[0]),
            LMB(chs[0], num_prototypes=8)
        )
        self.down1 = RGDS(chs[0], chs[1])
        
        # Encoder Stage 2: AS + PR(T=2) + LMB
        self.enc2 = nn.Sequential(
            AS(chs[1]),
            PR(chs[1], num_iterations=2),
            LMB(chs[1], num_prototypes=12)
        )
        self.down2 = RGDS(chs[1], chs[2])
        
        # Encoder Stage 3: AS + PR(T=3) + LMB
        self.enc3 = nn.Sequential(
            AS(chs[2]),
            PR(chs[2], num_iterations=3),
            LMB(chs[2], num_prototypes=16)
        )
        self.down3 = RGDS(chs[2], chs[3])
        
        # Bottleneck: PR + PGRF
        self.bottleneck = nn.Sequential(
            PR(chs[3], num_iterations=3),
            PGRF(chs[3], propagation_steps=3)
        )
        
        # Decoder Stage 3
        self.dec3 = nn.Conv2d(chs[3], chs[2], 1, bias=False)
        self.fuse3 = BF(chs[2])
        self.refine3 = ECB(chs[2], chs[2])
        
        # Decoder Stage 2
        self.dec2 = nn.Conv2d(chs[2], chs[1], 1, bias=False)
        self.fuse2 = BF(chs[1])
        self.refine2 = ECB(chs[1], chs[1])
        
        # Decoder Stage 1
        self.dec1 = nn.Conv2d(chs[1], chs[0], 1, bias=False)
        self.fuse1 = BF(chs[0])
        self.refine1 = ECB(chs[0], chs[0])
        
        # Output head
        self.final = nn.Sequential(
            ECB(chs[0], chs[0]),
            nn.Dropout2d(0.1),
            nn.Conv2d(chs[0], num_classes, 1, bias=True)
        )
        
        self._init_weights()
    
    
    def forward(self, x):
        x0 = self.stem(x)
        e1 = self.enc1(x0)
        
        x1 = self.down1(e1)
        e2 = self.enc2(x1)
        
        x2 = self.down2(e2)
        e3 = self.enc3(x2)
        
        x3 = self.down3(e3)
        bottleneck = self.bottleneck(x3)

        d3 = F.interpolate(self.dec3(bottleneck), scale_factor=2, mode='bilinear', align_corners=False)
        d3 = self.fuse3(d3, e3)
        d3 = self.refine3(d3)
        
        d2 = F.interpolate(self.dec2(d3), scale_factor=2, mode='bilinear', align_corners=False)
        d2 = self.fuse2(d2, e2)
        d2 = self.refine2(d2)
        
        d1 = F.interpolate(self.dec1(d2), scale_factor=2, mode='bilinear', align_corners=False)
        d1 = self.fuse1(d1, e1)
        d1 = self.refine1(d1)
        
        return self.final(d1)

    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                