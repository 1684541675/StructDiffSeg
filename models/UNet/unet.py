"""A conventional 2D U-Net for binary semantic segmentation.

Symmetric 4-level encoder/decoder with double 3x3 convolutions, max pooling,
skip concatenation and bilinear upsampling. Uses BatchNorm and ReLU.
The original U-Net paper used valid convolutions and a 64-channel stem;
this practical same-padding variant defaults to base_channels=32 for 2 GB GPUs.
Pass base_channels=64 for the wider 64/128/256/512/1024-channel model.

Inputs:  float tensor [B, 3, H, W].
Outputs: unnormalized class logits [B, 2, H, W].
No pretrained weights, SDF supervision, diffusion or special boundary modules.
"""
import torch
from torch import nn
import torch.nn.functional as F


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, base_channels=32):
        super().__init__()
        if in_channels <= 0 or num_classes <= 0 or base_channels <= 0:
            raise ValueError("Channels and class count must be positive")
        c = [base_channels * (2 ** i) for i in range(5)]
        self.enc1 = DoubleConv(in_channels, c[0])
        self.enc2 = DoubleConv(c[0], c[1])
        self.enc3 = DoubleConv(c[1], c[2])
        self.enc4 = DoubleConv(c[2], c[3])
        self.center = DoubleConv(c[3], c[4])
        self.pool = nn.MaxPool2d(2)
        self.dec4 = DoubleConv(c[4] + c[3], c[3])
        self.dec3 = DoubleConv(c[3] + c[2], c[2])
        self.dec2 = DoubleConv(c[2] + c[1], c[1])
        self.dec1 = DoubleConv(c[1] + c[0], c[0])
        self.head = nn.Conv2d(c[0], num_classes, kernel_size=1)

    @staticmethod
    def up_and_concat(decoder, skip):
        decoder = F.interpolate(
            decoder, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )
        return torch.cat([skip, decoder], dim=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        center = self.center(self.pool(e4))
        d4 = self.dec4(self.up_and_concat(center, e4))
        d3 = self.dec3(self.up_and_concat(d4, e3))
        d2 = self.dec2(self.up_and_concat(d3, e2))
        d1 = self.dec1(self.up_and_concat(d2, e1))
        return self.head(d1)
