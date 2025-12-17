from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, padding: int | None = None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class DownBlock(nn.Module):
    """Downsample by stride-2 Conv1d (no MaxPool)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        # kernel=4,pad=1,stride=2 gives exact half-length for even T
        self.down = ConvBlock1D(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.refine = ConvBlock1D(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        x = self.refine(x)
        return x


class UpBlock(nn.Module):
    """Upsample by linear interpolation + Conv1d (no ConvTranspose1d)."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.pre = ConvBlock1D(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.post = ConvBlock1D(out_ch + skip_ch, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="linear", align_corners=False)
        x = self.pre(x)

        if x.shape[-1] != skip.shape[-1]:
            # defensive: crop to the minimum length
            t = min(x.shape[-1], skip.shape[-1])
            x = x[..., :t]
            skip = skip[..., :t]

        x = torch.cat([x, skip], dim=1)
        x = self.post(x)
        return x


@dataclass
class WaveUNetConfig:
    in_channels: int = 1
    out_channels: int = 1
    base_channels: int = 32
    depth: int = 5


class WaveUNet1D(nn.Module):
    """Simplified Wave-U-Net style generator for (B, 1, 2048) -> (B, 1, 2048)."""

    def __init__(self, cfg: WaveUNetConfig = WaveUNetConfig()):
        super().__init__()
        self.cfg = cfg

        self.in_proj = ConvBlock1D(cfg.in_channels, cfg.base_channels, kernel_size=3, stride=1, padding=1)

        downs: List[nn.Module] = []
        enc_channels: List[int] = [cfg.base_channels]
        ch = cfg.base_channels
        for _ in range(cfg.depth):
            out_ch = ch * 2
            downs.append(DownBlock(ch, out_ch))
            ch = out_ch
            enc_channels.append(ch)
        self.downs = nn.ModuleList(downs)

        self.bottleneck = nn.Sequential(
            ConvBlock1D(ch, ch, kernel_size=3, stride=1, padding=1),
            ConvBlock1D(ch, ch, kernel_size=3, stride=1, padding=1),
        )

        # Skip connections are taken at resolutions BEFORE the deepest downsample.
        # Resolutions: base (2048) + each down output except the last.
        skip_channels = enc_channels[:-1]  # length == depth

        ups: List[nn.Module] = []
        for d in range(cfg.depth - 1, -1, -1):
            skip_ch = skip_channels[d]
            out_ch = skip_ch
            ups.append(UpBlock(ch, skip_ch, out_ch))
            ch = out_ch
        self.ups = nn.ModuleList(ups)

        self.out_proj = nn.Conv1d(ch, cfg.out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)

        skips: List[torch.Tensor] = [x]
        for i, down in enumerate(self.downs):
            x = down(x)
            # Do not take a skip at the deepest level (it would be too small and misaligned).
            if i < self.cfg.depth - 1:
                skips.append(x)

        x = self.bottleneck(x)

        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip)

        y = self.out_proj(x)
        return y


class PeriodicConvDiscriminator(nn.Module):
    """A simple 1D waveform discriminator block (MelGAN/HiFi-GAN-like)."""

    def __init__(self, in_channels: int = 1, channels: int = 16):
        super().__init__()
        chs = [channels, channels * 4, channels * 16, channels * 64, channels * 64]
        strides = [1, 2, 2, 2, 1]
        kernels = [15, 41, 41, 41, 5]
        groups = [1, 4, 16, 16, 1]

        layers: List[nn.Module] = []
        prev = in_channels
        for c, s, k, g in zip(chs, strides, kernels, groups):
            pad = (k - 1) // 2
            layers.append(nn.Conv1d(prev, c, kernel_size=k, stride=s, padding=pad, groups=g))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            prev = c
        layers.append(nn.Conv1d(prev, 1, kernel_size=3, stride=1, padding=1))

        self.net = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        feats: List[torch.Tensor] = []
        for layer in self.net:
            x = layer(x)
            if isinstance(layer, nn.LeakyReLU):
                feats.append(x)
        return x, feats


class MultiScale1DDiscriminator(nn.Module):
    """Multi-scale discriminator on waveform for future GAN training."""

    def __init__(self, num_scales: int = 3):
        super().__init__()
        self.num_scales = num_scales
        self.discriminators = nn.ModuleList([PeriodicConvDiscriminator() for _ in range(num_scales)])
        self.pools = nn.ModuleList([
            nn.Identity(),
            nn.AvgPool1d(kernel_size=4, stride=2, padding=1),
            nn.AvgPool1d(kernel_size=4, stride=2, padding=1),
        ])

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        logits: List[torch.Tensor] = []
        features: List[List[torch.Tensor]] = []
        cur = x
        for i in range(self.num_scales):
            if i > 0:
                cur = self.pools[i](cur)
            logit, feats = self.discriminators[i](cur)
            logits.append(logit)
            features.append(feats)
        return logits, features
