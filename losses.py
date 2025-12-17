from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _stft_mag(
    x: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    center: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute magnitude STFT for (B, 1, T) or (B, T)."""
    if x.dim() == 3:
        x = x.squeeze(1)
    window = torch.hann_window(win_length, device=x.device, dtype=x.dtype)
    spec = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        return_complex=True,
    )
    mag = spec.abs().clamp_min(eps)
    return mag


@dataclass(frozen=True)
class MRSTFTConfig:
    fft_sizes: Sequence[int] = (512, 1024, 2048)
    hop_ratio: int = 4  # hop = n_fft // hop_ratio
    win_ratio: int = 1  # win = n_fft // win_ratio
    sc_weight: float = 1.0
    logmag_weight: float = 1.0


class MultiResolutionSTFTLoss(nn.Module):
    """Multi-resolution STFT loss: spectral convergence + log-magnitude L1."""

    def __init__(self, cfg: MRSTFTConfig = MRSTFTConfig()):
        super().__init__()
        self.cfg = cfg

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        total = 0.0
        for n_fft in self.cfg.fft_sizes:
            hop = max(1, n_fft // self.cfg.hop_ratio)
            win = max(1, n_fft // self.cfg.win_ratio)

            x_mag = _stft_mag(y_hat, n_fft=n_fft, hop_length=hop, win_length=win)
            y_mag = _stft_mag(y, n_fft=n_fft, hop_length=hop, win_length=win)

            # Spectral Convergence: ||Y - X||_F / ||Y||_F
            diff = y_mag - x_mag
            diff_norm = torch.sqrt(torch.sum(diff * diff, dim=(1, 2)) + 1e-8)
            y_norm = torch.sqrt(torch.sum(y_mag * y_mag, dim=(1, 2)) + 1e-8)
            sc = torch.mean(diff_norm / y_norm)
            # Log magnitude L1
            logmag = F.l1_loss(torch.log(y_mag), torch.log(x_mag))

            total = total + self.cfg.sc_weight * sc + self.cfg.logmag_weight * logmag

        return total / float(len(self.cfg.fft_sizes))


@dataclass(frozen=True)
class HybridLossConfig:
    l1_weight: float = 1.0
    stft_weight: float = 1.0


class HybridTimeFreqLoss(nn.Module):
    """Hybrid L1 (time) + Multi-Res STFT (freq) loss."""

    def __init__(
        self,
        stft_cfg: MRSTFTConfig = MRSTFTConfig(),
        cfg: HybridLossConfig = HybridLossConfig(),
    ):
        super().__init__()
        self.cfg = cfg
        self.stft = MultiResolutionSTFTLoss(stft_cfg)

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        l1 = F.l1_loss(y_hat, y)
        stft = self.stft(y_hat, y)
        return self.cfg.l1_weight * l1 + self.cfg.stft_weight * stft
