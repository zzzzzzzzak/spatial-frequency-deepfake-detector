"""Frequency Forensics Stream.

Extracts spectral representations from 2D FFT log-power spectra, high-pass
residuals, and 1D Azimuthal radial profiles to detect generative synthesis
artifacts and spectral power decay anomalies.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import FrequencyModelConfig


class SpectralConvBlock(nn.Module):
    """Convolutional block designed for spectral magnitude pattern detection."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class FrequencyStream(nn.Module):
    """Dual-branch frequency forensic encoder."""
    def __init__(self, config: FrequencyModelConfig):
        super().__init__()
        self.config = config

        # Branch 1: 2D Spectral Pattern ConvNet
        layers = []
        curr_channels = config.in_channels
        for mult in config.channel_multipliers:
            out_channels = config.base_channels * mult
            layers.append(SpectralConvBlock(curr_channels, out_channels, stride=2))
            curr_channels = out_channels

        self.conv_branch = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_proj = nn.Linear(curr_channels, config.spectral_feature_dim)

        # Branch 2: 1D Azimuthal Radial Profile MLP
        self.radial_mlp = nn.Sequential(
            nn.Linear(config.radial_bins, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(p=config.dropout),
            nn.Linear(128, config.radial_feature_dim),
            nn.LayerNorm(config.radial_feature_dim),
        )

        # Fusion of 2D spectral + 1D radial representations
        combined_dim = config.spectral_feature_dim + config.radial_feature_dim
        self.fusion_fc = nn.Sequential(
            nn.Linear(combined_dim, config.fused_dim),
            nn.LayerNorm(config.fused_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(p=config.dropout),
        )

        # Auxiliary standalone frequency classifier
        self.classifier = nn.Linear(config.fused_dim, 2)

    def forward(
        self,
        spectral_map: torch.Tensor,
        radial_profile: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            spectral_map: 2D power spectrum or residual (B, 1, H, W).
            radial_profile: 1D Azimuthal profile (B, radial_bins).
            
        Returns:
            fused_features: Frequency feature vector (B, fused_dim).
            logits: Standalone frequency classification logits (B, 2).
        """
        # 2D spectral branch
        conv_out = self.conv_branch(spectral_map)
        pooled = self.global_pool(conv_out).flatten(1)
        spec_feat = self.conv_proj(pooled)

        # 1D radial profile branch
        radial_feat = self.radial_mlp(radial_profile)

        # Joint concatenation & projection
        combined = torch.cat([spec_feat, radial_feat], dim=-1)
        fused_features = self.fusion_fc(combined)
        logits = self.classifier(fused_features)

        return fused_features, logits
