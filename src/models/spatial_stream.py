"""Spatial Forensics Stream with Squeeze-and-Excitation Residual Backbone.

Extracts multi-scale spatial textures, boundary discrepancies, and facial
inconsistencies. Retains feature activations for Grad-CAM explainability.
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import SpatialModelConfig


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation channel attention block."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        scale = self.fc(x).view(b, c, 1, 1)
        return x * scale


class ResidualBlock(nn.Module):
    """Residual convolutional block with SE attention and downsampling support."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, use_se: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SqueezeExcitation(out_channels) if use_se else nn.Identity()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += residual
        return self.relu(out)


class SpatialStream(nn.Module):
    """Spatial forensics backbone extracting multi-scale boundary and blending representations."""
    def __init__(self, config: SpatialModelConfig):
        super().__init__()
        self.config = config
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(config.in_channels, config.base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(config.base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        # Stages
        curr_channels = config.base_channels
        self.stages = nn.ModuleList()
        for i, mult in enumerate(config.channel_multipliers):
            out_channels = config.base_channels * mult
            stride = 1 if i == 0 else 2
            stage = nn.Sequential(
                ResidualBlock(curr_channels, out_channels, stride=stride, use_se=config.use_se_attention),
                ResidualBlock(out_channels, out_channels, stride=1, use_se=config.use_se_attention),
            )
            self.stages.append(stage)
            curr_channels = out_channels

        self.last_conv_channels = curr_channels
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=config.dropout)
        self.proj = nn.Linear(self.last_conv_channels, config.feature_dim)
        
        # Auxiliary classification head for spatial supervision
        self.classifier = nn.Linear(config.feature_dim, 2)

    def get_target_layer_for_cam(self) -> nn.Module:
        """Return the final convolutional layer for Grad-CAM gradient tracking."""
        return self.stages[-1][-1].conv2

    def forward(
        self, x: torch.Tensor, return_spatial_map: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input RGB tensor (B, 3, H, W).
            return_spatial_map: Whether to return the unpooled 2D feature map.
            
        Returns:
            features: Projected 1D spatial feature embedding (B, feature_dim).
            logits: Standalone spatial classification logits (B, 2).
            spatial_map: 2D feature map from the final stage (B, C, H', W').
        """
        out = self.stem(x)
        for stage in self.stages:
            out = stage(out)

        spatial_map = out  # Saved for Grad-CAM hooks
        pooled = self.pool(spatial_map).flatten(1)
        features = self.proj(self.dropout(pooled))
        features = F.layer_norm(features, (features.shape[-1],))
        logits = self.classifier(features)

        return features, logits, spatial_map
