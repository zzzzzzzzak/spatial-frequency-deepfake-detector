"""Configuration dataclasses for the Spatial-Frequency Deepfake Detector."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class SpectralConfig:
    """Configuration for frequency-domain transforms."""
    image_size: int = 256
    num_radial_bins: int = 64
    high_pass_cutoff: float = 0.15
    butterworth_order: int = 2
    log_scale: bool = True
    eps: float = 1e-8


@dataclass
class SpatialModelConfig:
    """Configuration for spatial feature extraction backbone."""
    in_channels: int = 3
    base_channels: int = 32
    channel_multipliers: Tuple[int, ...] = (1, 2, 4, 8)
    feature_dim: int = 256
    dropout: float = 0.3
    use_se_attention: bool = True


@dataclass
class FrequencyModelConfig:
    """Configuration for frequency spectral encoder."""
    in_channels: int = 1
    base_channels: int = 16
    channel_multipliers: Tuple[int, ...] = (1, 2, 4)
    spectral_feature_dim: int = 128
    radial_bins: int = 64
    radial_feature_dim: int = 64
    fused_dim: int = 128
    dropout: float = 0.25


@dataclass
class FusionConfig:
    """Configuration for cross-attention gated multimodal fusion."""
    spatial_dim: int = 256
    freq_dim: int = 128
    hidden_dim: int = 256
    num_heads: int = 4
    num_classes: int = 2
    dropout: float = 0.3
    auxiliary_heads: bool = True


@dataclass
class TemporalConfig:
    """Configuration for video frame-level temporal aggregation."""
    pooling_method: str = "ema"  # 'ema', 'attention', 'top_k', 'mean'
    ema_decay: float = 0.85
    top_k_ratio: float = 0.20
    frame_stride: int = 2
    max_frames: int = 120


@dataclass
class DetectorConfig:
    """Master configuration for the dual-stream deepfake detector."""
    spectral: SpectralConfig = field(default_factory=SpectralConfig)
    spatial: SpatialModelConfig = field(default_factory=SpatialModelConfig)
    frequency: FrequencyModelConfig = field(default_factory=FrequencyModelConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    threshold: float = 0.50
    device: str = "auto"
