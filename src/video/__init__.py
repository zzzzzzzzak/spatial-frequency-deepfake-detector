"""Video processing and temporal aggregation for deepfake detection."""

from src.video.frame_extractor import VideoFrameExtractor, VideoMetadata
from src.video.temporal_pooling import (
    aggregate_temporal_scores,
    compute_ema_scores,
    compute_attention_pooling,
    detect_temporal_flicker,
)

__all__ = [
    "VideoFrameExtractor",
    "VideoMetadata",
    "aggregate_temporal_scores",
    "compute_ema_scores",
    "compute_attention_pooling",
    "detect_temporal_flicker",
]
