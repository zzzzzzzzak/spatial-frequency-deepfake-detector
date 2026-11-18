"""Unit tests for temporal aggregation, flicker detection, and video extraction."""

import os
import tempfile
import pytest
from src.video.temporal_pooling import (
    compute_ema_scores,
    compute_attention_pooling,
    compute_top_k_pooling,
    detect_temporal_flicker,
    aggregate_temporal_scores,
)
from src.video.frame_extractor import (
    VideoFrameExtractor,
    create_synthetic_test_video,
)


def test_compute_ema_scores():
    scores = [0.2, 0.4, 0.8, 0.9]
    ema = compute_ema_scores(scores, decay=0.5)
    assert len(ema) == 4
    assert ema[0] == 0.2
    assert ema[1] == 0.5 * 0.2 + 0.5 * 0.4


def test_compute_attention_pooling():
    scores = [0.1, 0.1, 0.9, 0.1]
    attn = compute_attention_pooling(scores, temperature=0.2)
    # The peak 0.9 should strongly dominate the aggregate
    assert attn > 0.60


def test_compute_top_k_pooling():
    scores = [0.1, 0.2, 0.3, 0.8, 0.9]
    top_k = compute_top_k_pooling(scores, top_ratio=0.40)  # Top 2 scores: 0.8, 0.9
    assert top_k == pytest.approx(0.85, abs=1e-3)


def test_detect_temporal_flicker():
    stable = [0.5, 0.51, 0.49, 0.50]
    flicker_stable = detect_temporal_flicker(stable)
    assert flicker_stable["mean_flicker"] < 0.05

    jumpy = [0.1, 0.9, 0.1, 0.9]
    flicker_jumpy = detect_temporal_flicker(jumpy)
    assert flicker_jumpy["mean_flicker"] > 0.50


def test_aggregate_temporal_scores():
    scores = [0.2, 0.3, 0.7, 0.8]
    res = aggregate_temporal_scores(scores, method="ema")
    assert "video_score" in res
    assert "mean_score" in res
    assert "max_score" in res
    assert res["max_score"] == 0.8


def test_video_extractor_on_synthetic_video():
    with tempfile.NamedTemporaryFile(suffix=".avi", delete=False) as tmp:
        vid_path = tmp.name

    try:
        create_synthetic_test_video(vid_path, num_frames=10, width=64, height=64)
        extractor = VideoFrameExtractor(frame_stride=2, max_frames=10)
        frames, timestamps = extractor.extract_frames(vid_path)
        assert len(frames) == 5  # Stride of 2 on 10 frames
        assert frames[0].shape == (64, 64, 3)
        assert len(timestamps) == 5
    finally:
        if os.path.exists(vid_path):
            os.remove(vid_path)
