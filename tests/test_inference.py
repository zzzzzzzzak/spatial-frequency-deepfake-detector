"""End-to-end unit tests for inference engine and forensic serialization."""

import os
import tempfile
import numpy as np
import pytest
from src.config import DetectorConfig
from src.inference import DeepfakeInferenceEngine, DetectionResult
from src.data.synthetic_generator import generate_authentic_face, generate_synthetic_face
from src.video.frame_extractor import create_synthetic_test_video


def test_inference_engine_predict_image():
    engine = DeepfakeInferenceEngine(device="cpu")
    img = generate_authentic_face(128, seed=42)
    result, heatmap, overlay = engine.predict_image(img, generate_explainability=True)

    assert isinstance(result, DetectionResult)
    assert result.media_type == "image"
    assert 0.0 <= result.confidence <= 1.0
    assert 0.0 <= result.spatial_score <= 1.0
    assert 0.0 <= result.frequency_score <= 1.0
    assert heatmap is not None
    assert overlay is not None
    assert overlay.shape == img.shape


def test_inference_engine_predict_video():
    engine = DeepfakeInferenceEngine(device="cpu")
    with tempfile.NamedTemporaryFile(suffix=".avi", delete=False) as tmp:
        vid_path = tmp.name

    try:
        create_synthetic_test_video(vid_path, num_frames=6, width=64, height=64)
        result, heatmap, overlay = engine.predict_video(vid_path)

        assert isinstance(result, DetectionResult)
        assert result.media_type == "video"
        assert result.frame_scores is not None
        assert len(result.frame_scores) > 0
        assert result.timeline_stats is not None
    finally:
        if os.path.exists(vid_path):
            os.remove(vid_path)


def test_detection_result_json_serialization():
    engine = DeepfakeInferenceEngine(device="cpu")
    img = generate_synthetic_face(128, artifact_type="checkerboard", seed=99)
    result, _, _ = engine.predict_image(img, generate_explainability=False)

    data = result.to_dict()
    assert isinstance(data, dict)
    assert "is_deepfake" in data
    assert "confidence" in data
    assert "spectral_divergence" in data
