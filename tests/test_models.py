"""Unit tests for spatial, frequency, fusion, and dual-stream architectures."""

import pytest
import torch
from src.config import DetectorConfig
from src.models.spatial_stream import SpatialStream
from src.models.frequency_stream import FrequencyStream
from src.models.fusion import CrossAttentionGatedFusion
from src.models.dual_stream_detector import DualStreamDetector


def test_spatial_stream_forward():
    cfg = DetectorConfig()
    model = SpatialStream(cfg.spatial)
    x = torch.rand(2, 3, 128, 128)
    feat, logits, spatial_map = model(x, return_spatial_map=True)
    assert feat.shape == (2, cfg.spatial.feature_dim)
    assert logits.shape == (2, 2)
    assert spatial_map.ndim == 4


def test_frequency_stream_forward():
    cfg = DetectorConfig()
    model = FrequencyStream(cfg.frequency)
    spec = torch.rand(2, 1, 128, 128)
    rad = torch.rand(2, cfg.frequency.radial_bins)
    fused_feat, logits = model(spec, rad)
    assert fused_feat.shape == (2, cfg.frequency.fused_dim)
    assert logits.shape == (2, 2)


def test_fusion_forward():
    cfg = DetectorConfig()
    fusion = CrossAttentionGatedFusion(cfg.fusion)
    sp_feat = torch.rand(2, cfg.fusion.spatial_dim)
    fr_feat = torch.rand(2, cfg.fusion.freq_dim)
    logits, gate, rep = fusion(sp_feat, fr_feat)
    assert logits.shape == (2, 2)
    assert gate.shape == (2, 1)
    assert (gate >= 0.0).all() and (gate <= 1.0).all()
    assert rep.shape == (2, cfg.fusion.hidden_dim)


def test_dual_stream_detector_end_to_end():
    cfg = DetectorConfig()
    model = DualStreamDetector(cfg)
    x = torch.rand(2, 3, 128, 128)
    out = model(x, return_intermediates=True)

    assert "logits" in out
    assert "probabilities" in out
    assert "deepfake_score" in out
    assert "spatial_logits" in out
    assert "freq_logits" in out
    assert "gate_weight" in out

    probs = out["probabilities"]
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5)

    # Test backward pass
    loss = out["logits"].sum()
    loss.backward()
    
    # Check gradients in spatial and frequency backbones
    assert model.spatial_stream.stem[0].weight.grad is not None
    assert model.frequency_stream.conv_branch[0].conv.weight.grad is not None
