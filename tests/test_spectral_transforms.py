"""Unit tests for frequency-domain transforms."""

import pytest
import torch
from src.transforms.spectral_transforms import (
    rgb_to_luminance,
    compute_2d_fft,
    compute_azimuthal_average,
    butterworth_highpass_filter,
    extract_spectral_features,
)


def test_rgb_to_luminance():
    x = torch.rand(2, 3, 64, 64)
    gray = rgb_to_luminance(x)
    assert gray.shape == (2, 1, 64, 64)
    assert gray.min() >= 0.0
    assert gray.max() <= 1.0


def test_compute_2d_fft():
    x = torch.rand(2, 3, 64, 64)
    fft_c, spec = compute_2d_fft(x, shift=True, log_scale=True)
    assert fft_c.shape == (2, 1, 64, 64)
    assert spec.shape == (2, 1, 64, 64)
    assert spec.min() >= 0.0
    assert spec.max() <= 1.0


def test_compute_azimuthal_average():
    x = torch.rand(2, 3, 64, 64)
    _, spec = compute_2d_fft(x, shift=True)
    rad = compute_azimuthal_average(spec, num_bins=32)
    assert rad.shape == (2, 32)
    assert rad.min() >= 0.0
    assert rad.max() <= 1.0


def test_butterworth_highpass_filter():
    x = torch.rand(2, 3, 64, 64)
    res = butterworth_highpass_filter(x, cutoff=0.2, order=2)
    assert res.shape == (2, 1, 64, 64)
    assert res.min() >= 0.0
    assert res.max() <= 1.0


def test_spectral_transforms_differentiability():
    x = torch.rand(1, 3, 32, 32, requires_grad=True)
    _, spec = compute_2d_fft(x)
    loss = spec.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_extract_spectral_features_keys():
    x = torch.rand(1, 3, 64, 64)
    feats = extract_spectral_features(x, num_bins=16)
    assert "log_spectrum" in feats
    assert "azimuthal_profile" in feats
    assert "high_pass_residual" in feats
    assert feats["azimuthal_profile"].shape == (1, 16)
