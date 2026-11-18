"""Unit tests for Grad-CAM visual attribution and spectral forensics."""

import numpy as np
import pytest
import torch
from src.config import DetectorConfig
from src.models.dual_stream_detector import DualStreamDetector
from src.explainability.gradcam import GradCAM, generate_cam_overlay
from src.explainability.spectral_forensics import (
    fit_natural_power_law,
    calculate_spectral_divergence,
    detect_periodic_peaks,
    analyze_spectral_profile,
)
from src.data.synthetic_generator import _create_natural_texture


def test_gradcam_generation():
    cfg = DetectorConfig()
    model = DualStreamDetector(cfg)
    cam = GradCAM(model, model.get_gradcam_target_layer())
    
    x = torch.rand(1, 3, 128, 128, requires_grad=True)
    heatmap = cam.generate(x, target_class=1)
    
    assert heatmap.shape == (128, 128)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0
    assert not torch.isnan(heatmap).any()
    cam.remove_hooks()


def test_cam_overlay_blend():
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    heatmap = np.ones((128, 128), dtype=np.float32) * 0.5
    overlay = generate_cam_overlay(img, heatmap, alpha=0.5)
    assert overlay.shape == (128, 128, 3)
    assert overlay.dtype == np.uint8


def test_fit_natural_power_law():
    # Generate 1/f^2 procedural texture
    tex = _create_natural_texture(size=128, alpha=2.0)
    fft = np.fft.fftshift(np.fft.fft2(tex))
    mag = np.abs(fft)
    
    # 1D radial slice
    radii = np.linspace(0.01, 1.0, 64)
    # Synthetic power following 1/r^2
    power = 1.0 / (radii**2)
    
    alpha, intercept, theoretical = fit_natural_power_law(radii, power)
    assert 1.7 <= alpha <= 2.3  # Close to theoretical 2.0


def test_spectral_divergence_and_peaks():
    emp = np.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.02])
    theo = emp.copy()
    # Identical curves have zero divergence
    div_zero = calculate_spectral_divergence(emp, theo)
    assert div_zero == pytest.approx(0.0, abs=1e-5)
    
    # Add peak to empirical
    emp_peak = emp.copy()
    emp_peak[5] = 0.9  # Sharp frequency spike
    peaks = detect_periodic_peaks(emp_peak, threshold_sigma=1.5)
    assert 5 in peaks


def test_analyze_spectral_profile_structure():
    profile = np.linspace(1.0, 0.01, 64)
    meta = analyze_spectral_profile(profile)
    assert "fitted_alpha" in meta
    assert "spectral_divergence" in meta
    assert "periodic_peaks" in meta
    assert "is_spectral_anomalous" in meta
