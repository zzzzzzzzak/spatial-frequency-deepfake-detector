"""Frequency-Domain Spectral Forensics and Radial Power Law Analysis.

Calculates deviations from the natural 1/f^alpha optical power spectrum
decay curve to detect generative synthesis, autoencoder smoothing,
and upsampling checkerboard artifacts.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def fit_natural_power_law(
    radii: np.ndarray,
    power: np.ndarray,
    fit_range: Tuple[float, float] = (0.05, 0.45),
) -> Tuple[float, float, np.ndarray]:
    """Fit natural power law P(r) = A * r^(-alpha) in log space: ln P = -alpha * ln r + ln A.
    
    Natural photographic images closely adhere to alpha approx 1.8 - 2.2 across
    low-to-mid spatial frequencies.
    
    Args:
        radii: 1D array of normalized radial distances in (0, 1].
        power: 1D array of empirical azimuthal radial power values.
        fit_range: Normalized frequency range (r_min, r_max) to fit the natural slope.
        
    Returns:
        alpha: Fitted power law exponent (decay rate).
        intercept: Logarithmic intercept ln A.
        fitted_power: Theoretical power curve extrapolated across all radii.
    """
    eps = 1e-8
    mask = (radii >= fit_range[0]) & (radii <= fit_range[1]) & (power > eps)
    if mask.sum() < 3:
        # Fallback if too few points
        alpha = 2.0
        intercept = np.log(np.maximum(power[0], eps))
        fitted = np.exp(intercept) * np.power(radii + eps, -alpha)
        return alpha, intercept, fitted

    log_r = np.log(radii[mask] + eps)
    log_p = np.log(power[mask] + eps)

    # Linear regression in log space: log_p = -alpha * log_r + intercept
    coeffs = np.polyfit(log_r, log_p, deg=1)
    alpha = -coeffs[0]
    intercept = coeffs[1]

    fitted_power = np.exp(intercept) * np.power(radii + eps, -alpha)
    return float(alpha), float(intercept), fitted_power


def calculate_spectral_divergence(
    empirical_power: np.ndarray,
    theoretical_power: np.ndarray,
    high_freq_start: float = 0.50,
) -> float:
    """Calculate the high-frequency spectral divergence metric.
    
    Measures the mean squared logarithmic residual between empirical power
    and natural theoretical power in the high-frequency regime.
    """
    eps = 1e-8
    num_bins = len(empirical_power)
    start_idx = int(num_bins * high_freq_start)
    
    emp_high = empirical_power[start_idx:]
    theo_high = theoretical_power[start_idx:]
    
    log_emp = np.log(np.maximum(emp_high, eps))
    log_theo = np.log(np.maximum(theo_high, eps))
    
    divergence = np.mean((log_emp - log_theo) ** 2)
    return float(divergence)


def detect_periodic_peaks(
    profile: np.ndarray,
    window_size: int = 2,
    threshold_sigma: float = 1.5,
) -> List[int]:
    """Identify isolated frequency spikes (e.g., transposed conv checkerboard artifacts)."""
    n = len(profile)
    peaks = []
    
    for i in range(1, n - 1):
        if profile[i] <= profile[i - 1] or profile[i] <= profile[i + 1]:
            continue
        left = profile[max(0, i - window_size):i]
        right = profile[i + 1:min(n, i + window_size + 1)]
        local_window = np.concatenate([left, right])
        mean = np.mean(local_window)
        std = np.std(local_window) + 1e-8
        
        z_score = (profile[i] - mean) / std
        diff = profile[i] - mean
        if (z_score >= threshold_sigma or diff > 0.15) and diff > 0.05:
            peaks.append(int(i))
            
    return peaks


def analyze_spectral_profile(radial_profile: np.ndarray) -> Dict[str, Any]:
    """Perform full forensic spectral analysis on an azimuthal radial profile.
    
    Args:
        radial_profile: 1D array of shape (num_bins,) normalized in [0, 1].
        
    Returns:
        Dictionary with fitted alpha, divergence metric, theoretical curve,
        and detected periodic peaks.
    """
    num_bins = len(radial_profile)
    radii = (np.arange(num_bins) + 1.0) / float(num_bins)
    
    alpha, intercept, theoretical = fit_natural_power_law(radii, radial_profile)
    divergence = calculate_spectral_divergence(radial_profile, theoretical)
    peaks = detect_periodic_peaks(radial_profile)
    
    # Is spectral anomaly present? (Abnormal alpha, high divergence, or periodic spikes)
    is_anomalous = (alpha < 1.2 or alpha > 2.8) or (divergence > 0.40) or (len(peaks) >= 2)

    return {
        "radii": radii.tolist(),
        "empirical_power": radial_profile.tolist(),
        "theoretical_power": theoretical.tolist(),
        "fitted_alpha": alpha,
        "intercept": intercept,
        "spectral_divergence": divergence,
        "periodic_peaks": peaks,
        "is_spectral_anomalous": is_anomalous,
    }
