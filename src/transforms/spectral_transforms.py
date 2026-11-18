"""Differentiable 2D frequency-domain transforms using PyTorch.

Implements 2D Fast Fourier Transform (FFT), centered log-magnitude power spectra,
azimuthal radial integration, and Butterworth high-pass filtering for forensic
detection of generative and upsampling artifacts.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn.functional as F


def rgb_to_luminance(x: torch.Tensor) -> torch.Tensor:
    """Convert RGB tensor (B, 3, H, W) to grayscale luminance (B, 1, H, W).
    
    Uses standard ITU-R BT.601 coefficients: Y = 0.299*R + 0.587*G + 0.114*B.
    """
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.shape[1] == 1:
        return x
    weights = torch.tensor([0.299, 0.587, 0.114], dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    return (x * weights).sum(dim=1, keepdim=True)


def compute_2d_fft(
    x: torch.Tensor,
    shift: bool = True,
    log_scale: bool = True,
    norm: str = "ortho",
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute 2D FFT and power spectrum of an image tensor.
    
    Args:
        x: Input image tensor of shape (B, C, H, W) or (B, 1, H, W).
        shift: Whether to apply fftshift (shift zero frequency to center).
        log_scale: Whether to apply log(1 + magnitude) scaling.
        norm: FFT normalization ('ortho' or 'backward').
        eps: Numerical epsilon to avoid log(0).
        
    Returns:
        fft_complex: Complex 2D FFT tensor of shape (B, 1, H, W).
        spectrum: Magnitude or log-magnitude spectrum of shape (B, 1, H, W).
    """
    gray = rgb_to_luminance(x)
    fft_c = torch.fft.fft2(gray, norm=norm)
    if shift:
        fft_c = torch.fft.fftshift(fft_c, dim=(-2, -1))

    mag = torch.abs(fft_c)
    if log_scale:
        spectrum = torch.log1p(mag + eps)
    else:
        spectrum = mag

    # Min-max normalize per sample to [0, 1] for numerical stability
    b, c, h, w = spectrum.shape
    flat = spectrum.view(b, -1)
    min_val = flat.min(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
    max_val = flat.max(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
    spectrum_norm = (spectrum - min_val) / (max_val - min_val + eps)

    return fft_c, spectrum_norm


def _get_radial_bins(
    h: int, w: int, num_bins: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute flattened radial bin indices and bin count masks."""
    cy = h / 2.0
    cx = w / 2.0
    y = torch.arange(h, device=device, dtype=torch.float32) - cy
    x = torch.arange(w, device=device, dtype=torch.float32) - cx
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    
    radius = torch.sqrt(grid_y**2 + grid_x**2)
    max_radius = min(cy, cx)
    
    # Normalized radial distance in [0, 1]
    norm_radius = radius / (max_radius + 1e-8)
    bin_idx = torch.clamp((norm_radius * num_bins).long(), 0, num_bins - 1)
    bin_idx_flat = bin_idx.view(-1)
    
    # Calculate counts per bin for fast division
    counts = torch.bincount(bin_idx_flat, minlength=num_bins).float()
    counts = torch.clamp(counts, min=1.0)
    return bin_idx_flat, counts


def compute_azimuthal_average(
    spectrum: torch.Tensor,
    num_bins: int = 64,
) -> torch.Tensor:
    """Compute the 1D Azimuthal radial average of a 2D centered spectrum.
    
    Integrates concentric annular rings around the center DC frequency
    to extract the radial power spectrum profile P(r).
    
    Args:
        spectrum: Centered 2D power spectrum tensor of shape (B, 1, H, W).
        num_bins: Number of discrete radial frequency bins.
        
    Returns:
        1D Azimuthal profile of shape (B, num_bins), normalized per sample.
    """
    b, c, h, w = spectrum.shape
    device = spectrum.device
    bin_idx_flat, counts = _get_radial_bins(h, w, num_bins, device)
    
    flat_spec = spectrum.view(b, h * w)
    
    # Vectorized accumulation across batches
    # Using scatter_add along dim 1
    profiles = torch.zeros((b, num_bins), device=device, dtype=spectrum.dtype)
    expanded_bins = bin_idx_flat.unsqueeze(0).expand(b, -1)
    profiles.scatter_add_(1, expanded_bins, flat_spec)
    
    # Divide by bin counts
    profiles = profiles / counts.unsqueeze(0)
    
    # Normalize profile to [0, 1] per sample
    p_min = profiles.min(dim=1, keepdim=True)[0]
    p_max = profiles.max(dim=1, keepdim=True)[0]
    profiles_norm = (profiles - p_min) / (p_max - p_min + 1e-8)
    
    return profiles_norm


def butterworth_highpass_filter(
    x: torch.Tensor,
    cutoff: float = 0.15,
    order: int = 2,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Apply a 2D Butterworth high-pass frequency filter.
    
    Attenuates low-frequency global structures (face shape, lighting) while
    preserving high-frequency residuals (synthesis grid noise, blending seams).
    
    Transfer function: H(u, v) = 1 / (1 + (D0 / D(u, v))^(2 * order))
    
    Args:
        x: Input image tensor (B, C, H, W) or (B, 1, H, W).
        cutoff: Normalized cutoff frequency D0 in (0, 0.5].
        order: Filter order controlling transition steepness.
        eps: Numerical epsilon.
        
    Returns:
        Filtered spatial residual tensor of shape (B, 1, H, W) normalized to [0, 1].
    """
    gray = rgb_to_luminance(x)
    b, c, h, w = gray.shape
    device = gray.device
    
    cy = h / 2.0
    cx = w / 2.0
    y = torch.arange(h, device=device, dtype=torch.float32) - cy
    x_coord = torch.arange(w, device=device, dtype=torch.float32) - cx
    grid_y, grid_x = torch.meshgrid(y, x_coord, indexing="ij")
    
    max_dim = min(cy, cx)
    d = torch.sqrt(grid_y**2 + grid_x**2) / (max_dim + eps)
    
    # Filter transfer function H(u, v)
    # H = 1 / (1 + (cutoff / (d + eps))^(2*order))
    h_filter = 1.0 / (1.0 + torch.pow((cutoff / (d + eps)), 2 * order))
    h_filter = h_filter.view(1, 1, h, w)
    
    # Compute centered FFT
    fft_c = torch.fft.fftshift(torch.fft.fft2(gray, norm="ortho"), dim=(-2, -1))
    
    # Filter in frequency domain
    fft_filtered = fft_c * h_filter
    
    # Inverse shift and IFFT
    fft_ishift = torch.fft.ifftshift(fft_filtered, dim=(-2, -1))
    residual = torch.real(torch.fft.ifft2(fft_ishift, norm="ortho"))
    
    # Normalize residual to [0, 1]
    res_flat = residual.view(b, -1)
    r_min = res_flat.min(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
    r_max = res_flat.max(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
    residual_norm = (residual - r_min) / (r_max - r_min + eps)
    
    return residual_norm


def extract_spectral_features(
    x: torch.Tensor,
    num_bins: int = 64,
    high_pass_cutoff: float = 0.15,
) -> Dict[str, torch.Tensor]:
    """Extract complete suite of spectral forensic representations.
    
    Returns:
        dict containing:
            'log_spectrum': 2D normalized log-power spectrum (B, 1, H, W)
            'azimuthal_profile': 1D radial power profile (B, num_bins)
            'high_pass_residual': Spatial high-pass residual (B, 1, H, W)
    """
    _, log_spectrum = compute_2d_fft(x, shift=True, log_scale=True)
    azimuthal_profile = compute_azimuthal_average(log_spectrum, num_bins=num_bins)
    residual = butterworth_highpass_filter(x, cutoff=high_pass_cutoff)
    
    return {
        "log_spectrum": log_spectrum,
        "azimuthal_profile": azimuthal_profile,
        "high_pass_residual": residual,
    }
