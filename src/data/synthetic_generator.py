"""Procedural forensic artifact generator for authentic vs synthetic benchmark datasets.

Generates mathematically controlled images for evaluating spatial boundary
artifacts and frequency spectral decay anomalies without requiring hundreds
of gigabytes of external video datasets.
"""

from typing import List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset


def _create_natural_texture(size: int = 256, alpha: float = 2.0, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """Generate 2D natural fractal noise following 1/f^alpha power spectrum."""
    if rng is None:
        rng = np.random.RandomState()
    
    # Generate white noise in frequency domain
    white_noise = rng.randn(size, size)
    fft = np.fft.fft2(white_noise)
    fft_shifted = np.fft.fftshift(fft)
    
    # 2D frequency grid
    cy, cx = size / 2.0, size / 2.0
    y, x = np.ogrid[:size, :size]
    radius = np.sqrt((y - cy)**2 + (x - cx)**2)
    radius[int(cy), int(cx)] = 1.0  # Avoid division by zero at DC
    
    # Apply 1/f^(alpha/2) filter to amplitude
    filter_amp = 1.0 / np.power(radius, alpha / 2.0)
    filtered_fft = fft_shifted * filter_amp
    
    # Invert back to spatial domain
    ifft = np.fft.ifft2(np.fft.ifftshift(filtered_fft))
    spatial = np.real(ifft)
    
    # Min-max normalize to [0, 1]
    s_min, s_max = spatial.min(), spatial.max()
    normalized = (spatial - s_min) / (s_max - s_min + 1e-8)
    return normalized


def generate_authentic_face(size: int = 256, seed: Optional[int] = None) -> np.ndarray:
    """Procedurally synthesize an authentic photographic face surrogate.
    
    Constructed with organic multi-scale textures that rigorously adhere
    to natural 1/f^2 spectral power distributions without artificial seams.
    """
    rng = np.random.RandomState(seed)
    # Generate 3 correlated RGB natural texture layers
    base_tex = _create_natural_texture(size, alpha=2.0, rng=rng)
    r = np.clip(base_tex * 0.85 + 0.15 * rng.rand(), 0.0, 1.0)
    g = np.clip(base_tex * 0.70 + 0.20 * rng.rand(), 0.0, 1.0)
    b = np.clip(base_tex * 0.60 + 0.15 * rng.rand(), 0.0, 1.0)
    
    img = np.stack([r, g, b], axis=-1)
    
    # Add gentle organic low-frequency illumination vignette
    y, x = np.ogrid[:size, :size]
    cy, cx = size / 2.0, size / 2.0
    vignette = 1.0 - 0.25 * ((y - cy)**2 + (x - cx)**2) / ((size / 2.0)**2)
    img = np.clip(img * vignette[..., np.newaxis], 0.0, 1.0)
    
    return (img * 255.0).astype(np.uint8)


def generate_synthetic_face(
    size: int = 256,
    artifact_type: str = "checkerboard",  # 'checkerboard', 'blending_seam', 'frequency_rolloff'
    seed: Optional[int] = None,
) -> np.ndarray:
    """Procedurally synthesize a manipulated / deepfake face surrogate.
    
    Injects specific mathematical anomalies characteristic of generative models:
      - 'checkerboard': High-frequency periodic 2D sinusoids from transposed convolutions.
      - 'blending_seam': Discontinuous boundary perimeter from face swapping & splicing.
      - 'frequency_rolloff': Unnatural high-frequency smoothing from autoencoders.
    """
    rng = np.random.RandomState(seed)
    authentic_base = generate_authentic_face(size, seed=seed).astype(np.float32) / 255.0
    
    if artifact_type == "checkerboard":
        # Periodic grid injection (Nyquist frequency peaks)
        freq_u = rng.choice([16, 32, 48])
        freq_v = rng.choice([16, 32, 48])
        y, x = np.ogrid[:size, :size]
        grid = np.sin(2 * np.pi * freq_u * x / size) * np.sin(2 * np.pi * freq_v * y / size)
        
        # Inject periodic artifact
        amplitude = rng.uniform(0.08, 0.16)
        synthetic = np.clip(authentic_base + amplitude * grid[..., np.newaxis], 0.0, 1.0)
        
    elif artifact_type == "blending_seam":
        # Simulate face swap blending boundary
        cy, cx = size / 2.0, size / 2.0
        radius = size * rng.uniform(0.25, 0.38)
        y, x = np.ogrid[:size, :size]
        dist = np.sqrt((y - cy)**2 + (x - cx)**2)
        
        # Inner altered region
        inner_donor = _create_natural_texture(size, alpha=1.2, rng=rng)[..., np.newaxis]
        inner_donor = np.repeat(inner_donor, 3, axis=-1)
        
        # Boundary mask with sharp transition seam
        mask = (dist < radius).astype(np.float32)
        # Add high-frequency edge seam along boundary
        edge_seam = np.exp(-((dist - radius)**2) / 8.0) * rng.uniform(0.2, 0.4)
        
        synthetic = authentic_base * (1.0 - mask[..., np.newaxis]) + inner_donor * mask[..., np.newaxis]
        synthetic = np.clip(synthetic + edge_seam[..., np.newaxis], 0.0, 1.0)
        
    elif artifact_type == "frequency_rolloff":
        # Over-smoothed autoencoder reconstruction (alpha ~ 3.5, excessive high-freq decay)
        smoothed = _create_natural_texture(size, alpha=3.8, rng=rng)[..., np.newaxis]
        smoothed = np.repeat(smoothed, 3, axis=-1)
        synthetic = np.clip(authentic_base * 0.3 + smoothed * 0.7, 0.0, 1.0)
        
    else:
        synthetic = authentic_base

    return (synthetic * 255.0).astype(np.uint8)


class SyntheticForensicsDataset(Dataset):
    """PyTorch Dataset yielding authentic and deepfake forensic pairs."""
    def __init__(self, num_samples: int = 100, size: int = 256, seed: int = 42):
        self.num_samples = num_samples
        self.size = size
        self.seed = seed
        self.samples = []
        
        artifacts = ["checkerboard", "blending_seam", "frequency_rolloff"]
        for i in range(num_samples):
            label = i % 2  # Alternate 0 (Real) and 1 (Deepfake)
            sample_seed = seed + i
            if label == 0:
                self.samples.append((0, "authentic", sample_seed))
            else:
                art = artifacts[(i // 2) % len(artifacts)]
                self.samples.append((1, art, sample_seed))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        label, art_type, s_seed = self.samples[idx]
        if label == 0:
            img = generate_authentic_face(self.size, seed=s_seed)
        else:
            img = generate_synthetic_face(self.size, artifact_type=art_type, seed=s_seed)

        # Convert to float32 tensor (3, H, W) normalized in [0, 1]
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        return tensor, label, art_type


def generate_benchmark_dataset(
    num_samples: int = 100, size: int = 256, seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """Generate pre-batched tensors for instant offline evaluation."""
    dataset = SyntheticForensicsDataset(num_samples=num_samples, size=size, seed=seed)
    images = []
    labels = []
    artifact_types = []

    for i in range(len(dataset)):
        img, lbl, art = dataset[i]
        images.append(img)
        labels.append(lbl)
        artifact_types.append(art)

    return torch.stack(images), torch.tensor(labels, dtype=torch.long), artifact_types
