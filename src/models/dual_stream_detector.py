"""Dual-Stream Spatial-Frequency Deepfake Detector.

Unifies spatial boundary forensics and frequency-domain spectral analysis
into an end-to-end differentiable detection system.
"""

from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import DetectorConfig
from src.models.spatial_stream import SpatialStream
from src.models.frequency_stream import FrequencyStream
from src.models.fusion import CrossAttentionGatedFusion
from src.transforms.spectral_transforms import (
    compute_2d_fft,
    compute_azimuthal_average,
    butterworth_highpass_filter,
)


class DualStreamDetector(nn.Module):
    """Unified Deep Learning detector leveraging spatial textures and frequency spectra."""
    def __init__(self, config: Optional[DetectorConfig] = None):
        super().__init__()
        self.config = config or DetectorConfig()

        # Spatial Forensics Stream
        self.spatial_stream = SpatialStream(self.config.spatial)

        # Frequency Forensics Stream
        self.frequency_stream = FrequencyStream(self.config.frequency)

        # Cross-Attention Gated Multimodal Fusion
        self.fusion = CrossAttentionGatedFusion(self.config.fusion)

    def get_gradcam_target_layer(self) -> nn.Module:
        """Target convolutional layer for spatial Grad-CAM hooks."""
        return self.spatial_stream.get_target_layer_for_cam()

    def extract_spectral_tensors(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute 2D centered log-power spectrum and 1D azimuthal profile."""
        _, spectrum = compute_2d_fft(
            x,
            shift=True,
            log_scale=self.config.spectral.log_scale,
            eps=self.config.spectral.eps,
        )
        radial_profile = compute_azimuthal_average(
            spectrum, num_bins=self.config.spectral.num_radial_bins
        )
        return spectrum, radial_profile

    def forward(
        self,
        x: torch.Tensor,
        spectrum: Optional[torch.Tensor] = None,
        radial_profile: Optional[torch.Tensor] = None,
        return_intermediates: bool = False,
    ) -> Dict[str, Any]:
        """End-to-end forward pass.
        
        Args:
            x: Input RGB batch (B, 3, H, W).
            spectrum: Optional precomputed 2D power spectrum (B, 1, H, W).
            radial_profile: Optional precomputed 1D radial profile (B, radial_bins).
            return_intermediates: Whether to include spatial feature maps & embeddings.
            
        Returns:
            Dictionary containing:
                'logits': Final fused classification logits (B, 2)
                'probabilities': Softmax probabilities (B, 2), where index 1 = Deepfake
                'deepfake_score': Probability of synthetic media (B,)
                'spatial_logits': Auxiliary spatial classification logits (B, 2)
                'freq_logits': Auxiliary frequency classification logits (B, 2)
                'gate_weight': Dynamic gating ratio balancing spatial vs frequency (B, 1)
        """
        # Compute spectral features if not provided
        if spectrum is None or radial_profile is None:
            spec_calc, rad_calc = self.extract_spectral_tensors(x)
            spectrum = spectrum if spectrum is not None else spec_calc
            radial_profile = radial_profile if radial_profile is not None else rad_calc

        # Forward spatial stream
        spatial_feat, spatial_logits, spatial_map = self.spatial_stream(x, return_spatial_map=True)

        # Forward frequency stream
        freq_feat, freq_logits = self.frequency_stream(spectrum, radial_profile)

        # Multimodal gated cross-attention fusion
        fused_logits, gate_weight, fused_rep = self.fusion(spatial_feat, freq_feat)

        # Calibrated probabilities: class 0 = Real, class 1 = Deepfake
        probs = F.softmax(fused_logits, dim=-1)
        deepfake_score = probs[:, 1]

        outputs = {
            "logits": fused_logits,
            "probabilities": probs,
            "deepfake_score": deepfake_score,
            "spatial_logits": spatial_logits,
            "freq_logits": freq_logits,
            "gate_weight": gate_weight,
        }

        if return_intermediates:
            outputs.update({
                "spatial_features": spatial_feat,
                "freq_features": freq_feat,
                "fused_representation": fused_rep,
                "spatial_map": spatial_map,
                "spectrum": spectrum,
                "radial_profile": radial_profile,
            })

        return outputs
