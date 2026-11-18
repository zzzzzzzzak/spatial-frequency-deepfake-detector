"""Cross-Attention Gated Multimodal Fusion Module.

Dynamically fuses spatial boundary representations and frequency-domain spectral
anomaly representations via multi-head cross-attention and adaptive gating.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import FusionConfig


class CrossAttentionGatedFusion(nn.Module):
    """Fuses spatial and frequency feature vectors with cross-attention and adaptive gating."""
    def __init__(self, config: FusionConfig):
        super().__init__()
        self.config = config
        hidden_dim = config.hidden_dim

        # Projections to common embedding space
        self.spatial_proj = nn.Sequential(
            nn.Linear(config.spatial_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.freq_proj = nn.Sequential(
            nn.Linear(config.freq_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Multi-Head Cross-Attention (Query: Spatial, Key/Value: Frequency)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm_attn = nn.LayerNorm(hidden_dim)

        # Adaptive Gating Network: computes lambda in [0, 1] balancing spatial vs spectral focus
        self.gate_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Final Classification MLP Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(p=config.dropout),
            nn.Linear(hidden_dim // 2, config.num_classes),
        )

    def forward(
        self, spatial_feat: torch.Tensor, freq_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            spatial_feat: Feature vector from spatial stream (B, spatial_dim).
            freq_feat: Feature vector from frequency stream (B, freq_dim).
            
        Returns:
            logits: Final fused classification logits (B, num_classes).
            gate_weight: Learned dynamic gate weights (B, 1) in [0, 1].
            fused_rep: Fused latent representation (B, hidden_dim).
        """
        # Project to common space (B, 1, hidden_dim) for attention
        s_emb = self.spatial_proj(spatial_feat).unsqueeze(1)
        f_emb = self.freq_proj(freq_feat).unsqueeze(1)

        # Cross-attention: Spatial queries Frequency
        attn_out, _ = self.cross_attention(query=s_emb, key=f_emb, value=f_emb)
        s_attended = self.norm_attn(s_emb + attn_out).squeeze(1)  # (B, hidden_dim)
        f_sq = f_emb.squeeze(1)

        # Compute dynamic gating factor
        gate_input = torch.cat([s_attended, f_sq], dim=-1)
        gate_weight = self.gate_fc(gate_input)  # (B, 1)

        # Adaptive gated combination
        fused_rep = gate_weight * s_attended + (1.0 - gate_weight) * f_sq

        # Classification logits
        logits = self.classifier(fused_rep)

        return logits, gate_weight, fused_rep
