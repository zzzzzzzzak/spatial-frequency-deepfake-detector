"""Gradient-weighted Class Activation Mapping (Grad-CAM) for Spatial Stream.

Produces visual attribution heatmaps highlighting facial regions that triggered
the synthetic media classification decision (e.g., blending seams, unnatural
textures around eyes, mouth, or boundary contours).
"""

from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """Computes Grad-CAM heatmaps for a designated convolutional layer."""
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self._hooks.extend([h1, h2])

    def remove_hooks(self):
        """Cleanly deregister hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int = 1,  # 1 = Deepfake
    ) -> torch.Tensor:
        """Generate Grad-CAM activation map.
        
        Args:
            input_tensor: Input image tensor (1, 3, H, W).
            target_class: Target class index to backpropagate.
            
        Returns:
            cam: 2D normalized heatmap tensor of shape (H, W) in [0, 1].
        """
        self.model.eval()
        self.model.zero_grad()
        
        # Ensure tensor requires grad
        if not input_tensor.requires_grad:
            input_tensor = input_tensor.clone().detach().requires_grad_(True)

        outputs = self.model(input_tensor)
        logits = outputs["logits"]
        score = logits[0, target_class]
        
        # Backpropagate target class score
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks failed to capture activations or gradients.")

        # Global average pooling of gradients: weights alpha_k
        alpha = torch.mean(self.gradients, dim=(-2, -1), keepdim=True)  # (1, C, 1, 1)
        
        # Linear combination of forward activation maps
        cam = torch.sum(alpha * self.activations, dim=1, keepdim=True)  # (1, 1, H', W')
        cam = F.relu(cam)

        # Upsample to input resolution
        _, _, h, w = input_tensor.shape
        cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach()

        # Min-max normalization
        c_min = cam.min()
        c_max = cam.max()
        if c_max > c_min:
            cam = (cam - c_min) / (c_max - c_min)
        else:
            cam = torch.zeros_like(cam)

        return cam


def generate_cam_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap_name: str = "jet",
) -> np.ndarray:
    """Blend a Grad-CAM heatmap over an RGB image.
    
    Args:
        image: Original image as uint8 (H, W, 3) or float32 in [0, 1].
        heatmap: 2D float array (H, W) with values in [0, 1].
        alpha: Blending weight of heatmap over image.
        colormap_name: Color map for heatmap rendering.
        
    Returns:
        RGB image with overlaid colored heatmap as uint8 array (H, W, 3).
    """
    import matplotlib.cm as cm

    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255.0).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

    h, w, _ = image.shape
    if heatmap.shape != (h, w):
        # Resize heatmap if needed
        from PIL import Image
        hm_img = Image.fromarray((heatmap * 255).astype(np.uint8))
        hm_img = hm_img.resize((w, h), Image.BILINEAR)
        heatmap = np.array(hm_img).astype(np.float32) / 255.0

    # Apply colormap
    import matplotlib
    if hasattr(matplotlib, "colormaps"):
        cmap = matplotlib.colormaps[colormap_name]
    else:
        cmap = cm.get_cmap(colormap_name)
    colored_cam = cmap(heatmap)[:, :, :3]  # Strip alpha, float [0, 1]
    colored_cam_uint8 = (colored_cam * 255.0).astype(np.uint8)

    # Blend
    overlay = (alpha * colored_cam_uint8 + (1.0 - alpha) * image).astype(np.uint8)
    return overlay
