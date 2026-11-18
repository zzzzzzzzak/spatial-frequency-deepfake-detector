"""Image preprocessing and forensic data augmentation transforms."""

from typing import Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def preprocess_image(
    image: Union[np.ndarray, Image.Image, torch.Tensor],
    target_size: Tuple[int, int] = (256, 256),
    normalize: bool = True,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Preprocess an input image into a standardized float32 tensor (1, 3, H, W).
    
    Args:
        image: Input RGB image as numpy array (H, W, 3), PIL Image, or torch Tensor.
        target_size: Tuple of (height, width).
        normalize: Whether to apply ImageNet mean and standard deviation normalization.
        device: PyTorch device to place tensor on.
        
    Returns:
        Tensor of shape (1, 3, H, W).
    """
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))

    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = image[:, :, :3]
        
        # Ensure HWC float32 in [0, 1]
        if image.dtype == np.uint8:
            tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        else:
            tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        tensor = tensor.unsqueeze(0)
    elif isinstance(image, torch.Tensor):
        tensor = image.float()
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    # Resize to target dimension
    if tensor.shape[-2:] != target_size:
        tensor = F.interpolate(tensor, size=target_size, mode="bilinear", align_corners=False)

    if device is not None:
        tensor = tensor.to(device)

    if normalize:
        mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=tensor.device).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std

    return tensor


def simulate_compression_and_blur(
    x: torch.Tensor,
    blur_kernel_size: int = 5,
    blur_sigma: float = 1.2,
    noise_level: float = 0.02,
) -> torch.Tensor:
    """Apply differentiable blur and additive noise to test perturbation robustness."""
    # Create 1D Gaussian kernel
    radius = blur_kernel_size // 2
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32, device=x.device)
    kernel_1d = torch.exp(-(coords**2) / (2 * blur_sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = (kernel_1d.unsqueeze(1) @ kernel_1d.unsqueeze(0)).view(1, 1, blur_kernel_size, blur_kernel_size)
    
    b, c, h, w = x.shape
    kernel_2d = kernel_2d.repeat(c, 1, 1, 1)
    
    # Depthwise 2D conv
    blurred = F.conv2d(x, kernel_2d, padding=radius, groups=c)
    
    # Additive Gaussian noise
    noise = torch.randn_like(blurred) * noise_level
    perturbed = torch.clamp(blurred + noise, 0.0, 1.0)
    return perturbed
