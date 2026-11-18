"""Unified Production Inference Engine for Image and Video Deepfake Forensics.

Supports single images, batches, and video streams with full Grad-CAM visual
attribution and frequency-domain spectral analysis.
"""

import argparse
from dataclasses import asdict, dataclass
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from src.config import DetectorConfig
from src.models.dual_stream_detector import DualStreamDetector
from src.transforms.image_transforms import preprocess_image
from src.transforms.spectral_transforms import (
    compute_2d_fft,
    compute_azimuthal_average,
)
from src.explainability.gradcam import GradCAM, generate_cam_overlay
from src.explainability.spectral_forensics import analyze_spectral_profile
from src.video.frame_extractor import VideoFrameExtractor
from src.video.temporal_pooling import aggregate_temporal_scores


@dataclass
class DetectionResult:
    """Structured forensic detection result."""
    media_type: str  # 'image' or 'video'
    is_deepfake: bool
    confidence: float
    decision_threshold: float
    spatial_score: float
    frequency_score: float
    gate_weight_spatial: float
    spectral_divergence: float
    fitted_alpha: float
    spectral_profile: Dict[str, Any]
    frame_scores: Optional[List[float]] = None
    timeline_stats: Optional[Dict[str, Any]] = None
    worst_frame_idx: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to serializable dictionary."""
        d = asdict(self)
        return d


class DeepfakeInferenceEngine:
    """Production inference engine for dual-stream deepfake detection."""
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        config: Optional[DetectorConfig] = None,
        device: str = "auto",
    ):
        self.config = config or DetectorConfig()
        
        # Determine device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize Model
        self.model = DualStreamDetector(self.config).to(self.device)
        self.model.eval()

        default_ckpt = "checkpoints/calibrated_weights.pt"
        target_ckpt = checkpoint_path if (checkpoint_path and os.path.isfile(checkpoint_path)) else (default_ckpt if os.path.isfile(default_ckpt) else None)
        if target_ckpt:
            state_dict = torch.load(target_ckpt, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)

        # Explainability & video submodules
        self.cam_extractor = GradCAM(self.model, self.model.get_gradcam_target_layer())
        self.frame_extractor = VideoFrameExtractor(
            frame_stride=self.config.temporal.frame_stride,
            max_frames=self.config.temporal.max_frames,
        )

    def predict_image(
        self,
        image_input: Union[str, np.ndarray, Image.Image],
        generate_explainability: bool = True,
    ) -> Tuple[DetectionResult, Optional[np.ndarray], Optional[np.ndarray]]:
        """Run complete forensic inspection on a single image.
        
        Args:
            image_input: File path, numpy array, or PIL Image.
            generate_explainability: Whether to compute Grad-CAM and spectral profiles.
            
        Returns:
            result: Structured DetectionResult dataclass.
            cam_heatmap: Optional 2D float heatmap in [0, 1].
            cam_overlay: Optional RGB blended image array.
        """
        # Load raw image for visualization
        if isinstance(image_input, str):
            raw_pil = Image.open(image_input).convert("RGB")
            raw_np = np.array(raw_pil)
        elif isinstance(image_input, Image.Image):
            raw_pil = image_input.convert("RGB")
            raw_np = np.array(raw_pil)
        else:
            raw_np = image_input
            raw_pil = Image.fromarray(raw_np.astype(np.uint8))

        tensor = preprocess_image(
            raw_pil,
            target_size=(self.config.spectral.image_size, self.config.spectral.image_size),
            normalize=True,
            device=self.device,
        )

        with torch.no_grad():
            outputs = self.model(tensor)
            deepfake_prob = float(outputs["deepfake_score"][0].cpu().item())
            spatial_prob = float(F.softmax(outputs["spatial_logits"], dim=-1)[0, 1].cpu().item())
            freq_prob = float(F.softmax(outputs["freq_logits"], dim=-1)[0, 1].cpu().item())
            gate_spatial = float(outputs["gate_weight"][0, 0].cpu().item())

        # Spectral profile analysis
        _, spec = compute_2d_fft(tensor, shift=True, log_scale=True)
        rad_profile_tensor = compute_azimuthal_average(
            spec, num_bins=self.config.spectral.num_radial_bins
        )
        rad_profile_np = rad_profile_tensor[0].cpu().numpy()
        spectral_meta = analyze_spectral_profile(rad_profile_np)

        cam_heatmap = None
        cam_overlay = None

        if generate_explainability:
            try:
                # Grad-CAM requires grad on input
                cam_tensor = self.cam_extractor.generate(tensor, target_class=1)
                cam_heatmap = cam_tensor.cpu().numpy()
                cam_overlay = generate_cam_overlay(raw_np, cam_heatmap)
            except Exception:
                cam_heatmap = np.zeros((raw_np.shape[0], raw_np.shape[1]), dtype=np.float32)
                cam_overlay = raw_np

        is_deepfake = deepfake_prob >= self.config.threshold

        result = DetectionResult(
            media_type="image",
            is_deepfake=is_deepfake,
            confidence=deepfake_prob,
            decision_threshold=self.config.threshold,
            spatial_score=spatial_prob,
            frequency_score=freq_prob,
            gate_weight_spatial=gate_spatial,
            spectral_divergence=spectral_meta["spectral_divergence"],
            fitted_alpha=spectral_meta["fitted_alpha"],
            spectral_profile=spectral_meta,
        )

        return result, cam_heatmap, cam_overlay

    def predict_video(
        self,
        video_path: str,
        temporal_method: Optional[str] = None,
    ) -> Tuple[DetectionResult, Optional[np.ndarray], Optional[np.ndarray]]:
        """Run temporal forensic inspection across video frames.
        
        Returns:
            result: Aggregated DetectionResult.
            worst_frame_heatmap: Grad-CAM heatmap of the most anomalous frame.
            worst_frame_overlay: Visual overlay of the most anomalous frame.
        """
        frames, timestamps = self.frame_extractor.extract_frames(video_path)
        if not frames:
            raise ValueError(f"No frames could be extracted from: {video_path}")

        method = temporal_method or self.config.temporal.pooling_method
        frame_scores = []
        spatial_scores = []
        freq_scores = []

        batch_tensors = []
        for f in frames:
            t = preprocess_image(
                f,
                target_size=(self.config.spectral.image_size, self.config.spectral.image_size),
                normalize=True,
                device=self.device,
            )
            batch_tensors.append(t)

        # Batch forward passes
        batch_size = 16
        with torch.no_grad():
            for i in range(0, len(batch_tensors), batch_size):
                sub_batch = torch.cat(batch_tensors[i:i + batch_size], dim=0)
                out = self.model(sub_batch)
                scores = out["deepfake_score"].cpu().tolist()
                s_scores = F.softmax(out["spatial_logits"], dim=-1)[:, 1].cpu().tolist()
                f_scores = F.softmax(out["freq_logits"], dim=-1)[:, 1].cpu().tolist()
                frame_scores.extend(scores)
                spatial_scores.extend(s_scores)
                freq_scores.extend(f_scores)

        # Temporal aggregation
        timeline = aggregate_temporal_scores(
            frame_scores,
            method=method,
            decay=self.config.temporal.ema_decay,
            top_k_ratio=self.config.temporal.top_k_ratio,
        )
        video_confidence = timeline["video_score"]
        is_deepfake = video_confidence >= self.config.threshold

        # Find most anomalous frame for explainability snapshot
        worst_idx = int(np.argmax(frame_scores))
        worst_frame = frames[worst_idx]
        worst_tensor = batch_tensors[worst_idx]

        try:
            cam_tensor = self.cam_extractor.generate(worst_tensor, target_class=1)
            cam_heatmap = cam_tensor.cpu().numpy()
            cam_overlay = generate_cam_overlay(worst_frame, cam_heatmap)
        except Exception:
            cam_heatmap = None
            cam_overlay = worst_frame

        # Spectral analysis of worst frame
        with torch.no_grad():
            _, spec = compute_2d_fft(worst_tensor, shift=True, log_scale=True)
            rad_profile_tensor = compute_azimuthal_average(
                spec, num_bins=self.config.spectral.num_radial_bins
            )
            rad_profile_np = rad_profile_tensor[0].cpu().numpy()
            spectral_meta = analyze_spectral_profile(rad_profile_np)

        result = DetectionResult(
            media_type="video",
            is_deepfake=is_deepfake,
            confidence=video_confidence,
            decision_threshold=self.config.threshold,
            spatial_score=float(np.mean(spatial_scores)),
            frequency_score=float(np.mean(freq_scores)),
            gate_weight_spatial=0.5,
            spectral_divergence=spectral_meta["spectral_divergence"],
            fitted_alpha=spectral_meta["fitted_alpha"],
            spectral_profile=spectral_meta,
            frame_scores=frame_scores,
            timeline_stats=timeline,
            worst_frame_idx=worst_idx,
        )

        return result, cam_heatmap, cam_overlay


def main():
    """Command-line interface for deepfake inference."""
    parser = argparse.ArgumentParser(description="Spatial-Frequency Deepfake Detection CLI")
    parser.add_argument("--input", type=str, required=True, help="Path to input image or video")
    parser.add_argument("--threshold", type=float, default=0.50, help="Classification decision threshold")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu, cuda, auto)")
    parser.add_argument("--output-json", type=str, default=None, help="Save structured detection JSON")
    parser.add_argument("--output-overlay", type=str, default=None, help="Save Grad-CAM visual overlay image")
    args = parser.parse_args()

    cfg = DetectorConfig(threshold=args.threshold)
    engine = DeepfakeInferenceEngine(config=cfg, device=args.device)

    is_video = args.input.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm"))
    if is_video:
        result, heatmap, overlay = engine.predict_video(args.input)
    else:
        result, heatmap, overlay = engine.predict_image(args.input)

    status = "AI-GENERATED / DEEPFAKE" if result.is_deepfake else "AUTHENTIC MEDIA"
    print(f"\n=======================================================")
    print(f" Forensic Assessment: {status}")
    print(f" Confidence Score:    {result.confidence:.4f} (Threshold: {result.decision_threshold})")
    print(f" Spatial Score:       {result.spatial_score:.4f}")
    print(f" Frequency Score:     {result.frequency_score:.4f}")
    print(f" Spectral Divergence: {result.spectral_divergence:.4f} (Fitted Alpha: {result.fitted_alpha:.2f})")
    print(f"=======================================================\n")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"Saved JSON report to: {args.output_json}")

    if args.output_overlay and overlay is not None:
        Image.fromarray(overlay).save(args.output_overlay)
        print(f"Saved visual attribution overlay to: {args.output_overlay}")


if __name__ == "__main__":
    main()
