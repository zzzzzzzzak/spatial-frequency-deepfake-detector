"""OpenCV video frame extraction and metadata inspection."""

from dataclasses import dataclass
import os
from typing import Generator, List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class VideoMetadata:
    """Video stream characteristics."""
    filepath: str
    total_frames: int
    fps: float
    duration_seconds: float
    width: int
    height: int


class VideoFrameExtractor:
    """Extracts frames from video files with stride control and sampling limits."""
    def __init__(self, frame_stride: int = 2, max_frames: int = 120):
        self.frame_stride = max(1, frame_stride)
        self.max_frames = max_frames

    def get_metadata(self, video_path: str) -> VideoMetadata:
        """Read video stream metadata without loading all frames into RAM."""
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video stream: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if total_frames > 0 else 0.0
        cap.release()

        return VideoMetadata(
            filepath=video_path,
            total_frames=total_frames,
            fps=fps,
            duration_seconds=duration,
            width=width,
            height=height,
        )

    def extract_frames(
        self, video_path: str
    ) -> Tuple[List[np.ndarray], List[float]]:
        """Extract RGB frames and timestamps at the specified stride.
        
        Returns:
            frames: List of RGB numpy arrays (H, W, 3).
            timestamps: List of frame timestamp offsets in seconds.
        """
        metadata = self.get_metadata(video_path)
        cap = cv2.VideoCapture(video_path)

        frames = []
        timestamps = []
        frame_idx = 0

        while cap.isOpened() and len(frames) < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % self.frame_stride == 0:
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(rgb_frame)
                timestamps.append(frame_idx / metadata.fps)

            frame_idx += 1

        cap.release()
        return frames, timestamps


def create_synthetic_test_video(
    output_path: str,
    num_frames: int = 30,
    width: int = 128,
    height: int = 128,
    fps: int = 24,
) -> str:
    """Generate a lightweight synthetic mp4/avi video for testing."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for i in range(num_frames):
        # Create shifting synthetic pattern
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        color = int((i / num_frames) * 200 + 40)
        cv2.circle(frame, (width // 2, height // 2), 20 + (i % 10), (color, 120, 200), -1)
        writer.write(frame)

    writer.release()
    return output_path
