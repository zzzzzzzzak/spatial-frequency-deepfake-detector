"""Synthetic forensic data generation for reproducible testing and evaluation."""

from src.data.synthetic_generator import (
    generate_synthetic_face,
    generate_authentic_face,
    generate_benchmark_dataset,
    SyntheticForensicsDataset,
)

__all__ = [
    "generate_synthetic_face",
    "generate_authentic_face",
    "generate_benchmark_dataset",
    "SyntheticForensicsDataset",
]
