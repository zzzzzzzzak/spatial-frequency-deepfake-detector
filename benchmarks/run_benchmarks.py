"""Offline benchmark suite evaluating accuracy, EER, AUC-ROC, and inference latency."""

import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from typing import Any, Dict, List, Tuple
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)
import torch
import torch.nn.functional as F

from src.config import DetectorConfig
from src.models.dual_stream_detector import DualStreamDetector
from src.data.synthetic_generator import generate_benchmark_dataset


def compute_equal_error_rate(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    """Compute Equal Error Rate (EER) where False Acceptance Rate == False Rejection Rate."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1.0 - tpr
    # Find intersection point where |fpr - fnr| is minimized
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    eer_threshold = float(thresholds[idx]) if idx < len(thresholds) else 0.50
    return eer, eer_threshold


def run_benchmarks(
    num_samples: int = 100,
    size: int = 256,
    device: str = "cpu",
    output_json: str = "benchmarks/benchmark_results.json",
) -> Dict[str, Any]:
    """Run full forensic evaluation and benchmark suite."""
    print(f"Generating synthetic forensic test dataset ({num_samples} samples)...")
    images, labels, artifact_types = generate_benchmark_dataset(num_samples=num_samples, size=size, seed=42)
    y_true = labels.numpy()

    print("Initializing Dual-Stream Detector...")
    cfg = DetectorConfig()
    dev = torch.device(device)
    model = DualStreamDetector(cfg).to(dev)

    ckpt_path = "checkpoints/calibrated_weights.pt"
    if not os.path.isfile(ckpt_path):
        print("Calibrated weights not found. Running self-contained fast calibration...")
        from src.train import train_calibrated_model
        model, _ = train_calibrated_model(num_samples=120, epochs=4, save_path=ckpt_path, device=device)
    else:
        print(f"Loading calibrated weights from: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=dev, weights_only=True))

    model.eval()

    # Inference & latency measurements
    latencies = []
    fused_probs = []
    spatial_probs = []
    freq_probs = []

    print("Executing benchmark evaluation passes...")
    with torch.no_grad():
        for i in range(len(images)):
            x = images[i:i + 1].to(dev)
            
            t0 = time.perf_counter()
            out = model(x)
            t1 = time.perf_counter()
            
            latencies.append((t1 - t0) * 1000.0)  # ms
            fused_probs.append(float(out["deepfake_score"][0].cpu().item()))
            spatial_probs.append(float(F.softmax(out["spatial_logits"], dim=-1)[0, 1].cpu().item()))
            freq_probs.append(float(F.softmax(out["freq_logits"], dim=-1)[0, 1].cpu().item()))

    y_score = np.array(fused_probs)
    y_pred = (y_score >= 0.50).astype(int)

    # Metrics
    auc = float(roc_auc_score(y_true, y_score))
    eer, eer_thresh = compute_equal_error_rate(y_true, y_score)
    brier = float(brier_score_loss(y_true, y_score))
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # Latencies
    p50_lat = float(np.percentile(latencies, 50))
    p95_lat = float(np.percentile(latencies, 95))
    p99_lat = float(np.percentile(latencies, 99))

    # Modality comparison
    sp_f1 = float(f1_score(y_true, (np.array(spatial_probs) >= 0.50).astype(int), zero_division=0))
    fr_f1 = float(f1_score(y_true, (np.array(freq_probs) >= 0.50).astype(int), zero_division=0))

    results = {
        "num_samples": num_samples,
        "device": device,
        "roc_auc": round(auc, 4),
        "equal_error_rate": round(eer, 4),
        "eer_threshold": round(eer_thresh, 4),
        "brier_score": round(brier, 4),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "modality_comparison": {
            "spatial_only_f1": round(sp_f1, 4),
            "frequency_only_f1": round(fr_f1, 4),
            "fused_dual_stream_f1": round(f1, 4),
        },
        "latency_ms": {
            "p50": round(p50_lat, 2),
            "p95": round(p95_lat, 2),
            "p99": round(p99_lat, 2),
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)

    # Pretty print summary
    print("\n" + "=" * 60)
    print(" DUAL-STREAM DEEPFAKE FORENSICS BENCHMARK REPORT")
    print("=" * 60)
    print(f" Samples Evaluated:     {num_samples} (50% Authentic, 50% Manipulated)")
    print(f" ROC-AUC:               {auc:.4f}")
    print(f" Equal Error Rate (EER):{eer * 100:.2f}% (Threshold: {eer_thresh:.4f})")
    print(f" Brier Score:           {brier:.4f} (Calibration Quality)")
    print(f" Accuracy:              {acc * 100:.2f}%")
    print(f" F1 Score:              {f1:.4f}")
    print(f" Precision:             {prec:.4f}")
    print(f" Recall:                {rec:.4f}")
    print("-" * 60)
    print(f" Latency p50:           {p50_lat:.2f} ms / frame")
    print(f" Latency p95:           {p95_lat:.2f} ms / frame")
    print(f" Latency p99:           {p99_lat:.2f} ms / frame")
    print("=" * 60)
    print(f"Saved benchmark report to: {output_json}\n")

    return results


if __name__ == "__main__":
    run_benchmarks(num_samples=100, device="cpu")
