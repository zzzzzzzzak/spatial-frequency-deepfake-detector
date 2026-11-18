"""Temporal pooling algorithms for aggregating video frame scores."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def compute_ema_scores(scores: List[float], decay: float = 0.85) -> List[float]:
    """Compute Exponential Moving Average (EMA) smoothed score sequence.
    
    Formula: s_ema[t] = decay * s_ema[t-1] + (1 - decay) * s[t]
    """
    if not scores:
        return []
    
    ema = [scores[0]]
    for s in scores[1:]:
        val = decay * ema[-1] + (1.0 - decay) * s
        ema.append(float(val))
    return ema


def compute_attention_pooling(scores: List[float], temperature: float = 0.5) -> float:
    """Softmax attention-weighted pooling emphasizing peak anomalous frames.
    
    Weights: w_i = exp(s_i / temp) / sum(exp(s_j / temp))
    Aggregate: S = sum(w_i * s_i)
    """
    if not scores:
        return 0.0
    arr = np.array(scores, dtype=np.float64)
    # Numerical stability
    scaled = (arr - np.max(arr)) / max(temperature, 1e-4)
    weights = np.exp(scaled)
    weights = weights / np.sum(weights)
    return float(np.sum(weights * arr))


def compute_top_k_pooling(scores: List[float], top_ratio: float = 0.20) -> float:
    """Average the top k% highest anomaly scores."""
    if not scores:
        return 0.0
    arr = np.array(scores, dtype=np.float64)
    k = max(1, int(len(arr) * top_ratio))
    top_scores = np.partition(arr, -k)[-k:]
    return float(np.mean(top_scores))


def detect_temporal_flicker(scores: List[float], threshold: float = 0.25) -> Dict[str, float]:
    """Analyze high-frequency score jitter between adjacent frames.
    
    Sudden jumpy variations between consecutive frames often flag temporal
    incoherency typical of frame-by-frame face swap generators.
    """
    if len(scores) < 2:
        return {"mean_flicker": 0.0, "max_flicker": 0.0, "flicker_ratio": 0.0}
    
    arr = np.array(scores, dtype=np.float64)
    diffs = np.abs(np.diff(arr))
    
    mean_flicker = float(np.mean(diffs))
    max_flicker = float(np.max(diffs))
    flicker_ratio = float(np.mean(diffs > threshold))
    
    return {
        "mean_flicker": mean_flicker,
        "max_flicker": max_flicker,
        "flicker_ratio": flicker_ratio,
    }


def aggregate_temporal_scores(
    frame_scores: List[float],
    method: str = "ema",
    decay: float = 0.85,
    top_k_ratio: float = 0.20,
) -> Dict[str, Any]:
    """Produce comprehensive video-level aggregation."""
    if not frame_scores:
        return {
            "video_score": 0.0,
            "mean_score": 0.0,
            "max_score": 0.0,
            "smoothed_scores": [],
            "flicker_stats": {},
        }

    ema_scores = compute_ema_scores(frame_scores, decay=decay)
    attn_score = compute_attention_pooling(frame_scores)
    top_k_score = compute_top_k_pooling(frame_scores, top_ratio=top_k_ratio)
    mean_score = float(np.mean(frame_scores))
    max_score = float(np.max(frame_scores))
    flicker = detect_temporal_flicker(frame_scores)

    if method == "ema":
        video_score = ema_scores[-1]
    elif method == "attention":
        video_score = attn_score
    elif method == "top_k":
        video_score = top_k_score
    else:
        video_score = mean_score

    return {
        "video_score": float(video_score),
        "mean_score": mean_score,
        "max_score": max_score,
        "attention_score": attn_score,
        "top_k_score": top_k_score,
        "smoothed_scores": ema_scores,
        "flicker_stats": flicker,
    }
