"""Streamlit Visual Forensics Dashboard for Spatial-Frequency Deepfake Detection."""

import io
import json
import os
import tempfile
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import streamlit as st
import torch

from src.config import DetectorConfig
from src.inference import DeepfakeInferenceEngine
from src.data.synthetic_generator import (
    generate_authentic_face,
    generate_synthetic_face,
)
from src.video.frame_extractor import create_synthetic_test_video

st.set_page_config(
    page_title="Dual-Stream Deepfake Forensics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .status-badge-fake {
        background-color: #fee2e2;
        color: #991b1b;
        font-weight: 700;
        padding: 8px 16px;
        border-radius: 6px;
        font-size: 1.2rem;
        display: inline-block;
    }
    .status-badge-real {
        background-color: #dcfce7;
        color: #166534;
        font-weight: 700;
        padding: 8px 16px;
        border-radius: 6px;
        font-size: 1.2rem;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_inference_engine(threshold: float = 0.50):
    cfg = DetectorConfig(threshold=threshold)
    engine = DeepfakeInferenceEngine(config=cfg, device="cpu")
    return engine


def main():
    st.title("🛡️ Spatial-Frequency Deepfake Forensics Platform")
    st.markdown(
        "**Multi-modal AI Authenticity Verification** combining spatial boundary inspection "
        "and 2D FFT azimuthal spectral decay forensics."
    )

    # Sidebar Controls
    st.sidebar.header("Forensic Settings")
    threshold = st.sidebar.slider("Decision Threshold", 0.10, 0.90, 0.50, 0.05)
    temporal_method = st.sidebar.selectbox("Video Temporal Aggregation", ["ema", "attention", "top_k", "mean"])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Input Source")
    input_source = st.sidebar.radio(
        "Select Source",
        ["Curated Benchmark Samples", "Upload Image/Video"]
    )

    engine = get_inference_engine(threshold=threshold)

    media_to_analyze = None
    media_type = "image"

    if input_source == "Curated Benchmark Samples":
        sample_choice = st.sidebar.selectbox(
            "Benchmark Case",
            [
                "Authentic Photographic Texture (1/f² Spectrum)",
                "StyleGAN-style Checkerboard Artifact (Transposed Conv)",
                "Poisson Face-Swap Blending Seam (Spatial Discontinuity)",
                "Autoencoder Over-smoothed Face (High-Frequency Roll-off)",
                "Video Stream (Temporal Face Inconsistency)",
            ],
        )

        if sample_choice == "Authentic Photographic Texture (1/f² Spectrum)":
            media_to_analyze = generate_authentic_face(256, seed=101)
            media_type = "image"
        elif sample_choice == "StyleGAN-style Checkerboard Artifact (Transposed Conv)":
            media_to_analyze = generate_synthetic_face(256, artifact_type="checkerboard", seed=202)
            media_type = "image"
        elif sample_choice == "Poisson Face-Swap Blending Seam (Spatial Discontinuity)":
            media_to_analyze = generate_synthetic_face(256, artifact_type="blending_seam", seed=303)
            media_type = "image"
        elif sample_choice == "Autoencoder Over-smoothed Face (High-Frequency Roll-off)":
            media_to_analyze = generate_synthetic_face(256, artifact_type="frequency_rolloff", seed=404)
            media_type = "image"
        else:
            with tempfile.NamedTemporaryFile(suffix=".avi", delete=False) as tmp_vid:
                create_synthetic_test_video(tmp_vid.name, num_frames=30)
                media_to_analyze = tmp_vid.name
                media_type = "video"

    else:
        uploaded_file = st.sidebar.file_uploader(
            "Upload Media File", type=["png", "jpg", "jpeg", "mp4", "avi", "mov"]
        )
        if uploaded_file is not None:
            if uploaded_file.name.lower().endswith((".mp4", ".avi", ".mov")):
                with tempfile.NamedTemporaryFile(suffix=os.path.splitext(uploaded_file.name)[1], delete=False) as tmp_vid:
                    tmp_vid.write(uploaded_file.read())
                    media_to_analyze = tmp_vid.name
                    media_type = "video"
            else:
                media_to_analyze = np.array(Image.open(uploaded_file).convert("RGB"))
                media_type = "image"

    if media_to_analyze is None:
        st.info("👈 Please select or upload a media sample from the sidebar to begin forensic analysis.")
        return

    # Execute Inference
    with st.spinner("Executing dual-stream forensic inspection..."):
        if media_type == "video":
            result, heatmap, overlay = engine.predict_video(media_to_analyze, temporal_method=temporal_method)
        else:
            result, heatmap, overlay = engine.predict_image(media_to_analyze, generate_explainability=True)

    # Top Verdict Banner
    st.markdown("### Forensic Assessment")
    col_verdict, col_conf, col_sp, col_fr = st.columns(4)

    with col_verdict:
        if result.is_deepfake:
            st.markdown('<div class="status-badge-fake">🚨 AI-GENERATED / DEEPFAKE</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-badge-real">✅ AUTHENTIC MEDIA</div>', unsafe_allow_html=True)

    with col_conf:
        st.metric("Deepfake Probability", f"{result.confidence * 100:.1f}%", f"Threshold: {result.decision_threshold:.2f}")

    with col_sp:
        st.metric("Spatial Stream Score", f"{result.spatial_score * 100:.1f}%")

    with col_fr:
        st.metric("Frequency Stream Score", f"{result.frequency_score * 100:.1f}%")

    st.markdown("---")

    # Forensic Inspection Tabs
    tab_spatial, tab_spectral, tab_video, tab_audit = st.tabs(
        ["🖼️ Spatial Forensics (Grad-CAM)", "📊 Frequency Forensics (2D FFT)", "🎞️ Temporal Timeline", "📑 Audit Report"]
    )

    with tab_spatial:
        st.subheader("Spatial Boundary & Blending Attribution")
        st.markdown(
            "Grad-CAM visual attribution highlighting localized pixel regions that triggered "
            "the synthetic classification (e.g., blending seams, unnatural contours, or facial anomalies)."
        )
        col1, col2 = st.columns(2)
        with col1:
            if media_type == "image":
                st.image(media_to_analyze, caption="Input Media", use_container_width=True)
            else:
                st.image(overlay, caption="Key Representative Frame", use_container_width=True)

        with col2:
            if overlay is not None:
                st.image(overlay, caption="Overlaid Grad-CAM Visual Attribution", use_container_width=True)
            else:
                st.info("Attribution overlay unavailable.")

    with tab_spectral:
        st.subheader("Frequency-Domain Spectral Forensics")
        st.markdown(
            "Natural photographic images follow an optical power law $P(r) \sim 1/r^{\\alpha}$ (where $\\alpha \\approx 2.0$). "
            "Generative upsampling creates periodic peaks, while autoencoders cause abnormal high-frequency roll-off."
        )

        col_spec_stat1, col_spec_stat2, col_spec_stat3 = st.columns(3)
        with col_spec_stat1:
            st.metric("Fitted Radial Exponent (α)", f"{result.fitted_alpha:.2f}", "Expected: 1.8 - 2.2")
        with col_spec_stat2:
            st.metric("High-Frequency Divergence", f"{result.spectral_divergence:.4f}", "Anomalous if > 0.40")
        with col_spec_stat3:
            peaks = result.spectral_profile.get("periodic_peaks", [])
            st.metric("Periodic Nyquist Peaks", f"{len(peaks)} detected")

        # Plot Radial Azimuthal Curve vs Natural Reference
        fig, ax = plt.subplots(figsize=(8, 4))
        radii = result.spectral_profile.get("radii", [])
        emp_power = result.spectral_profile.get("empirical_power", [])
        theo_power = result.spectral_profile.get("theoretical_power", [])

        if radii and emp_power:
            ax.plot(radii, emp_power, label="Empirical Azimuthal Spectrum", color="#2563eb", lw=2)
            if theo_power:
                ax.plot(radii, theo_power, label=f"Theoretical Natural Decay (α={result.fitted_alpha:.2f})", color="#dc2626", linestyle="--", lw=1.5)
            ax.set_xlabel("Normalized Radial Frequency (r / R_max)")
            ax.set_ylabel("Normalized Azimuthal Power P(r)")
            ax.set_title("Azimuthal Radial Power Spectrum vs Natural Reference Baseline")
            ax.grid(True, alpha=0.3)
            ax.legend()
            st.pyplot(fig)

    with tab_video:
        st.subheader("Temporal Frame-by-Frame Stability Analysis")
        if media_type == "video" and result.frame_scores:
            st.line_chart(result.frame_scores)
            flicker = result.timeline_stats.get("flicker_stats", {})
            col_f1, col_f2 = st.columns(2)
            col_f1.metric("Mean Inter-Frame Flicker", f"{flicker.get('mean_flicker', 0.0):.4f}")
            col_f2.metric("Max Inter-Frame Flicker", f"{flicker.get('max_flicker', 0.0):.4f}")
        else:
            st.info("Single image analyzed. Temporal timeline is populated for multi-frame video inputs.")

    with tab_audit:
        st.subheader("Structured Forensic Audit Log")
        json_str = json.dumps(result.to_dict(), indent=2)
        st.download_button(
            label="📥 Download Forensic Audit JSON",
            data=json_str,
            file_name="forensic_report.json",
            mime="application/json",
        )
        st.json(result.to_dict())


if __name__ == "__main__":
    main()
