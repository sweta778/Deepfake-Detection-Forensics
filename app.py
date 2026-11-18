import os
import tempfile
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from mtcnn import MTCNN

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Deepfake Vision Forensics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for polished dark/light forensic theme
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .verdict-fake {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(185, 28, 28, 0.25));
        border: 2px solid #ef4444;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
    }
    .verdict-real {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.15), rgba(21, 128, 61, 0.25));
        border: 2px solid #22c55e;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
    }
    .card-metric {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .frame-card {
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
if not os.path.exists(MODELS_DIR):
    MODELS_DIR = BASE_DIR

# ---------------------------------------------------------
# Resource Caching: Models & Detectors
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Loading Deep Learning Models & Face Detector...")
def load_detection_resources():
    eff_path = os.path.join(MODELS_DIR, "efficientnet_model.h5")
    xcep_path = os.path.join(MODELS_DIR, "xception_model.h5")
    
    eff_model = None
    xcep_model = None
    
    if os.path.exists(eff_path):
        eff_model = tf.keras.models.load_model(eff_path)
    if os.path.exists(xcep_path):
        xcep_model = tf.keras.models.load_model(xcep_path)
        
    detector = MTCNN()
    return eff_model, xcep_model, detector

eff_model, xcep_model, detector = load_detection_resources()

# ---------------------------------------------------------
# Forensic Helper Functions
# ---------------------------------------------------------
def apply_elliptical_mask(image):
    """Applies elliptical mask to crop background while preserving face boundaries."""
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    axes = (int(w * 0.42), int(h * 0.47))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    masked = cv2.bitwise_and(image, image, mask=mask)
    return masked

def generate_gradcam_heatmap(model, face_normalized):
    """Generates Grad-CAM activation heatmap for face input."""
    try:
        # Check if model has sublayer efficientnetb0
        if "efficientnetb0" in [l.name for l in model.layers]:
            base = model.get_layer("efficientnetb0")
            grad_model = tf.keras.models.Model(
                inputs=model.inputs,
                outputs=[base.get_layer("top_conv").output, model.output]
            )
        else:
            last_conv = None
            for layer in reversed(model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    last_conv = layer.name
                    break
            if last_conv is None:
                return None
            grad_model = tf.keras.models.Model(
                inputs=model.inputs,
                outputs=[model.get_layer(last_conv).output, model.output]
            )
            
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(np.expand_dims(face_normalized, axis=0))
            loss = predictions[:, 0]
            
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()
    except Exception:
        return None

def overlay_heatmap(face_rgb, heatmap, alpha=0.45):
    """Blends Grad-CAM heatmap with original RGB face."""
    if heatmap is None:
        return face_rgb
    h, w = face_rgb.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(face_rgb, 1 - alpha, heatmap_colored, alpha, 0)
    return overlay

def analyze_video_frames(video_path, num_frames=10, threshold=0.50, enable_gradcam=True, progress_bar=None, status_text=None):
    """Samples frames, extracts faces, performs neural inference, and calculates explainability."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    duration_sec = total_frames / fps if fps > 0 else 0
    
    if total_frames <= 0:
        cap.release()
        return None
        
    frame_indices = np.linspace(0, total_frames - 1, min(num_frames, total_frames), dtype=int)
    results = []
    
    active_model = eff_model if eff_model is not None else xcep_model
    
    for i, idx in enumerate(frame_indices):
        if progress_bar:
            progress_bar.progress((i + 1) / len(frame_indices))
        if status_text:
            timestamp = idx / fps if fps > 0 else 0
            status_text.text(f"Analyzing frame {i+1}/{len(frame_indices)} (Time: {timestamp:.2f}s)...")
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
            
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = detector.detect_faces(rgb)
        
        if not detections:
            continue
            
        # Select largest detected face
        best_face = max(detections, key=lambda d: d['box'][2] * d['box'][3])
        x, y, w, h = best_face['box']
        x, y = max(0, x), max(0, y)
        face_crop = rgb[y:y+h, x:x+w]
        if face_crop.size == 0:
            continue
            
        face_resized = cv2.resize(face_crop, (224, 224))
        face_normalized = face_resized.astype("float32") / 255.0
        face_input = np.expand_dims(face_normalized, axis=0)
        
        # Neural inference
        if eff_model is not None and xcep_model is not None:
            p_eff = float(eff_model.predict(face_input, verbose=0)[0][0])
            p_xcep = float(xcep_model.predict(face_input, verbose=0)[0][0])
            prob = 0.5 * p_eff + 0.5 * p_xcep
        elif eff_model is not None:
            prob = float(eff_model.predict(face_input, verbose=0)[0][0])
        else:
            prob = float(xcep_model.predict(face_input, verbose=0)[0][0])
            
        # Optional Grad-CAM
        heatmap_overlay = None
        if enable_gradcam and active_model is not None:
            raw_heatmap = generate_gradcam_heatmap(active_model, face_normalized)
            if raw_heatmap is not None:
                heatmap_overlay = overlay_heatmap(face_resized, raw_heatmap)
                
        # Elliptical mask
        masked_face = apply_elliptical_mask(face_resized)
        
        timestamp = idx / fps if fps > 0 else 0
        results.append({
            "frame_idx": int(idx),
            "timestamp": round(timestamp, 2),
            "fake_prob": float(prob),
            "is_fake": bool(prob > threshold),
            "face_orig": face_resized,
            "face_masked": masked_face,
            "face_gradcam": heatmap_overlay
        })
        
    cap.release()
    return {
        "results": results,
        "total_video_frames": total_frames,
        "fps": fps,
        "duration": duration_sec
    }

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.title("Forensic Controls")
    
    st.markdown("### Active Models")
    if eff_model is not None and xcep_model is not None:
        st.success("🟢 50/50 Ensemble: EfficientNet-B0 + Xception")
    elif eff_model is not None:
        st.success("🟢 EfficientNet-B0 (+Spatial Attention)")
    elif xcep_model is not None:
        st.success("🟢 Xception Deep Learning")
    else:
        st.error("🔴 No model weights found!")
        
    st.markdown("---")
    st.markdown("### Detection Settings")
    
    num_samples = st.slider("Sample Frames to Analyze", min_value=5, max_value=30, value=10, step=1,
                            help="Number of uniformly distributed frames sampled from the video.")
    
    threshold = st.slider("Decision Threshold (Fake)", min_value=0.10, max_value=0.90, value=0.50, step=0.05,
                          help="Probabilities higher than this threshold are classified as FAKE.")
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        if st.button("Standard (0.50)", use_container_width=True):
            threshold = 0.50
    with col_t2:
        if st.button("Celeb-DF (0.35)", use_container_width=True):
            threshold = 0.35
            
    st.markdown("---")
    st.markdown("### Visual Explainability")
    enable_gradcam = st.toggle("Generate Grad-CAM Heatmaps", value=True,
                               help="Highlights spatial regions (seams, eyes, mouth) driving the fake classification.")
    show_masking = st.toggle("Show Background Masking", value=True,
                             help="Displays elliptical background suppression on detected faces.")
    
    st.markdown("---")
    with st.expander("ℹ️ About the Project"):
        st.markdown("""
        **FaceForensics++ Deepfake Forensics**
        * Deep learning verification using spatial attention networks.
        * Analyzes blending seams, facial warping, and boundary artifacts.
        * Validated on FaceForensics++ (95.33% accuracy) & Celeb-DF v2.
        """)

# ---------------------------------------------------------
# Main UI Header
# ---------------------------------------------------------
st.markdown('<div class="main-header">Deepfake Forensics & Verification Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated multi-frame facial manipulation analysis with explainable Grad-CAM heatmaps.</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# Video Source Selection
# ---------------------------------------------------------
input_tab1, input_tab2 = st.tabs(["📁 Upload Video File", "🎬 Use Sample Video"])
selected_video_path = None
uploaded_temp_file = None

with input_tab1:
    uploaded_file = st.file_uploader("Upload video to examine (.mp4, .avi, .mov)", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        selected_video_path = tfile.name
        uploaded_temp_file = tfile.name
        st.video(selected_video_path, format="video/mp4")

with input_tab2:
    sample_files = {
        "Sample 1: Manipulated Video (FaceForensics++ Deepfake)": os.path.join(BASE_DIR, "gradcam_fake_output.mp4"),
        "Sample 2: Clean Authentic Video (Genuine Real Footage)": os.path.join(BASE_DIR, "sample_authentic_real.mp4"),
        "Sample 3: Heatmap Overlay Demonstration (Grad-CAM output)": os.path.join(BASE_DIR, "gradcam_real_output.mp4")
    }
    
    available_samples = {k: v for k, v in sample_files.items() if os.path.exists(v)}
    if available_samples:
        chosen_sample_name = st.selectbox("Choose a preloaded sample video:", list(available_samples.keys()))
        selected_video_path = available_samples[chosen_sample_name]
        st.video(selected_video_path, format="video/mp4")
        if "Heatmap Overlay" in chosen_sample_name:
            st.caption("ℹ️ Note: This sample already has a colored Grad-CAM heatmap baked into the video frames from earlier experiments.")
    else:
        st.info("No sample videos found in repository root.")

# ---------------------------------------------------------
# Forensics Execution Button
# ---------------------------------------------------------
if selected_video_path is not None:
    st.markdown("<br>", unsafe_allow_html=True)
    start_analysis = st.button("🔍 Run Deepfake Forensics Now", type="primary", use_container_width=True)
        
    if start_analysis:
        prog_bar = st.progress(0)
        status_box = st.empty()
        
        with st.spinner("Executing deep learning forensic inspection..."):
            forensic_data = analyze_video_frames(
                selected_video_path,
                num_frames=num_samples,
                threshold=threshold,
                enable_gradcam=enable_gradcam,
                progress_bar=prog_bar,
                status_text=status_box
            )
            
        prog_bar.empty()
        status_box.empty()
        
        if not forensic_data or not forensic_data["results"]:
            st.error("⚠️ No faces were detected in the sampled frames. Ensure the video contains clearly visible frontal faces.")
        else:
            results = forensic_data["results"]
            fake_probs = [r["fake_prob"] for r in results]
            avg_prob = float(np.mean(fake_probs))
            is_verdict_fake = avg_prob > threshold
            confidence = (avg_prob * 100) if is_verdict_fake else ((1.0 - avg_prob) * 100)
            
            # ---------------------------------------------------------
            # Verdict Card
            # ---------------------------------------------------------
            if is_verdict_fake:
                st.markdown(f"""
                <div class="verdict-fake">
                    <h1 style="color: #ef4444; margin: 0; font-size: 2.4rem;">🚨 MANIPULATION DETECTED (FAKE)</h1>
                    <p style="font-size: 1.25rem; margin-top: 10px; color: #fca5a5;">
                        Overall Confidence: <b>{confidence:.2f}%</b> &bull; Average Fake Likelihood: <b>{avg_prob * 100:.2f}%</b>
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="verdict-real">
                    <h1 style="color: #22c55e; margin: 0; font-size: 2.4rem;">✅ AUTHENTIC FOOTAGE (REAL)</h1>
                    <p style="font-size: 1.25rem; margin-top: 10px; color: #86efac;">
                        Overall Confidence: <b>{confidence:.2f}%</b> &bull; Average Fake Likelihood: <b>{avg_prob * 100:.2f}%</b>
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
            # ---------------------------------------------------------
            # Key Metrics Cards
            # ---------------------------------------------------------
            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                st.metric("Final Verdict", "FAKE" if is_verdict_fake else "REAL")
            with m2:
                st.metric("Confidence", f"{confidence:.1f}%")
            with m3:
                st.metric("Avg Fake Probability", f"{avg_prob * 100:.1f}%")
            with m4:
                st.metric("Faces Extracted", f"{len(results)}/{num_samples}")
            with m5:
                st.metric("Video Duration", f"{forensic_data['duration']:.1f}s")
                
            st.markdown("---")
            
            # ---------------------------------------------------------
            # Frame-by-Frame Timeline Chart
            # ---------------------------------------------------------
            st.subheader("📈 Temporal Fake Probability Timeline")
            st.caption("Tracking manipulation probability across video timestamps against the decision threshold.")
            
            chart_df = pd.DataFrame({
                "Timestamp (s)": [f"{r['timestamp']}s" for r in results],
                "Fake Probability (%)": [r["fake_prob"] * 100 for r in results],
                "Threshold (%)": [threshold * 100] * len(results)
            }).set_index("Timestamp (s)")
            
            st.line_chart(chart_df, color=["#ef4444", "#94a3b8"])
            
            # ---------------------------------------------------------
            # Detailed Frame Gallery & Explainability Inspector
            # ---------------------------------------------------------
            st.subheader("🔬 Facial Inspector & Grad-CAM Explainability")
            st.caption("Visual inspection of extracted faces, boundary masks, and class-activation attention heatmaps.")
            
            for i, r in enumerate(results):
                status_badge = "🔴 FAKE" if r["is_fake"] else "🟢 REAL"
                frame_title = f"Frame #{i+1} — Timestamp: {r['timestamp']}s — {status_badge} ({r['fake_prob']*100:.1f}% fake likelihood)"
                
                with st.expander(frame_title, expanded=(i < 3 or r["is_fake"])):
                    col_orig, col_masked, col_cam = st.columns(3)
                    
                    with col_orig:
                        st.markdown("**1. Extracted Face (RGB)**")
                        st.image(r["face_orig"], use_container_width=True)
                        
                    with col_masked:
                        st.markdown("**2. Elliptical Boundary Mask**")
                        if show_masking:
                            st.image(r["face_masked"], use_container_width=True)
                        else:
                            st.info("Masking display disabled in sidebar.")
                            
                    with col_cam:
                        st.markdown("**3. Grad-CAM Activation Heatmap**")
                        if r["face_gradcam"] is not None:
                            st.image(r["face_gradcam"], use_container_width=True)
                        else:
                            st.info("Grad-CAM visualization unavailable or disabled.")
                            
            # ---------------------------------------------------------
            # Downloadable Forensic Report
            # ---------------------------------------------------------
            st.markdown("---")
            st.subheader("📑 Export Forensic Audit Log")
            
            export_df = pd.DataFrame([{
                "Frame_Number": i + 1,
                "Video_Frame_Index": r["frame_idx"],
                "Timestamp_Sec": r["timestamp"],
                "Fake_Probability": round(r["fake_prob"], 4),
                "Classification": "FAKE" if r["is_fake"] else "REAL"
            } for i, r in enumerate(results)])
            
            csv_data = export_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Forensic Audit Log (.CSV)",
                data=csv_data,
                file_name="deepfake_forensic_report.csv",
                mime="text/csv",
                use_container_width=True
            )

# Clean up temp file on unmount/reload if needed
if uploaded_temp_file and os.path.exists(uploaded_temp_file):
    try:
        # Don't delete immediately while session is active; Python tempfile will handle OS cleanup
        pass
    except Exception:
        pass
