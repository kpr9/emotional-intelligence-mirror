import streamlit as st
import tempfile
import os
import pandas as pd

from utils.audio import load_audio, segment_audio
from core.analyzer import analyze_segments, load_model
from core.report import generate_report
from visuals.charts import (
    plot_signal_timeline,
    plot_acoustic_timeline,
    plot_emotion_distribution,
)

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Interview EI Mirror",
    page_icon="🎙️",
    layout="wide",
)

# ── Load CSS ─────────────────────────────────────────────────────────────────

css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────

st.markdown('<p class="main-header">🎙️ Interview Emotional Intelligence Mirror</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Upload a recorded mock interview. '
    'Get a vocal self-awareness report — designed for candidates, not recruiters.</p>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="ethics-note">⚠️ <strong>Ethics:</strong> This tool is for <strong>candidate self-improvement only</strong>. '
    'Output is never used in hiring decisions. Your audio is processed locally and not stored.</div>',
    unsafe_allow_html=True,
)

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Settings")

    segment_duration = st.slider(
        "Segment duration (seconds)",
        min_value=5,
        max_value=30,
        value=10,
        step=5,
        help="How long each audio chunk is. Shorter = more granular. Longer = more stable.",
    )

    st.markdown("---")
    st.markdown("**How it works:**")
    st.markdown(
        "1. Upload your mock interview audio\n"
        "2. Audio splits into segments\n"
        "3. Each segment: emotion classified + acoustic features extracted\n"
        "4. Scores blended (50% model + 50% acoustics)\n"
        "5. Coaching report generated"
    )

    st.markdown("---")
    st.markdown("**Signal Guide:**")
    st.markdown("🔴 **Stress Index** — nervousness level")
    st.markdown("🔵 **Confidence Score** — composure")
    st.markdown("🟢 **Vocal Stability** — voice consistency")
    st.markdown("🟠 **Engagement Level** — energy and interest")

    st.markdown("---")
    st.caption("Model: wav2vec2 (IEMOCAP-trained)\nScores are directional, not clinical.")

# ── File Upload ───────────────────────────────────────────────────────────────

uploaded_file = st.file_uploader(
    "Upload interview audio",
    type=["wav", "mp3", "m4a", "ogg", "flac"],
    help="Supported formats: WAV, MP3, M4A, OGG, FLAC"
)

if uploaded_file is None:
    st.info("👆 Upload a mock interview recording to get started.")
    st.stop()

# ── Pre-load model ────────────────────────────────────────────────────────────

with st.spinner("Loading emotion model (first run takes ~30 seconds)..."):
    load_model()

# ── Save uploaded file to temp ────────────────────────────────────────────────

suffix = os.path.splitext(uploaded_file.name)[-1]
with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
    tmp.write(uploaded_file.read())
    tmp_path = tmp.name

# ── Audio info ────────────────────────────────────────────────────────────────

st.audio(uploaded_file)

try:
    y, sr = load_audio(tmp_path)
    duration_sec = len(y) / sr
    n_segments   = int(duration_sec // segment_duration)

    col1, col2, col3 = st.columns(3)
    col1.metric("Duration", f"{int(duration_sec // 60)}m {int(duration_sec % 60)}s")
    col2.metric("Sample Rate", f"{sr} Hz")
    col3.metric("Segments to Analyze", str(n_segments))

except Exception as e:
    st.error(f"Could not load audio: {e}")
    os.unlink(tmp_path)
    st.stop()

if n_segments == 0:
    st.warning("Audio too short. Please upload at least 30 seconds.")
    os.unlink(tmp_path)
    st.stop()

# ── Run Analysis ──────────────────────────────────────────────────────────────

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

if st.button("🔍 Analyze Interview", type="primary", use_container_width=True):

    segments = segment_audio(y, sr, segment_sec=segment_duration)

    progress_bar  = st.progress(0)
    status_text   = st.empty()

    def update_progress(i, total):
        progress_bar.progress(i / total)
        status_text.text(f"Analyzing segment {i} of {total}...")

    with st.spinner("Running acoustic analysis..."):
        df = analyze_segments(segments, progress_callback=update_progress)

    progress_bar.empty()
    status_text.empty()

    if df.empty:
        st.error("No segments could be analyzed. Check audio quality.")
        os.unlink(tmp_path)
        st.stop()

    # ── Summary Score Cards ───────────────────────────────────────────────────

    st.markdown("## 📊 Summary Scores")

    avg_stress     = df["stress_index"].mean()
    avg_confidence = df["confidence_score"].mean()
    avg_stability  = df["vocal_stability"].mean()
    avg_engagement = df["engagement_level"].mean()

    c1, c2, c3, c4 = st.columns(4)

    def score_card(col, label, value, color_emoji):
        col.markdown(
            f"""
            <div class="score-card">
                <div class="score-label">{color_emoji} {label}</div>
                <div class="score-value">{value:.1f}</div>
                <div class="score-unit">/ 10</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    score_card(c1, "Stress Index",     avg_stress,     "🔴")
    score_card(c2, "Confidence Score", avg_confidence, "🔵")
    score_card(c3, "Vocal Stability",  avg_stability,  "🟢")
    score_card(c4, "Engagement Level", avg_engagement, "🟠")

    # ── Acoustic Summary ──────────────────────────────────────────────────────

    st.markdown("#### Acoustic Measurements")
    a1, a2, a3 = st.columns(3)
    a1.metric("Avg Pitch (F0)",  f"{df['pitch_mean_hz'].mean():.1f} Hz")
    a2.metric("Avg Jitter",      f"{df['jitter'].mean():.4f}")
    a3.metric("Dominant Emotion",df["detected_emotion"].value_counts().index[0].upper())

    # ── Charts ────────────────────────────────────────────────────────────────

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("## 📈 Emotional Timeline")

    tab1, tab2, tab3 = st.tabs(["Signal Scores", "Raw Acoustics", "Emotion Distribution"])

    with tab1:
        fig1 = plot_signal_timeline(df)
        st.pyplot(fig1)

    with tab2:
        fig2 = plot_acoustic_timeline(df)
        st.pyplot(fig2)
        st.caption(
            "Pitch and jitter are involuntary acoustic signals. "
            "They reveal stress that the speaker cannot consciously suppress."
        )

    with tab3:
        fig3 = plot_emotion_distribution(df)
        st.pyplot(fig3)

    # ── Segment Table ─────────────────────────────────────────────────────────

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("## 🔎 Segment-Level Detail")

    display_cols = [
        "timestamp", "detected_emotion", "model_confidence",
        "stress_index", "confidence_score", "vocal_stability",
        "engagement_level", "pitch_mean_hz", "jitter"
    ]
    st.dataframe(
        df[display_cols].rename(columns={
            "timestamp":        "Time",
            "detected_emotion": "Emotion",
            "model_confidence": "Model Conf %",
            "stress_index":     "Stress",
            "confidence_score": "Confidence",
            "vocal_stability":  "Stability",
            "engagement_level": "Engagement",
            "pitch_mean_hz":    "Pitch (Hz)",
            "jitter":           "Jitter",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # ── Coaching Report ───────────────────────────────────────────────────────

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("## 📝 Coaching Report")

    report_text = generate_report(df)
    st.text(report_text)

    # ── Downloads ─────────────────────────────────────────────────────────────

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("## ⬇️ Download Results")

    d1, d2 = st.columns(2)

    with d1:
        st.download_button(
            label="📄 Download Coaching Report (.txt)",
            data=report_text,
            file_name="coaching_report.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with d2:
        csv_data = df[display_cols].to_csv(index=False)
        st.download_button(
            label="📊 Download Segment Data (.csv)",
            data=csv_data,
            file_name="segment_analysis.csv",
            mime="text/csv",
            use_container_width=True,
        )

# ── Cleanup ───────────────────────────────────────────────────────────────────

try:
    os.unlink(tmp_path)
except Exception:
    pass