import numpy as np
import pandas as pd
from transformers import pipeline as hf_pipeline

from core.features import extract_acoustic_features
from core.scorer import compute_blended_scores
from utils.audio import clean_audio

_emotion_pipeline = None


def load_model():
    """Load pretrained model once and cache it."""
    global _emotion_pipeline
    if _emotion_pipeline is None:
        _emotion_pipeline = hf_pipeline(
            task="audio-classification",
            model="superb/wav2vec2-base-superb-er",
            device=-1,  # CPU — works without GPU
        )
    return _emotion_pipeline


def analyze_segment(audio_array: np.ndarray, sr: int = 16000) -> dict | None:
    """
    Full per-segment pipeline:
    1. Clean audio
    2. Emotion classification (pretrained wav2vec2)
    3. Acoustic feature extraction (pitch, jitter, energy, ZCR)
    4. Blend both into final interview signal scores
    """
    model = load_model()

    audio_clean = clean_audio(audio_array, sr)

    # Skip segments shorter than 1 second
    if len(audio_clean) < sr * 1.0:
        return None

    # Step 1: Emotion classification
    result    = model({"array": audio_clean, "sampling_rate": sr}, top_k=1)
    raw_label = result[0]["label"]
    model_conf= result[0]["score"]

    # Step 2: Acoustic features
    acoustic = extract_acoustic_features(audio_clean, sr)

    # Step 3: Blended scores
    return compute_blended_scores(raw_label, model_conf, acoustic)


def analyze_segments(segments: list[dict], progress_callback=None) -> pd.DataFrame:
    """
    Run analysis across all audio segments.
    progress_callback(i, total) — optional, used by Streamlit progress bar.
    """
    results = []
    total   = len(segments)

    for i, seg in enumerate(segments):
        scores = analyze_segment(seg["audio"])

        if scores:
            scores["segment"]   = seg["index"]
            scores["timestamp"] = seg["timestamp"]
            scores["start_sec"] = seg["start_sec"]
            results.append(scores)

        if progress_callback:
            progress_callback(i + 1, total)

    return pd.DataFrame(results)