import numpy as np

# Base signal weights per emotion label (50% of final score)
EMOTION_BASE = {
    "angry":   {"stress": 0.8, "confidence": 0.5, "stability": 0.3, "engagement": 0.6},
    "happy":   {"stress": 0.1, "confidence": 0.8, "stability": 0.7, "engagement": 0.9},
    "neutral": {"stress": 0.2, "confidence": 0.6, "stability": 0.8, "engagement": 0.5},
    "sad":     {"stress": 0.6, "confidence": 0.2, "stability": 0.4, "engagement": 0.2},
}

# Normalize model output labels to 4-class
MODEL_LABEL_MAP = {
    "ang": "angry",  "angry":   "angry",
    "hap": "happy",  "happy":   "happy",
    "neu": "neutral","neutral": "neutral",
    "sad": "sad",
}


def _norm(value: float, min_val: float, max_val: float) -> float:
    return float(np.clip((value - min_val) / (max_val - min_val + 1e-6), 0.0, 1.0))


def compute_blended_scores(
    raw_label: str,
    model_confidence: float,
    acoustic: dict,
) -> dict:
    """
    Final signal score = 50% emotion label base + 50% acoustic features.

    STRESS:     pitch_mean ↑ + jitter ↑ + pitch_std ↑  → stress ↑
    CONFIDENCE: energy_mean ↑ + pitch_std ↓ + jitter ↓ → confidence ↑
    STABILITY:  pitch_std ↓ + energy_std ↓ + jitter ↓  → stability ↑
    ENGAGEMENT: energy_mean ↑ + zcr ↑ + pitch_mean ↑  → engagement ↑
    """
    label = MODEL_LABEL_MAP.get(raw_label.lower(), "neutral")
    base  = EMOTION_BASE.get(label, EMOTION_BASE["neutral"])

    # Normalize acoustic features to 0-1
    pitch_mod     = _norm(acoustic["pitch_mean"], 85, 300)
    pitch_std_mod = _norm(acoustic["pitch_std"],   0,  80)
    jitter_mod    = _norm(acoustic["jitter"],       0,   0.05)
    energy_mod    = _norm(acoustic["energy_mean"], 0,   0.08)
    energy_std_mod= _norm(acoustic["energy_std"],  0,   0.05)
    zcr_mod       = _norm(acoustic["zcr_mean"],    0,   0.2)

    # Acoustic components
    acoustic_stress = pitch_mod * 0.35 + jitter_mod * 0.40 + pitch_std_mod * 0.25

    acoustic_confidence = np.clip(
        energy_mod * 0.50 - pitch_std_mod * 0.25 - jitter_mod * 0.25 + 0.5, 0, 1
    )

    acoustic_stability = np.clip(
        1.0 - (pitch_std_mod * 0.40 + energy_std_mod * 0.35 + jitter_mod * 0.25), 0, 1
    )

    acoustic_engagement = np.clip(
        energy_mod * 0.45 + zcr_mod * 0.35 + pitch_mod * 0.20, 0, 1
    )

    def blend(base_val, acoustic_val):
        return round(float(np.clip((base_val * 0.5 + acoustic_val * 0.5) * 10, 0, 10)), 2)

    return {
        "detected_emotion":  label,
        "model_confidence":  round(model_confidence * 100, 1),
        "stress_index":      blend(base["stress"],     acoustic_stress),
        "confidence_score":  blend(base["confidence"], acoustic_confidence),
        "vocal_stability":   blend(base["stability"],  acoustic_stability),
        "engagement_level":  blend(base["engagement"], acoustic_engagement),
        # Raw acoustics — shown in dashboard detail table
        "pitch_mean_hz": round(acoustic["pitch_mean"], 1),
        "pitch_std_hz":  round(acoustic["pitch_std"],  1),
        "jitter":        round(acoustic["jitter"],     4),
        "energy_mean":   round(acoustic["energy_mean"],4),
        "zcr_mean":      round(acoustic["zcr_mean"],   4),
    }