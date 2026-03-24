import librosa
import numpy as np


def extract_acoustic_features(y: np.ndarray, sr: int = 16000) -> dict:
    """
    Extract 6 acoustic features from raw audio segment.
    These measure HOW the emotion is expressed — not just what category.

    pitch_mean  — avg fundamental frequency. Stress raises pitch involuntarily.
    pitch_std   — pitch variability. Nervous voices wander.
    jitter      — cycle-to-cycle pitch variation. Impossible to fake under stress.
    energy_mean — average loudness. Low = disengaged.
    energy_std  — volume consistency. High = erratic delivery.
    zcr_mean    — zero crossing rate. Higher in tense, active speech.
    """
    features = {
        "pitch_mean":  0.0,
        "pitch_std":   0.0,
        "jitter":      0.0,
        "energy_mean": 0.0,
        "energy_std":  0.0,
        "zcr_mean":    0.0,
    }

    try:
        # Pitch
        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
        )
        f0_voiced = f0[voiced_flag] if voiced_flag is not None else np.array([])

        if len(f0_voiced) > 1:
            features["pitch_mean"] = float(np.mean(f0_voiced))
            features["pitch_std"]  = float(np.std(f0_voiced))
            features["jitter"]     = float(
                np.mean(np.abs(np.diff(f0_voiced))) / (np.mean(f0_voiced) + 1e-6)
            )

        # Energy
        rms = librosa.feature.rms(y=y)[0]
        features["energy_mean"] = float(np.mean(rms))
        features["energy_std"]  = float(np.std(rms))

        # ZCR
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features["zcr_mean"] = float(np.mean(zcr))

    except Exception:
        pass  # Return zero features on failure — segment still processed

    return features