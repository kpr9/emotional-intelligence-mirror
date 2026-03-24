import pandas as pd


def generate_report(df: pd.DataFrame) -> str:
    """
    Generate human-readable coaching report from analysis DataFrame.
    References actual acoustic values — not just template text.
    """
    avg_stress     = df["stress_index"].mean()
    avg_confidence = df["confidence_score"].mean()
    avg_stability  = df["vocal_stability"].mean()
    avg_engagement = df["engagement_level"].mean()

    peak_stress_time = df.loc[df["stress_index"].idxmax(), "timestamp"]
    best_conf_time   = df.loc[df["confidence_score"].idxmax(), "timestamp"]
    avg_pitch        = df["pitch_mean_hz"].mean()
    avg_jitter       = df["jitter"].mean()
    dominant_emotion = df["detected_emotion"].value_counts().index[0]
    variety          = df["detected_emotion"].nunique()

    if avg_jitter < 0.01:
        jitter_text = "normal — vocal cords stable under pressure"
    elif avg_jitter < 0.03:
        jitter_text = "mildly elevated — some tension present"
    else:
        jitter_text = "high — significant involuntary vocal stress detected"

    lines = [
        "=" * 58,
        "   INTERVIEW EMOTIONAL INTELLIGENCE REPORT",
        "   Acoustic-Blended Analysis",
        "=" * 58,
        f"\n  Dominant Emotion      : {dominant_emotion.upper()}",
        f"  Emotional Variety     : {variety} distinct states across session",
        f"\n  Stress Index          : {avg_stress:.1f} / 10",
        f"  Confidence Score      : {avg_confidence:.1f} / 10",
        f"  Vocal Stability       : {avg_stability:.1f} / 10",
        f"  Engagement Level      : {avg_engagement:.1f} / 10",
        f"\n  Avg Pitch (F0)        : {avg_pitch:.1f} Hz",
        f"  Avg Jitter            : {avg_jitter:.4f}  ({jitter_text})",
        "\n" + "-" * 58,
        "  COACHING INSIGHTS",
        "-" * 58,
    ]

    # Stress
    if avg_stress >= 7:
        lines += [
            f"\n  [STRESS]  High stress detected. Peak at {peak_stress_time}.",
            f"  Your pitch averaged {avg_pitch:.0f}Hz — elevated range signals arousal.",
            "  Tip: Practice slow breathing. Pause 2s before answering.",
        ]
    elif avg_stress >= 4:
        lines += [
            f"\n  [STRESS]  Moderate stress. Peaked at {peak_stress_time}.",
            "  Tip: Pause briefly before answering under pressure.",
        ]
    else:
        lines.append("\n  [STRESS]  Stress well controlled throughout. Good composure.")

    # Confidence
    if avg_confidence >= 7:
        lines += [f"\n  [CONFIDENCE]  Strong confidence. Best moment at {best_conf_time}."]
    elif avg_confidence >= 4:
        lines += [
            f"\n  [CONFIDENCE]  Moderate confidence. Best moment at {best_conf_time}.",
            "  Tip: Speak in shorter, clearer sentences with deliberate pauses.",
        ]
    else:
        lines += [
            "\n  [CONFIDENCE]  Low confidence signals detected.",
            "  Tip: Slow your speaking rate — confidence sounds unhurried.",
        ]

    # Stability
    if avg_stability >= 7:
        lines.append("\n  [STABILITY]  Voice was steady and consistent throughout.")
    elif avg_stability >= 4:
        lines += [
            "\n  [STABILITY]  Some vocal instability detected mid-session.",
            "  Tip: Avoid rushing — your voice steadies when you slow down.",
        ]
    else:
        lines += [
            "\n  [STABILITY]  Significant instability throughout.",
            "  Tip: Record yourself and listen back regularly to build awareness.",
        ]

    # Jitter
    lines += [
        f"\n  [JITTER]  Involuntary stress marker: {jitter_text}.",
    ]
    if avg_jitter >= 0.03:
        lines += [
            "  Jitter is difficult to consciously control.",
            "  Tip: Regular mock interview practice reduces baseline anxiety.",
        ]
    elif avg_jitter >= 0.01:
        lines.append("  Mild jitter is normal in interview conditions.")
    else:
        lines.append("  Low jitter — strong vocal composure detected.")

    # Engagement
    if avg_engagement >= 7:
        lines.append("\n  [ENGAGEMENT]  High energy and engagement maintained.")
    elif avg_engagement >= 4:
        lines += [
            "\n  [ENGAGEMENT]  Engagement varied across segments.",
            "  Tip: Treat every question with equal energy and interest.",
        ]
    else:
        lines += [
            "\n  [ENGAGEMENT]  Low engagement signals detected.",
            "  Tip: Show enthusiasm through vocal variety and pace changes.",
        ]

    lines += [
        "\n" + "=" * 58,
        "  METHODOLOGY",
        "  Scores = 50% emotion model (wav2vec2, IEMOCAP-trained)",
        "         + 50% acoustic features (pitch, jitter, energy, ZCR)",
        "  Directional feedback for self-improvement only.",
        "  Not a clinical assessment. Not used in hiring decisions.",
        "  The user owns their data.",
        "=" * 58,
    ]

    return "\n".join(lines)