# 🎙️ Interview Emotional Intelligence Mirror

> AI-powered vocal self-awareness tool for interview preparation.  
> Built for candidates — not recruiters.

---

## What It Does

Upload a recorded mock interview. The system:

1. Splits audio into 10-second segments
2. Classifies emotion per segment using a pretrained wav2vec2 model (IEMOCAP-trained)
3. Extracts acoustic features — pitch, jitter, energy, ZCR — from the raw waveform
4. Blends both (50% model + 50% acoustics) into four interview-relevant signal scores
5. Generates a coaching report with timestamped insights

**Output signals:**
- 🔴 Stress Index
- 🔵 Confidence Score  
- 🟢 Vocal Stability
- 🟠 Engagement Level

---

## Ethical Design

This tool is explicitly **not** designed for recruiters to evaluate candidates.  
Emotion-based hiring decisions are ethically risky and legally problematic.

The user is always the candidate. The feedback is a mirror — not a judgment.  
Audio is processed locally. No data is stored.

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Emotion Model | `superb/wav2vec2-base-superb-er` (HuggingFace) |
| Acoustic Features | librosa (pitch, jitter, energy, ZCR) |
| Audio Processing | librosa + noisereduce |
| Visualization | matplotlib |

---

## Setup & Run

```bash
# 1. Clone
git clone https://github.com/yourusername/Interview_EI_Mirror.git
cd Interview_EI_Mirror

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Project Structure

```
Interview_EI_Mirror/
├── app.py              # Streamlit UI — entry point
├── core/
│   ├── analyzer.py     # Orchestrates segment analysis
│   ├── features.py     # Acoustic feature extraction
│   ├── scorer.py       # Blended signal scoring logic
│   └── report.py       # Coaching report generator
├── utils/
│   └── audio.py        # Load, clean, segment audio
├── visuals/
│   └── charts.py       # Timeline + distribution charts
├── assets/
│   └── style.css       # Custom styling
└── requirements.txt
```

---

## Model Selection Rationale

| Model | Macro F1 (RAVDESS, 4-class) | Notes |
|---|---|---|
| SVM + MFCC (baseline) | ~0.45 | Classical ML, fast, interpretable |
| wav2vec2-IEMOCAP | 0.25 | Lower on acted speech, better on real conversational |

IEMOCAP is trained on dyadic conversational speech — two people talking — which is closer to real interview context than theatrical datasets like RAVDESS. The F1 gap on RAVDESS is expected and documented as a known limitation.

---

## Known Limitations

- Model trained on English speech — performance may vary for non-native speakers
- Acted emotion datasets (RAVDESS) do not fully represent natural interview speech
- Jitter approximation is computed via librosa, not clinical-grade Praat analysis
- Real-time mode not yet implemented (planned)
- Speaker diarization not yet integrated — best used with single-speaker recordings

---

## Future Work

- Speaker diarization (pyannote.audio) to handle two-speaker recordings
- Real-time analysis mode during live mock interviews
- Per-question segmentation based on silence detection
- Fine-tuning on interview-specific labeled data
