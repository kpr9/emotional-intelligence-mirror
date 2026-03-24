import librosa
import noisereduce as nr
import numpy as np
import soundfile as sf
import subprocess
import tempfile
import os


def _convert_to_wav(file_path: str) -> str:
    """
    Convert any audio format to a temporary WAV file using ffmpeg.
    Falls back gracefully if ffmpeg is not installed.
    Returns path to wav file (may be same as input if already wav).
    """
    ext = os.path.splitext(file_path)[-1].lower()
    if ext in [".wav"]:
        return file_path, False  # no temp file created

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.close()

    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", file_path, "-ar", "16000", "-ac", "1", tmp.name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode == 0:
            return tmp.name, True
    except FileNotFoundError:
        pass  # ffmpeg not installed

    os.unlink(tmp.name)
    return file_path, False  # fall back to librosa direct load


def load_audio(file_path: str, sr: int = 16000) -> tuple[np.ndarray, int]:
    """
    Load any audio format at target sample rate.
    Tries ffmpeg conversion first (handles M4A, AAC, etc on Windows).
    Falls back to librosa direct load.
    """
    converted_path, was_converted = _convert_to_wav(file_path)

    try:
        y, loaded_sr = librosa.load(converted_path, sr=sr, mono=True)
    finally:
        if was_converted and os.path.exists(converted_path):
            os.unlink(converted_path)

    return y, loaded_sr


def clean_audio(y: np.ndarray, sr: int) -> np.ndarray:
    """Noise reduction + silence trimming."""
    y = nr.reduce_noise(y=y, sr=sr)
    y, _ = librosa.effects.trim(y, top_db=20)
    return y.astype(np.float32)


def segment_audio(y: np.ndarray, sr: int, segment_sec: int = 10) -> list[dict]:
    """
    Split audio into fixed-length segments.
    Returns list of dicts with audio array + timestamp metadata.
    """
    segment_samples = segment_sec * sr
    n_segments = int(np.floor(len(y) / segment_samples))

    segments = []
    for i in range(n_segments):
        start = i * segment_samples
        end = start + segment_samples
        chunk = y[start:end]

        minutes = (i * segment_sec) // 60
        seconds = (i * segment_sec) % 60

        segments.append({
            "index":     i + 1,
            "timestamp": f"{minutes:02d}:{seconds:02d}",
            "start_sec": i * segment_sec,
            "audio":     chunk,
        })

    return segments