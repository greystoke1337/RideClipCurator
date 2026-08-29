"""Audio analysis: speech presence (YAMNet) and transcript (Whisper) (spec §5.6).

YAMNet runs on CPU on purpose — it's a small model and TensorFlow dropped
native-Windows GPU support, so fighting that isn't worth it here. Whisper
uses the GPU via torch, same as RAM in tagging.py.
"""

import subprocess
import wave
from pathlib import Path

import numpy as np

from ridecurator.config import AUDIO_SILENCE_RMS_DBFS, YAMNET_SPEECH_CLASSES, YAMNET_SPEECH_THRESHOLD

_yamnet_model = None
_yamnet_class_names = None
_whisper_model = None


def extract_audio(video_path: str, wav_dir: str) -> str:
    """Pull mono 16kHz audio out of the proxy for YAMNet/Whisper."""
    video_path = Path(video_path)
    wav_dir = Path(wav_dir)
    wav_dir.mkdir(parents=True, exist_ok=True)
    wav_path = wav_dir / f"{video_path.stem}.wav"

    if not wav_path.exists():
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(video_path),
                "-vn", "-ac", "1", "-ar", "16000", str(wav_path),
            ],
            capture_output=True, text=True, check=True,
        )
    return str(wav_path)


def audio_level_dbfs(wav_path: str) -> float:
    """Quick RMS-based level check, used to skip Whisper on wind/engine-noise-only
    GoPro clips before spending GPU time on them (spec §5.6)."""
    with wave.open(wav_path, "rb") as wf:
        raw = wf.readframes(wf.getnframes())
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float64)

    if samples.size == 0:
        return -np.inf

    rms = np.sqrt(np.mean(samples ** 2))
    if rms == 0:
        return -np.inf
    return float(20 * np.log10(rms / 32768.0))


def has_decent_audio(wav_path: str, threshold_dbfs: float = AUDIO_SILENCE_RMS_DBFS) -> bool:
    return audio_level_dbfs(wav_path) >= threshold_dbfs


def _load_yamnet():
    global _yamnet_model, _yamnet_class_names
    if _yamnet_model is None:
        import csv

        import tensorflow_hub as hub

        _yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
        class_map_path = _yamnet_model.class_map_path().numpy().decode("utf-8")
        with open(class_map_path) as f:
            _yamnet_class_names = [row["display_name"] for row in csv.DictReader(f)]
    return _yamnet_model, _yamnet_class_names


def detect_speech(wav_path: str) -> bool:
    """True if YAMNet flags speech/conversation/narration above threshold anywhere in the clip."""
    import soundfile as sf

    model, class_names = _load_yamnet()
    waveform, sample_rate = sf.read(wav_path, dtype="float32")
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if sample_rate != 16000:
        raise ValueError(f"expected 16kHz audio, got {sample_rate} — check extract_audio()")

    scores, _, _ = model(waveform)
    scores = scores.numpy()
    max_scores = scores.max(axis=0)

    for class_name in YAMNET_SPEECH_CLASSES:
        idx = class_names.index(class_name)
        if max_scores[idx] >= YAMNET_SPEECH_THRESHOLD:
            return True
    return False


def transcribe(wav_path: str, model_size: str = "base") -> str:
    """Full transcript via Whisper. Only call this on clips that passed
    has_decent_audio() — running it on wind/engine noise wastes GPU time
    and produces garbage text."""
    global _whisper_model
    import whisper

    if _whisper_model is None:
        _whisper_model = whisper.load_model(model_size)

    result = _whisper_model.transcribe(wav_path)
    return result["text"].strip()
