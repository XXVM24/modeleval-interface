"""
Speech clarity metrics using VAD-based SNR estimation.
No reference audio needed — works on a single file.
clarity_score = clip(snr_db/30, 0, 1) * 0.5 + whisper_confidence * 0.5
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


def compute_audio_clarity(
    audio_path: str,
    whisper_confidence: Optional[float] = None,
    top_db: float = 30.0,
) -> dict:
    """Return {snr_db, speech_ratio, clarity_score} for an audio file. Returns zeros on any error."""
    try:
        import librosa
    except ImportError:
        return {"snr_db": 0.0, "speech_ratio": 0.0, "clarity_score": 0.0}

    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
    except Exception:
        return {"snr_db": 0.0, "speech_ratio": 0.0, "clarity_score": 0.0}

    if len(y) == 0:
        return {"snr_db": 0.0, "speech_ratio": 0.0, "clarity_score": 0.0}

    # Detect speech (non-silent) intervals
    try:
        intervals = librosa.effects.split(y, top_db=top_db)
    except Exception:
        intervals = np.array([]).reshape(0, 2)

    total_samples = len(y)
    speech_samples = sum(end - start for start, end in intervals)
    speech_ratio = speech_samples / total_samples if total_samples > 0 else 0.0

    snr_db = _estimate_snr(y, intervals)

    # Compose clarity_score
    snr_component = min(max(snr_db / 30.0, 0.0), 1.0)
    if whisper_confidence is not None:
        wc = min(max(float(whisper_confidence), 0.0), 1.0)
        clarity_score = snr_component * 0.5 + wc * 0.5
    else:
        clarity_score = snr_component

    return {
        "snr_db":       round(float(snr_db), 2),
        "speech_ratio": round(float(speech_ratio), 4),
        "clarity_score": round(float(clarity_score), 4),
    }


def _estimate_snr(y: np.ndarray, intervals: np.ndarray) -> float:
    """SNR from speech/silence RMS ratio. Falls back to -60 dBFS floor when no silence detected."""
    if len(intervals) == 0:
        # No speech detected at all
        return 0.0

    # Build speech mask
    mask = np.zeros(len(y), dtype=bool)
    for start, end in intervals:
        mask[start:end] = True

    speech = y[mask]
    silence = y[~mask]

    signal_rms = _rms(speech)

    if len(silence) < 128:
        # Virtually no silence — assume very clean audio
        # Use -60 dBFS as the noise floor estimate
        noise_rms = _rms(y) * 1e-3  # ~60 dB below full scale
    else:
        noise_rms = _rms(silence)

    if noise_rms < 1e-10 or signal_rms < 1e-10:
        return 0.0

    return float(20.0 * math.log10(signal_rms / noise_rms))


def _rms(x: np.ndarray) -> float:
    if len(x) == 0:
        return 0.0
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
