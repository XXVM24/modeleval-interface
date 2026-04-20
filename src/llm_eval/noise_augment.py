"""
Adds calibrated white noise at specified SNR levels for noise robustness testing.
Output: 16-bit WAV files keyed as {"clean": path, "snr_20": path, ...}
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, List

import numpy as np


def add_white_noise(audio_path: str, snr_db: float, out_path: str) -> str:
    """Add white noise to audio_path at the given SNR (dB) and write to out_path."""
    try:
        import librosa
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError(
            "librosa and soundfile are required for noise augmentation. "
            "Install them with: pip install librosa soundfile"
        ) from exc

    y, sr = librosa.load(audio_path, sr=None, mono=True)

    signal_rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
    if signal_rms < 1e-10:
        # Silent file — write as-is
        sf.write(out_path, y, sr, subtype="PCM_16")
        return out_path

    # Compute noise amplitude to achieve the target SNR
    noise_rms_target = signal_rms / (10 ** (snr_db / 20.0))
    noise = np.random.randn(len(y)).astype(np.float32)
    noise_rms_actual = float(np.sqrt(np.mean(noise.astype(np.float64) ** 2)))
    if noise_rms_actual > 1e-10:
        noise = noise * (noise_rms_target / noise_rms_actual)

    noisy = y + noise
    # Clip to [-1, 1] to prevent clipping distortion after float→int16 conversion
    noisy = np.clip(noisy, -1.0, 1.0)

    os.makedirs(Path(out_path).parent, exist_ok=True)
    sf.write(out_path, noisy, sr, subtype="PCM_16")
    return out_path


def generate_noise_variants(
    audio_path: str,
    snr_levels: List[float],
    out_dir: str,
) -> Dict[str, str]:
    """Generate {"clean": path, "snr_20": path, ...} for one audio file at given SNR levels."""
    os.makedirs(out_dir, exist_ok=True)
    stem = Path(audio_path).stem

    variants: Dict[str, str] = {"clean": audio_path}
    for snr in snr_levels:
        label = f"snr_{int(snr)}"
        out_file = str(Path(out_dir) / f"{stem}_{label}.wav")
        add_white_noise(audio_path, snr_db=snr, out_path=out_file)
        variants[label] = out_file

    return variants


def generate_noise_variants_batch(
    audio_paths: List[str],
    snr_levels: List[float],
    out_dir: str,
    on_progress=None,
) -> Dict[str, Dict[str, str]]:
    """Run generate_noise_variants over a list of files. Returns {original_path: variants_dict}."""
    total = len(audio_paths)
    results: Dict[str, Dict[str, str]] = {}
    for i, path in enumerate(audio_paths):
        if on_progress:
            on_progress(i / max(total, 1), f"Generating noise variants for {Path(path).name}…")
        results[path] = generate_noise_variants(path, snr_levels, out_dir)
    if on_progress:
        on_progress(1.0, "Noise variants ready.")
    return results
