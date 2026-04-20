"""
Whisper-based audio transcription via the OpenAI API.

Returns a callable that transcribes a local audio file path and also
provides a speech-confidence score derived from Whisper's no_speech_prob.

Return type changed to Tuple[str, float]:
    (transcription_text, whisper_confidence)

whisper_confidence = 1 - avg(no_speech_prob across segments)
Range: [0, 1], where 1 = certain speech, 0 = likely silence/noise.

Note on Chinese audio:
  Whisper transcribes Mandarin audio as Traditional Chinese by default.
  Use the T→S converter in the UI Tools tab to convert to Simplified Chinese.
"""

import os
from typing import Callable, Tuple


def make_whisper_transcriber(api_key: str = None) -> Callable[[str], Tuple[str, float]]:
    """
    Return a transcribe(audio_path) → (text, whisper_confidence) function
    backed by OpenAI Whisper API.

    api_key: if None, reads OPENAI_API_KEY from the environment.
    Raises ValueError if no key is available.
    """
    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise ValueError(
            "Whisper transcription requires OPENAI_API_KEY to be set. "
            "Set it in your environment before launching app.py."
        )

    from openai import OpenAI
    client = OpenAI(api_key=key)

    def transcribe(audio_path: str) -> Tuple[str, float]:
        with open(audio_path, "rb") as f:
            result = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="verbose_json",
            )

        text = result.text if hasattr(result, "text") else ""

        # Compute confidence from per-segment no_speech_prob
        confidence = 1.0
        try:
            segments = result.segments if hasattr(result, "segments") else []
            if segments:
                avg_no_speech = sum(
                    float(s.get("no_speech_prob", 0.0) if isinstance(s, dict)
                          else getattr(s, "no_speech_prob", 0.0))
                    for s in segments
                ) / len(segments)
                confidence = max(0.0, min(1.0, 1.0 - avg_no_speech))
        except Exception:
            confidence = 1.0

        return text, confidence

    return transcribe
