"""Adapter for Google Gemini via the google-genai SDK (v1+), synchronous."""

import time
from pathlib import Path

from google import genai
from google.genai import types

from .base import BaseModel, ModelConfig, ModelResponse

_AUDIO_MIME_MAP = {
    ".wav":  "audio/wav",
    ".wave": "audio/wav",
    ".mp3":  "audio/mp3",
    ".m4a":  "audio/mp4",
    ".ogg":  "audio/ogg",
    ".flac": "audio/flac",
    ".webm": "audio/webm",
}


class GeminiModel(BaseModel):

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._client = genai.Client(api_key=config.api_key)
        self._gen_config = types.GenerateContentConfig(
            max_output_tokens=config.max_tokens,
            temperature=config.temperature,
        )
        self._model_id = config.model_id

    def supports_audio(self) -> bool:
        return True

    def generate(self, prompt: str) -> ModelResponse:
        start = time.perf_counter()
        try:
            response = self._client.models.generate_content(
                model=self._model_id,
                contents=prompt,
                config=self._gen_config,
            )
            # Extract text and finish_reason from first candidate
            text, finish_reason, ratings_str = "", "", ""
            try:
                candidate = response.candidates[0]
                finish_reason = candidate.finish_reason.name
                ratings_str = ", ".join(
                    f"{r.category.name}={r.probability.name}"
                    for r in (candidate.safety_ratings or [])
                )
                # Collect partial text from parts if response.text is None
                text = response.text or "".join(
                    p.text for p in candidate.content.parts if hasattr(p, "text") and p.text
                )
            except Exception:
                text = response.text or ""

            error = None
            if finish_reason == "MAX_TOKENS":
                error = f"Truncated at max_tokens (finish_reason=MAX_TOKENS) — increase max_tokens in config"
            elif not text:
                detail = finish_reason or "unknown"
                if ratings_str:
                    detail += f" [{ratings_str}]"
                error = f"No text returned (finish_reason={detail})"

            return ModelResponse(
                model_name=self.name,
                question=prompt,
                prediction=text.strip() if text else "",
                latency_seconds=time.perf_counter() - start,
                error=error,
            )
        except Exception as e:
            return ModelResponse(
                model_name=self.name,
                question=prompt,
                prediction="",
                latency_seconds=time.perf_counter() - start,
                error=str(e),
            )

    def generate_audio(self, audio_path: str) -> ModelResponse:
        start = time.perf_counter()
        try:
            suffix = Path(audio_path).suffix.lower()
            mime = _AUDIO_MIME_MAP.get(suffix, "audio/wav")
            audio_bytes = Path(audio_path).read_bytes()

            response = self._client.models.generate_content(
                model=self._model_id,
                contents=[types.Part.from_bytes(data=audio_bytes, mime_type=mime)],
                config=self._gen_config,
            )
            text = response.text
            if text is None:
                reason = ""
                try:
                    reason = response.candidates[0].finish_reason.name
                except Exception:
                    pass
                return ModelResponse(
                    model_name=self.name,
                    question=Path(audio_path).name,
                    prediction="",
                    latency_seconds=time.perf_counter() - start,
                    error=f"No text returned (finish_reason={reason or 'unknown'})",
                )
            return ModelResponse(
                model_name=self.name,
                question=Path(audio_path).name,
                prediction=text.strip(),
                latency_seconds=time.perf_counter() - start,
            )
        except Exception as e:
            return ModelResponse(
                model_name=self.name,
                question=Path(audio_path).name,
                prediction="",
                latency_seconds=time.perf_counter() - start,
                error=str(e),
            )
