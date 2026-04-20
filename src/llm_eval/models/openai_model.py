"""Adapter for ChatGPT via the OpenAI API (synchronous)."""

import base64
import time
from pathlib import Path

from openai import OpenAI

from .base import BaseModel, ModelConfig, ModelResponse

# Formats accepted by the OpenAI audio input API
_AUDIO_FORMAT_MAP = {
    ".wav":  "wav",
    ".wave": "wav",
    ".mp3":  "mp3",
    ".m4a":  "m4a",
    ".ogg":  "ogg",
    ".flac": "flac",
    ".webm": "webm",
}


class OpenAIModel(BaseModel):
    """Calls api.openai.com synchronously."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._client = OpenAI(
            api_key=config.api_key,
            timeout=config.timeout,
        )

    def supports_audio(self) -> bool:
        return "audio" in self.config.model_id.lower()

    def generate(self, prompt: str) -> ModelResponse:
        start = time.perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=self.config.model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
            prediction = response.choices[0].message.content.strip()
            return ModelResponse(
                model_name=self.name,
                question=prompt,
                prediction=prediction,
                latency_seconds=time.perf_counter() - start,
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
            fmt = _AUDIO_FORMAT_MAP.get(suffix, "wav")
            with open(audio_path, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode("utf-8")

            response = self._client.chat.completions.create(
                model=self.config.model_id,
                messages=[{
                    "role": "user",
                    "content": [{
                        "type": "input_audio",
                        "input_audio": {"data": audio_b64, "format": fmt},
                    }],
                }],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
            prediction = response.choices[0].message.content.strip()
            return ModelResponse(
                model_name=self.name,
                question=Path(audio_path).name,
                prediction=prediction,
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
