"""Adapter for Fireworks AI (OpenAI-compatible API), synchronous."""

import time

from openai import OpenAI

from .base import BaseModel, ModelConfig, ModelResponse

FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"


class FireworksModel(BaseModel):
    """Calls Fireworks AI via the OpenAI-compatible endpoint."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url or FIREWORKS_BASE_URL,
            timeout=config.timeout,
        )

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
