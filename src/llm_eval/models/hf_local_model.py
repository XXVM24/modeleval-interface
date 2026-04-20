"""
Direct Hugging Face local model inference.

Loads a model from a local snapshot directory (e.g. downloaded via
huggingface-cli or modelscope). No vLLM or HTTP server required.

Memory management:
- Call load() before running a batch of questions.
- Call unload() after to free GPU memory before loading the next model.
- The EvalRunner handles this automatically.
"""

import time
import torch
from typing import Optional

from .base import BaseModel, ModelConfig, ModelResponse


class HFLocalModel(BaseModel):
    """
    Loads and runs a HuggingFace causal-LM model directly in Python.

    Supports any AutoModelForCausalLM-compatible checkpoint: LLaMA, Mistral,
    Qwen, ChatGLM, Baichuan, DeepSeek, etc.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._model = None
        self._tokenizer = None

    # ------------------------------------------------------------------
    # Load / unload (called by EvalRunner around each model's question loop)
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load model weights and tokenizer into GPU/CPU memory."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path = self.config.model_path
        print(f"  Loading {self.name} from {model_path} ...")

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        # Ensure a pad token exists (many causal-LM models don't set one)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        device_map = "auto" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        self._model.eval()
        print(f"  {self.name} loaded.")

    def unload(self) -> None:
        """Delete model from memory and clear the GPU cache."""
        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  {self.name} unloaded.")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> ModelResponse:
        """Run inference synchronously. Must call load() first."""
        if self._model is None:
            return ModelResponse(
                model_name=self.name,
                question=prompt,
                prediction="",
                latency_seconds=0.0,
                error="Model not loaded. Call load() before generate().",
            )

        start = time.perf_counter()
        try:
            input_ids = self._build_input(prompt)
            with torch.no_grad():
                output_ids = self._model.generate(
                    input_ids,
                    max_new_tokens=self.config.max_tokens,
                    temperature=max(self.config.temperature, 1e-7),
                    do_sample=self.config.temperature > 0,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                )
            # Decode only the newly generated tokens (skip the prompt)
            new_tokens = output_ids[0][input_ids.shape[-1]:]
            prediction = self._tokenizer.decode(
                new_tokens, skip_special_tokens=True
            ).strip()
            latency = time.perf_counter() - start
            return ModelResponse(
                model_name=self.name,
                question=prompt,
                prediction=prediction,
                latency_seconds=latency,
            )
        except Exception as e:
            latency = time.perf_counter() - start
            return ModelResponse(
                model_name=self.name,
                question=prompt,
                prediction="",
                latency_seconds=latency,
                error=str(e),
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_input(self, prompt: str) -> torch.Tensor:
        """
        Format the prompt using the model's chat template if available,
        otherwise fall back to a plain user-turn format.
        """
        messages = [{"role": "user", "content": prompt}]

        if hasattr(self._tokenizer, "apply_chat_template") and \
                self._tokenizer.chat_template is not None:
            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # Fallback for models without a chat template
            text = f"User: {prompt}\nAssistant:"

        device = next(self._model.parameters()).device
        inputs = self._tokenizer(text, return_tensors="pt").to(device)
        return inputs["input_ids"]
