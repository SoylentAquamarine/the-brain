"""
adapters/huggingface_adapter.py — Adapter for Hugging Face Inference API.

Hugging Face hosts thousands of open-source models on free serverless
inference endpoints.  The free tier rate-limits by model but costs nothing.
Great for specialist tasks: translation, classification, summarisation.

Uses the `huggingface_hub` InferenceClient (OpenAI-compatible interface).

Required env var:
    HUGGINGFACE_API_KEY  — obtain from https://huggingface.co/settings/tokens

Optional env var:
    HUGGINGFACE_MODEL    — defaults to "mistralai/Mistral-7B-Instruct-v0.3"
"""

from __future__ import annotations

import os
from typing import List

try:
    from huggingface_hub import InferenceClient
    _SDK_AVAILABLE = True
except ImportError:
    _SDK_AVAILABLE = False

from brain.adapters.base import BaseAdapter
from brain.task import Task, TaskResult, TaskType


class HuggingFaceAdapter(BaseAdapter):
    """Wraps the `huggingface_hub` InferenceClient."""

    PROVIDER_KEY = "huggingface"
    TIER = "free"
    COST_PER_1K_TOKENS = None

    SUPPORTED_TASK_TYPES: List[TaskType] = [
        TaskType.GENERAL,
        TaskType.SUMMARIZATION,
        TaskType.TRANSLATION,
        TaskType.CLASSIFICATION,
        TaskType.FACTUAL_QA,
        TaskType.CREATIVE,
    ]

    # Llama 3 8B Instruct is the most reliable chat model on the free serverless tier.
    DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

    def __init__(self) -> None:
        self._api_key    = os.getenv("HUGGINGFACE_API_KEY", "")
        self._model_name = os.getenv("HUGGINGFACE_MODEL", self.DEFAULT_MODEL)
        self._client     = None

        if _SDK_AVAILABLE and self._api_key:
            self._client = InferenceClient(
                model=self._model_name,
                token=self._api_key,
            )

    def is_available(self) -> bool:
        return bool(_SDK_AVAILABLE and self._api_key and self._client)

    def complete(self, task: Task) -> TaskResult:
        if not self.is_available():
            return self._make_result(
                task, self.PROVIDER_KEY, self._model_name, "",
                error="HuggingFace adapter not available — check HUGGINGFACE_API_KEY"
            )

        start = self._start_timer()
        try:
            response = self._client.chat_completion(
                messages=[{"role": "user", "content": task.full_prompt()}],
                max_tokens=task.max_tokens,
            )
            content = response.choices[0].message.content or ""
            tokens  = response.usage.total_tokens if response.usage else 0

            return self._make_result(
                task, self.PROVIDER_KEY, self._model_name,
                content=content,
                tokens_used=tokens,
                latency_ms=self._elapsed_ms(start),
                cost_usd=None,
            )

        except Exception as exc:
            return self._make_result(
                task, self.PROVIDER_KEY, self._model_name, "",
                latency_ms=self._elapsed_ms(start),
                error=str(exc),
            )
