"""
adapters/openrouter_adapter.py — Adapter for OpenRouter.

OpenRouter is a single API key that gives access to 50+ models,
many of which are completely free (marked with :free suffix).
Excellent as a fallback layer — if one model is down, swap to another.

Required env var:
    OPENROUTER_API_KEY  — obtain from https://openrouter.ai/keys

Optional env var:
    OPENROUTER_MODEL    — defaults to "meta-llama/llama-3.1-8b-instruct:free"

Free models available (as of 2026):
    meta-llama/llama-3.1-8b-instruct:free
    google/gemma-2-9b-it:free
    mistralai/mistral-7b-instruct:free
    microsoft/phi-3-mini-128k-instruct:free
"""

from __future__ import annotations

import os
from typing import List

try:
    from openai import OpenAI
    _SDK_AVAILABLE = True
except ImportError:
    _SDK_AVAILABLE = False

from brain.adapters.base import BaseAdapter
from brain.task import Task, TaskResult, TaskType


class OpenRouterAdapter(BaseAdapter):
    """OpenRouter — one key, access to 50+ models, many free."""

    PROVIDER_KEY = "openrouter"
    TIER = "free"
    COST_PER_1K_TOKENS = None  # free models are free; paid models vary

    SUPPORTED_TASK_TYPES: List[TaskType] = list(TaskType)

    DESCRIPTION   = "OpenRouter — single key for 50+ models, many completely free"
    QUALITY_SCORE = 5
    SPEED_TIER    = "standard"

    DEFAULT_MODEL = "google/gemma-4-26b-a4b-it:free"
    MODELS = [
        "google/gemma-4-26b-a4b-it:free",
    ]
    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self) -> None:
        self._api_key = os.getenv("OPENROUTER_API_KEY", "")
        self._model   = os.getenv("OPENROUTER_MODEL", self.DEFAULT_MODEL)
        self._client  = None

        if _SDK_AVAILABLE and self._api_key:
            self._client = OpenAI(
                base_url=self.BASE_URL,
                api_key=self._api_key,
            )

    def is_available(self) -> bool:
        return bool(_SDK_AVAILABLE and self._api_key and self._client)

    def complete(self, task: Task) -> TaskResult:
        if not self.is_available():
            return self._make_result(
                task, self.PROVIDER_KEY, self._model, "",
                error="OpenRouter adapter not available — check OPENROUTER_API_KEY"
            )

        start = self._start_timer()
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                max_tokens=task.max_tokens,
                messages=[{"role": "user", "content": task.full_prompt()}],
            )
            choice  = response.choices[0]
            content = choice.message.content or ""
            tokens  = response.usage.total_tokens if response.usage else 0

            return self._make_result(
                task, self.PROVIDER_KEY, response.model,
                content=content,
                tokens_used=tokens,
                latency_ms=self._elapsed_ms(start),
                cost_usd=None,  # free model
                finish_reason=choice.finish_reason,
            )

        except Exception as exc:
            return self._make_result(
                task, self.PROVIDER_KEY, self._model, "",
                latency_ms=self._elapsed_ms(start),
                error=str(exc),
            )
