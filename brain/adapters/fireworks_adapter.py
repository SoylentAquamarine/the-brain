"""
adapters/fireworks_adapter.py — Adapter for Fireworks AI.

Fireworks AI offers fast inference on open-source models with a free tier.
Good for coding and general tasks as an additional free fallback.

Required env var:
    FIREWORKS_API_KEY  — obtain from https://fireworks.ai/

Optional env var:
    FIREWORKS_MODEL    — defaults to "accounts/fireworks/models/llama-v3p1-70b-instruct"
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


class FireworksAdapter(BaseAdapter):
    """Fireworks AI — fast open-source model inference, free tier."""

    PROVIDER_KEY = "fireworks"
    TIER = "free"
    COST_PER_1K_TOKENS = None

    SUPPORTED_TASK_TYPES: List[TaskType] = [
        TaskType.CODING,
        TaskType.FACTUAL_QA,
        TaskType.GENERAL,
        TaskType.EXTRACTION,
        TaskType.SUMMARIZATION,
        TaskType.CLASSIFICATION,
    ]

    DEFAULT_MODEL = "accounts/fireworks/models/llama-v3p1-70b-instruct"
    BASE_URL = "https://api.fireworks.ai/inference/v1"

    def __init__(self) -> None:
        self._api_key = os.getenv("FIREWORKS_API_KEY", "")
        self._model   = os.getenv("FIREWORKS_MODEL", self.DEFAULT_MODEL)
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
                error="Fireworks adapter not available — check FIREWORKS_API_KEY"
            )

        start = self._start_timer()
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                max_tokens=task.max_tokens,
                messages=[
                    {"role": "system", "content": "Reply concisely and directly. No preamble or explanations unless the task requires them."},
                    {"role": "user", "content": task.full_prompt()},
                ],
            )
            choice  = response.choices[0]
            content = choice.message.content or ""
            tokens  = response.usage.total_tokens if response.usage else 0

            return self._make_result(
                task, self.PROVIDER_KEY, response.model,
                content=content,
                tokens_used=tokens,
                latency_ms=self._elapsed_ms(start),
                cost_usd=None,
                finish_reason=choice.finish_reason,
            )

        except Exception as exc:
            return self._make_result(
                task, self.PROVIDER_KEY, self._model, "",
                latency_ms=self._elapsed_ms(start),
                error=str(exc),
            )
