"""
adapters/xai_adapter.py — Adapter for xAI (Grok).

xAI is Elon Musk's AI company. Grok is their model — a competitor to
GPT-4 and Claude. Has a free API tier.

Note: Not to be confused with Groq (the chip company). xAI = the model.
Groq = the hardware. They sound the same but are completely different.

Required env var:
    XAI_API_KEY  — obtain from https://x.ai/api

Optional env var:
    XAI_MODEL    — defaults to "grok-3-mini" (free tier)
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


class XAIAdapter(BaseAdapter):
    """xAI Grok — strong general-purpose model with a free API tier."""

    PROVIDER_KEY = "xai"
    TIER = "free"
    COST_PER_1K_TOKENS = None

    SUPPORTED_TASK_TYPES: List[TaskType] = [
        TaskType.REASONING,
        TaskType.FACTUAL_QA,
        TaskType.CODING,
        TaskType.GENERAL,
        TaskType.SUMMARIZATION,
        TaskType.CREATIVE,
    ]

    DEFAULT_MODEL = "grok-3-mini"
    BASE_URL = "https://api.x.ai/v1"

    def __init__(self) -> None:
        self._api_key = os.getenv("XAI_API_KEY", "")
        self._model   = os.getenv("XAI_MODEL", self.DEFAULT_MODEL)
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
                error="xAI adapter not available — check XAI_API_KEY"
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
                cost_usd=None,
                finish_reason=choice.finish_reason,
            )

        except Exception as exc:
            return self._make_result(
                task, self.PROVIDER_KEY, self._model, "",
                latency_ms=self._elapsed_ms(start),
                error=str(exc),
            )
