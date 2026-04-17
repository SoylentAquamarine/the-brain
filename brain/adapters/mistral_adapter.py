"""
adapters/mistral_adapter.py — Adapter for Mistral AI models.

Mistral offers a free tier on La Plateforme with capable open-weight models.
Mistral Small is a strong general-purpose model; Codestral is purpose-built
for coding tasks and rivals GPT-4o on many benchmarks.

Required env var:
    MISTRAL_API_KEY  — obtain from https://console.mistral.ai/

Optional env var:
    MISTRAL_MODEL    — defaults to "mistral-small-latest" (free tier)
"""

from __future__ import annotations

import os
from typing import List

try:
    # mistralai v2+ moved the client into mistralai.client
    from mistralai.client import Mistral
    _SDK_AVAILABLE = True
except ImportError:
    try:
        from mistralai import Mistral
        _SDK_AVAILABLE = True
    except ImportError:
        _SDK_AVAILABLE = False

from brain.adapters.base import BaseAdapter
from brain.task import Task, TaskResult, TaskType


class MistralAdapter(BaseAdapter):
    """Wraps the official `mistralai` Python SDK."""

    PROVIDER_KEY = "mistral"
    TIER = "free"
    COST_PER_1K_TOKENS = None

    SUPPORTED_TASK_TYPES: List[TaskType] = [
        TaskType.CODING,
        TaskType.REASONING,
        TaskType.GENERAL,
        TaskType.FACTUAL_QA,
        TaskType.SUMMARIZATION,
        TaskType.TRANSLATION,
        TaskType.CLASSIFICATION,
    ]

    DEFAULT_MODEL = "mistral-small-latest"

    def __init__(self) -> None:
        self._api_key = os.getenv("MISTRAL_API_KEY", "")
        self._model   = os.getenv("MISTRAL_MODEL", self.DEFAULT_MODEL)
        self._client  = None

        if _SDK_AVAILABLE and self._api_key:
            self._client = Mistral(api_key=self._api_key)

    def is_available(self) -> bool:
        return bool(_SDK_AVAILABLE and self._api_key and self._client)

    def complete(self, task: Task) -> TaskResult:
        if not self.is_available():
            return self._make_result(
                task, self.PROVIDER_KEY, self._model, "",
                error="Mistral adapter not available — check MISTRAL_API_KEY"
            )

        start = self._start_timer()
        try:
            response = self._client.chat.complete(
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
