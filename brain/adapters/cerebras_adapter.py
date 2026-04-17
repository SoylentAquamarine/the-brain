"""
adapters/cerebras_adapter.py — Adapter for Cerebras Cloud inference.

Cerebras runs Llama models on custom wafer-scale silicon and is currently
the FASTEST inference available — often 1,500–2,000 tokens/sec, beating
Groq.  The free tier is generous and ideal for any task where speed matters.

Required env var:
    CEREBRAS_API_KEY  — obtain from https://cloud.cerebras.ai/

Optional env var:
    CEREBRAS_MODEL    — defaults to "llama3.1-8b" (fastest free model)
"""

from __future__ import annotations

import os
from typing import List

try:
    from cerebras.cloud.sdk import Cerebras
    _SDK_AVAILABLE = True
except ImportError:
    _SDK_AVAILABLE = False

from brain.adapters.base import BaseAdapter
from brain.task import Task, TaskResult, TaskType


class CerebrasAdapter(BaseAdapter):
    """Wraps the official `cerebras-cloud-sdk` Python SDK."""

    PROVIDER_KEY = "cerebras"
    TIER = "free"
    COST_PER_1K_TOKENS = None

    # Cerebras is the top pick for any task needing raw speed.
    SUPPORTED_TASK_TYPES: List[TaskType] = [
        TaskType.CLASSIFICATION,
        TaskType.FACTUAL_QA,
        TaskType.SUMMARIZATION,
        TaskType.GENERAL,
        TaskType.TRANSLATION,
        TaskType.EXTRACTION,
    ]

    # llama3.1-8b is the fastest; llama3.1-70b for higher quality.
    DEFAULT_MODEL = "llama3.1-8b"

    def __init__(self) -> None:
        self._api_key = os.getenv("CEREBRAS_API_KEY", "")
        self._model   = os.getenv("CEREBRAS_MODEL", self.DEFAULT_MODEL)
        self._client  = None

        if _SDK_AVAILABLE and self._api_key:
            self._client = Cerebras(api_key=self._api_key)

    def is_available(self) -> bool:
        return bool(_SDK_AVAILABLE and self._api_key and self._client)

    def complete(self, task: Task) -> TaskResult:
        if not self.is_available():
            return self._make_result(
                task, self.PROVIDER_KEY, self._model, "",
                error="Cerebras adapter not available — check CEREBRAS_API_KEY"
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
