"""
adapters/openai_adapter.py — Adapter for OpenAI's GPT models.

OpenAI is a strong general-purpose worker, especially for coding tasks
and instruction-following.  The free-tier / credit balance is consumed
only when this adapter is selected by the router.

Required env var:
    OPENAI_API_KEY  — obtain from https://platform.openai.com/api-keys

Optional env var:
    OPENAI_MODEL    — defaults to "gpt-4o-mini" (cheap, capable)
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


class OpenAIAdapter(BaseAdapter):
    """Wraps the official `openai` Python SDK (v1+)."""

    PROVIDER_KEY = "openai"
    TIER = "free"
    COST_PER_1K_TOKENS = None

    # GPT shines at coding, extraction, and structured output.
    SUPPORTED_TASK_TYPES: List[TaskType] = [
        TaskType.CODING,
        TaskType.EXTRACTION,
        TaskType.REASONING,
        TaskType.FACTUAL_QA,
        TaskType.CLASSIFICATION,
        TaskType.SUMMARIZATION,
        TaskType.TRANSLATION,
        TaskType.GENERAL,
    ]

    # Use the mini model by default to conserve credits.
    DESCRIPTION   = "GPT-4o-mini — strong all-rounder, reliable instruction following"
    QUALITY_SCORE = 7
    SPEED_TIER    = "standard"

    DEFAULT_MODEL = "gpt-4o-mini"
    MODELS = [
        "gpt-4o-mini",
        "gpt-4o",
        "o4-mini",
    ]

    def __init__(self) -> None:
        self._api_key = os.getenv("OPENAI_API_KEY", "")
        self._model   = os.getenv("OPENAI_MODEL", self.DEFAULT_MODEL)
        self._client  = None

        if _SDK_AVAILABLE and self._api_key:
            self._client = OpenAI(api_key=self._api_key)

    # ------------------------------------------------------------------
    # BaseAdapter contract
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        return bool(_SDK_AVAILABLE and self._api_key and self._client)

    def complete(self, task: Task) -> TaskResult:
        if not self.is_available():
            return self._make_result(
                task, self.PROVIDER_KEY, self._model, "",
                error="OpenAI adapter not available — check OPENAI_API_KEY"
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
            cost    = (tokens / 1000) * self.COST_PER_1K_TOKENS if self.COST_PER_1K_TOKENS else None

            return self._make_result(
                task, self.PROVIDER_KEY, response.model,
                content=content,
                tokens_used=tokens,
                latency_ms=self._elapsed_ms(start),
                cost_usd=cost,
                finish_reason=choice.finish_reason,
            )

        except Exception as exc:
            return self._make_result(
                task, self.PROVIDER_KEY, self._model, "",
                latency_ms=self._elapsed_ms(start),
                error=str(exc),
            )
