"""
adapters/deepseek_adapter.py — Adapter for DeepSeek models.

DeepSeek offers a free tier and its models rival GPT-4o on coding and math
benchmarks at a fraction of the cost.  DeepSeek-V3 (deepseek-chat) is the
general model; DeepSeek-Coder is purpose-built for code tasks.

DeepSeek's API is OpenAI-compatible — we reuse the openai SDK pointed at
DeepSeek's base URL, so no extra dependency is needed.

Required env var:
    DEEPSEEK_API_KEY  — obtain from https://platform.deepseek.com/

Optional env var:
    DEEPSEEK_MODEL    — defaults to "deepseek-chat" (DeepSeek-V3)
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

_DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class DeepSeekAdapter(BaseAdapter):
    """
    Uses the OpenAI-compatible DeepSeek API.
    Requires `openai` SDK (already in requirements.txt).
    """

    PROVIDER_KEY = "deepseek"
    TIER = "free"           # Free tier available; very cheap beyond that
    COST_PER_1K_TOKENS = None

    # DeepSeek-V3 matches or beats GPT-4o on coding and reasoning.
    SUPPORTED_TASK_TYPES: List[TaskType] = [
        TaskType.CODING,
        TaskType.REASONING,
        TaskType.FACTUAL_QA,
        TaskType.GENERAL,
        TaskType.SUMMARIZATION,
        TaskType.EXTRACTION,
        TaskType.CLASSIFICATION,
    ]

    DEFAULT_MODEL = "deepseek-chat"   # DeepSeek-V3

    def __init__(self) -> None:
        self._api_key = os.getenv("DEEPSEEK_API_KEY", "")
        self._model   = os.getenv("DEEPSEEK_MODEL", self.DEFAULT_MODEL)
        self._client  = None

        if _SDK_AVAILABLE and self._api_key:
            self._client = OpenAI(
                api_key=self._api_key,
                base_url=_DEEPSEEK_BASE_URL,
            )

    def is_available(self) -> bool:
        return bool(_SDK_AVAILABLE and self._api_key and self._client)

    def complete(self, task: Task) -> TaskResult:
        if not self.is_available():
            return self._make_result(
                task, self.PROVIDER_KEY, self._model, "",
                error="DeepSeek adapter not available — check DEEPSEEK_API_KEY"
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
