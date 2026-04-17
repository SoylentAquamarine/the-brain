"""
adapters/together_adapter.py — Adapter for Together AI.

Together AI hosts 50+ open-source models (Llama, Qwen, Mixtral, DBRX, etc.)
with $25 free credits on sign-up.  It's the best option when you want to
experiment with specific open-source models not available elsewhere.

Together's API is OpenAI-compatible — we reuse the openai SDK with Together's
base URL, so no extra dependency is needed.

Required env var:
    TOGETHER_API_KEY  — obtain from https://api.together.ai/

Optional env var:
    TOGETHER_MODEL    — defaults to "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo"
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

_TOGETHER_BASE_URL = "https://api.together.xyz/v1"


class TogetherAdapter(BaseAdapter):
    """
    Uses the OpenAI-compatible Together AI API.
    Requires `openai` SDK (already in requirements.txt).
    """

    PROVIDER_KEY = "together"
    TIER = "free"           # $25 free credits on sign-up
    COST_PER_1K_TOKENS = None

    SUPPORTED_TASK_TYPES: List[TaskType] = [
        TaskType.GENERAL,
        TaskType.FACTUAL_QA,
        TaskType.SUMMARIZATION,
        TaskType.CREATIVE,
        TaskType.REASONING,
        TaskType.CODING,
        TaskType.CLASSIFICATION,
        TaskType.TRANSLATION,
    ]

    # A capable multimodal Llama model — change via TOGETHER_MODEL env var
    # to any model listed at https://api.together.ai/models
    DEFAULT_MODEL = "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo"

    def __init__(self) -> None:
        self._api_key = os.getenv("TOGETHER_API_KEY", "")
        self._model   = os.getenv("TOGETHER_MODEL", self.DEFAULT_MODEL)
        self._client  = None

        if _SDK_AVAILABLE and self._api_key:
            self._client = OpenAI(
                api_key=self._api_key,
                base_url=_TOGETHER_BASE_URL,
            )

    def is_available(self) -> bool:
        return bool(_SDK_AVAILABLE and self._api_key and self._client)

    def complete(self, task: Task) -> TaskResult:
        if not self.is_available():
            return self._make_result(
                task, self.PROVIDER_KEY, self._model, "",
                error="Together adapter not available — check TOGETHER_API_KEY"
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
