"""
adapters/groq_adapter.py — Adapter for Groq's inference API.

Groq runs open-source models (Llama 3, Mixtral) on custom LPU hardware at
extraordinary speed — often 400+ tokens/sec.  The free tier is generous and
makes Groq the ideal choice for classification, quick Q&A, and any task where
latency matters more than frontier-model quality.

Required env var:
    GROQ_API_KEY  — obtain from https://console.groq.com/keys

Optional env var:
    GROQ_MODEL    — defaults to "llama3-8b-8192" (fast, free, capable)
"""

from __future__ import annotations

import os
from typing import List

try:
    from groq import Groq
    _SDK_AVAILABLE = True
except ImportError:
    _SDK_AVAILABLE = False

from brain.adapters.base import BaseAdapter
from brain.task import Task, TaskResult, TaskType


class GroqAdapter(BaseAdapter):
    """Wraps the official `groq` Python SDK."""

    PROVIDER_KEY = "groq"
    TIER = "free"
    COST_PER_1K_TOKENS = None  # Free tier

    # Groq/Llama is excellent for fast classification and factual retrieval.
    # Avoid routing complex reasoning or creative tasks here — use Claude/GPT.
    SUPPORTED_TASK_TYPES: List[TaskType] = [
        TaskType.CLASSIFICATION,
        TaskType.FACTUAL_QA,
        TaskType.SUMMARIZATION,
        TaskType.TRANSLATION,
        TaskType.GENERAL,
    ]

    # llama3-8b-8192 is the fastest free model; upgrade to llama3-70b for quality.
    DEFAULT_MODEL = "llama3-8b-8192"

    def __init__(self) -> None:
        self._api_key = os.getenv("GROQ_API_KEY", "")
        self._model   = os.getenv("GROQ_MODEL", self.DEFAULT_MODEL)
        self._client  = None

        if _SDK_AVAILABLE and self._api_key:
            self._client = Groq(api_key=self._api_key)

    # ------------------------------------------------------------------
    # BaseAdapter contract
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        return bool(_SDK_AVAILABLE and self._api_key and self._client)

    def complete(self, task: Task) -> TaskResult:
        if not self.is_available():
            return self._make_result(
                task, self.PROVIDER_KEY, self._model, "",
                error="Groq adapter not available — check GROQ_API_KEY"
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
                cost_usd=None,  # Free tier
                finish_reason=choice.finish_reason,
            )

        except Exception as exc:
            return self._make_result(
                task, self.PROVIDER_KEY, self._model, "",
                latency_ms=self._elapsed_ms(start),
                error=str(exc),
            )
