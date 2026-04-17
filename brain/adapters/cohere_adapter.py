"""
adapters/cohere_adapter.py — Adapter for Cohere's Command models.

Cohere offers a free trial tier and specialises in retrieval-augmented
generation (RAG), text extraction, and structured summarisation.  Its
Command-R model is a solid free-tier option for extraction and classification.

Required env var:
    COHERE_API_KEY  — obtain from https://dashboard.cohere.com/api-keys

Optional env var:
    COHERE_MODEL    — defaults to "command-r" (free trial tier)
"""

from __future__ import annotations

import os
from typing import List

try:
    import cohere
    _SDK_AVAILABLE = True
except ImportError:
    _SDK_AVAILABLE = False

from brain.adapters.base import BaseAdapter
from brain.task import Task, TaskResult, TaskType


class CohereAdapter(BaseAdapter):
    """Wraps the official `cohere` Python SDK (v5+)."""

    PROVIDER_KEY = "cohere"
    TIER = "free"          # Trial key = free up to rate limits
    COST_PER_1K_TOKENS = None

    # Cohere excels at structured extraction and classification.
    SUPPORTED_TASK_TYPES: List[TaskType] = [
        TaskType.EXTRACTION,
        TaskType.CLASSIFICATION,
        TaskType.SUMMARIZATION,
        TaskType.FACTUAL_QA,
        TaskType.GENERAL,
    ]

    DEFAULT_MODEL = "command-r"

    def __init__(self) -> None:
        self._api_key = os.getenv("COHERE_API_KEY", "")
        self._model   = os.getenv("COHERE_MODEL", self.DEFAULT_MODEL)
        self._client  = None

        if _SDK_AVAILABLE and self._api_key:
            self._client = cohere.ClientV2(api_key=self._api_key)

    # ------------------------------------------------------------------
    # BaseAdapter contract
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        return bool(_SDK_AVAILABLE and self._api_key and self._client)

    def complete(self, task: Task) -> TaskResult:
        if not self.is_available():
            return self._make_result(
                task, self.PROVIDER_KEY, self._model, "",
                error="Cohere adapter not available — check COHERE_API_KEY"
            )

        start = self._start_timer()
        try:
            response = self._client.chat(
                model=self._model,
                max_tokens=task.max_tokens,
                messages=[{"role": "user", "content": task.full_prompt()}],
            )
            content = ""
            if response.message and response.message.content:
                content = response.message.content[0].text

            tokens = 0
            if response.usage and response.usage.tokens:
                tokens = (
                    (response.usage.tokens.input_tokens or 0)
                    + (response.usage.tokens.output_tokens or 0)
                )

            return self._make_result(
                task, self.PROVIDER_KEY, self._model,
                content=content,
                tokens_used=tokens,
                latency_ms=self._elapsed_ms(start),
                cost_usd=None,
                finish_reason=response.finish_reason,
            )

        except Exception as exc:
            return self._make_result(
                task, self.PROVIDER_KEY, self._model, "",
                latency_ms=self._elapsed_ms(start),
                error=str(exc),
            )
