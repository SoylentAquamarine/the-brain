"""
adapters/sambanova_adapter.py — Adapter for SambaNova Cloud.

SambaNova runs Llama models on custom RDU hardware at speeds that rival
Cerebras. Completely free tier, no credit card required.

Required env var:
    SAMBANOVA_API_KEY  — obtain from https://cloud.sambanova.ai/

Optional env var:
    SAMBANOVA_MODEL    — defaults to "Meta-Llama-3.1-8B-Instruct"
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


class SambaNovaAdapter(BaseAdapter):
    """SambaNova Cloud — free, very fast Llama inference."""

    PROVIDER_KEY = "sambanova"
    TIER = "free"
    COST_PER_1K_TOKENS = None

    SUPPORTED_TASK_TYPES: List[TaskType] = [
        TaskType.FACTUAL_QA,
        TaskType.CLASSIFICATION,
        TaskType.SUMMARIZATION,
        TaskType.CODING,
        TaskType.GENERAL,
        TaskType.EXTRACTION,
    ]

    DESCRIPTION   = "SambaNova RDU — 70B/405B Llama, highest free-tier quality for reasoning"
    QUALITY_SCORE = 9
    SPEED_TIER    = "fast"

    DEFAULT_MODEL = "Meta-Llama-3.3-70B-Instruct"
    MODELS = [
        "Meta-Llama-3.3-70B-Instruct",
        "Llama-4-Maverick-17B-128E-Instruct",
        "DeepSeek-V3-0324",
    ]
    BASE_URL = "https://api.sambanova.ai/v1"

    def __init__(self) -> None:
        self._api_key = os.getenv("SAMBANOVA_API_KEY", "")
        self._model   = os.getenv("SAMBANOVA_MODEL", self.DEFAULT_MODEL)
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
                error="SambaNova adapter not available — check SAMBANOVA_API_KEY"
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
