"""
adapters/anthropic_adapter.py — Adapter for Anthropic's Claude models.

Claude is the orchestrator's "home" model and is also available as a worker
for tasks that genuinely require its capabilities (complex reasoning, nuanced
writing, long-context synthesis).

Required env var:
    ANTHROPIC_API_KEY  — obtain from https://console.anthropic.com/
"""

from __future__ import annotations

import os
from typing import List

try:
    import anthropic
    _SDK_AVAILABLE = True
except ImportError:
    _SDK_AVAILABLE = False

from brain.adapters.base import BaseAdapter
from brain.task import Task, TaskResult, TaskType


class AnthropicAdapter(BaseAdapter):
    """Wraps the official `anthropic` Python SDK."""

    PROVIDER_KEY = "anthropic"
    TIER = "paid"
    COST_PER_1K_TOKENS = 0.003   # Sonnet 3.5 blended estimate; varies by model

    # Claude excels at everything, but we list it explicitly so the router
    # can treat it as the high-quality fallback.
    SUPPORTED_TASK_TYPES: List[TaskType] = list(TaskType)

    # Default model — override with env var ANTHROPIC_MODEL.
    DEFAULT_MODEL = "claude-sonnet-4-6"

    def __init__(self) -> None:
        self._api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self._model   = os.getenv("ANTHROPIC_MODEL", self.DEFAULT_MODEL)

        # Build the client only when the SDK is installed and a key exists.
        self._client = None
        if _SDK_AVAILABLE and self._api_key:
            self._client = anthropic.Anthropic(api_key=self._api_key)

    # ------------------------------------------------------------------
    # BaseAdapter contract
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        return bool(_SDK_AVAILABLE and self._api_key and self._client)

    def complete(self, task: Task) -> TaskResult:
        if not self.is_available():
            return self._make_result(
                task, self.PROVIDER_KEY, self._model, "",
                error="Anthropic adapter not available — check ANTHROPIC_API_KEY"
            )

        start = self._start_timer()
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=task.max_tokens,
                messages=[{"role": "user", "content": task.full_prompt()}],
            )
            content = response.content[0].text if response.content else ""
            tokens  = (response.usage.input_tokens or 0) + (response.usage.output_tokens or 0)
            cost    = (tokens / 1000) * self.COST_PER_1K_TOKENS

            return self._make_result(
                task, self.PROVIDER_KEY, response.model,
                content=content,
                tokens_used=tokens,
                latency_ms=self._elapsed_ms(start),
                cost_usd=cost,
                stop_reason=response.stop_reason,
            )

        except Exception as exc:
            return self._make_result(
                task, self.PROVIDER_KEY, self._model, "",
                latency_ms=self._elapsed_ms(start),
                error=str(exc),
            )
