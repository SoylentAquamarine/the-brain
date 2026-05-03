"""
adapters/mistral_adapter.py — Adapter for Mistral AI models.

Mistral offers a free tier on La Plateforme.  Mistral Small is the strongest
free model for coding and reasoning tasks — it consistently outperforms
Llama 3.1 8B on code generation benchmarks.

Required env var:
    MISTRAL_API_KEY  — obtain from https://console.mistral.ai/

Optional env var:
    MISTRAL_MODEL    — defaults to "mistral-small-latest" (free tier)
"""

from __future__ import annotations

import logging
import os
from typing import List

try:
    # mistralai v2+ moved the client class into mistralai.client.
    # We try that path first, then fall back to the top-level import
    # for older versions to stay compatible across SDK releases.
    from mistralai.client import Mistral
    _SDK_AVAILABLE = True
except ImportError:
    try:
        from mistralai import Mistral  # type: ignore[no-redef]
        _SDK_AVAILABLE = True
    except ImportError:
        _SDK_AVAILABLE = False

from brain.adapters.base import BaseAdapter
from brain.task import Task, TaskResult, TaskType

logger = logging.getLogger(__name__)


class MistralAdapter(BaseAdapter):
    """Wraps the official `mistralai` Python SDK."""

    PROVIDER_KEY = "mistral"
    TIER = "free"
    COST_PER_1K_TOKENS = None

    # Mistral Small is the best free coding and reasoning model available.
    SUPPORTED_TASK_TYPES: List[TaskType] = [
        TaskType.CODING,
        TaskType.REASONING,
        TaskType.GENERAL,
        TaskType.FACTUAL_QA,
        TaskType.SUMMARIZATION,
        TaskType.TRANSLATION,
        TaskType.CLASSIFICATION,
        TaskType.CREATIVE,
        TaskType.EXTRACTION,
    ]

    DESCRIPTION   = "Mistral Small — best free quality for coding, creative, and extraction"
    QUALITY_SCORE = 8
    SPEED_TIER    = "standard"

    DEFAULT_MODEL = "mistral-small-latest"
    MODELS = [
        "mistral-small-latest",
        "mistral-large-latest",
        "open-mistral-7b",
        "open-mixtral-8x7b",
    ]

    def __init__(self) -> None:
        """Initialise the Mistral client from environment variables."""
        self._api_key = os.getenv("MISTRAL_API_KEY", "")
        self._model   = os.getenv("MISTRAL_MODEL", self.DEFAULT_MODEL)
        self._client  = None

        if _SDK_AVAILABLE and self._api_key:
            self._client = Mistral(api_key=self._api_key)

    def is_available(self) -> bool:
        """Return True if the SDK is installed and an API key is configured."""
        return bool(_SDK_AVAILABLE and self._api_key and self._client)

    def complete(self, task: Task) -> TaskResult:
        """
        Send *task* to Mistral and return a TaskResult.

        Parameters
        ----------
        task : Task

        Returns
        -------
        TaskResult
        """
        if not self.is_available():
            return self._make_result(
                task, self.PROVIDER_KEY, self._model, "",
                error="Mistral adapter not available — check MISTRAL_API_KEY",
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
                finish_reason=str(choice.finish_reason),
            )

        except Exception as exc:  # noqa: BLE001
            # Mistral SDK doesn't export granular exception types in all versions.
            # Catch broadly here but log so bugs are still visible in the logs.
            logger.warning("Mistral error: %s", exc)
            return self._make_result(
                task, self.PROVIDER_KEY, self._model, "",
                latency_ms=self._elapsed_ms(start),
                error=str(exc),
            )
