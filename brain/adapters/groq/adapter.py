"""
adapters/groq_adapter.py — Adapter for Groq's inference API.

Groq runs open-source models (Llama 3.1) on custom LPU hardware at
extraordinary speed — often 400–1500 tokens/sec.  The free tier is generous
and makes Groq an excellent choice for classification, quick Q&A, and any
task where response time matters more than frontier-model quality.

Required env var:
    GROQ_API_KEY  — obtain from https://console.groq.com/keys

Optional env var:
    GROQ_MODEL    — defaults to "llama-3.1-8b-instant" (fast, free, capable)
"""

from __future__ import annotations

import logging
import os
from typing import List

try:
    from groq import Groq
    # Import the specific exception types we want to catch so we don't
    # accidentally swallow programming errors with a bare `except Exception`.
    from groq import APIError, APIConnectionError, RateLimitError, AuthenticationError
    _SDK_AVAILABLE = True
except ImportError:
    _SDK_AVAILABLE = False

from brain.adapters.base import BaseAdapter
from brain.task import Task, TaskResult, TaskType

logger = logging.getLogger(__name__)


class GroqAdapter(BaseAdapter):
    """Wraps the official `groq` Python SDK."""

    PROVIDER_KEY = "groq"
    TIER = "free"
    COST_PER_1K_TOKENS = None   # Free tier — no per-token charge

    # Groq/Llama is excellent for fast, lightweight tasks.
    # Avoid routing complex reasoning or creative tasks here — Mistral is better.
    SUPPORTED_TASK_TYPES: List[TaskType] = [
        TaskType.CLASSIFICATION,
        TaskType.FACTUAL_QA,
        TaskType.SUMMARIZATION,
        TaskType.TRANSLATION,
        TaskType.GENERAL,
    ]

    # llama-3.1-8b-instant replaced the decommissioned llama3-8b-8192.
    # Use GROQ_MODEL env var to switch to llama-3.3-70b-versatile for higher quality.
    DESCRIPTION   = "Very fast Llama/Gemma inference on LPU hardware (~400 tok/s)"
    QUALITY_SCORE = 6
    SPEED_TIER    = "fast"

    DEFAULT_MODEL = "llama-3.1-8b-instant"
    MODELS = [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
    ]

    def __init__(self) -> None:
        """Initialise the Groq client from environment variables."""
        self._api_key = os.getenv("GROQ_API_KEY", "")
        self._model   = os.getenv("GROQ_MODEL", self.DEFAULT_MODEL)
        self._client  = None

        # Only build the client if the SDK is installed and a key exists.
        # This prevents hard failures at import time for unconfigured providers.
        if _SDK_AVAILABLE and self._api_key:
            self._client = Groq(api_key=self._api_key)

    def is_available(self) -> bool:
        """Return True if the SDK is installed and an API key is configured."""
        return bool(_SDK_AVAILABLE and self._api_key and self._client)

    def complete(self, task: Task) -> TaskResult:
        """
        Send *task* to Groq and return a TaskResult.

        Catches Groq-specific API exceptions and maps them to a failed
        TaskResult rather than raising, so the orchestrator can fall back.

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
                error="Groq adapter not available — check GROQ_API_KEY",
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

        # Catch specific Groq exceptions so programming errors still surface.
        except (APIError, APIConnectionError, RateLimitError, AuthenticationError) as exc:
            logger.warning("Groq API error: %s", exc)
            return self._make_result(
                task, self.PROVIDER_KEY, self._model, "",
                latency_ms=self._elapsed_ms(start),
                error=str(exc),
            )
