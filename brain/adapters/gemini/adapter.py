"""
adapters/gemini_adapter.py — Adapter for Google Gemini models.

Gemini 2.5 Flash Lite is FREE up to generous rate limits and supports a
1M-token context window — ideal for long-document summarisation and
extraction tasks that would be expensive elsewhere.

Uses the current `google-genai` SDK.
Note: the older `google-generativeai` package is deprecated — do not use it.

Required env var:
    GEMINI_API_KEY  — obtain from https://aistudio.google.com/app/apikey

Optional env var:
    GEMINI_MODEL    — defaults to "gemini-2.5-flash-lite" (free, large context)
"""

from __future__ import annotations

import logging
import os
from typing import List

try:
    from google import genai
    from google.genai import types as genai_types
    # Import specific error types so we catch only API problems,
    # not programming mistakes like AttributeError or TypeError.
    from google.genai.errors import APIError as GeminiAPIError
    _SDK_AVAILABLE = True
except ImportError:
    _SDK_AVAILABLE = False

from brain.adapters.base import BaseAdapter
from brain.task import Task, TaskResult, TaskType

logger = logging.getLogger(__name__)


class GeminiAdapter(BaseAdapter):
    """Wraps the `google-genai` SDK (replaces deprecated google-generativeai)."""

    PROVIDER_KEY = "gemini"
    TIER = "free"           # Flash Lite is free up to daily/minute rate limits
    COST_PER_1K_TOKENS = None

    # Gemini's main differentiator is its enormous context window.
    # Route tasks with long input here — summarisation, extraction, translation.
    SUPPORTED_TASK_TYPES: List[TaskType] = [
        TaskType.SUMMARIZATION,
        TaskType.EXTRACTION,
        TaskType.TRANSLATION,
        TaskType.CLASSIFICATION,
        TaskType.FACTUAL_QA,
        TaskType.GENERAL,
    ]

    # gemini-2.5-flash-lite confirmed working on free tier (2026-04-17).
    # gemini-2.0-flash has a zero free-tier quota on new API keys.
    DEFAULT_MODEL = "gemini-2.5-flash-lite"

    def __init__(self) -> None:
        """Initialise the Gemini client from environment variables."""
        self._api_key    = os.getenv("GEMINI_API_KEY", "")
        self._model_name = os.getenv("GEMINI_MODEL", self.DEFAULT_MODEL)
        self._client     = None

        if _SDK_AVAILABLE and self._api_key:
            self._client = genai.Client(api_key=self._api_key)

    def is_available(self) -> bool:
        """Return True if the SDK is installed and an API key is configured."""
        return bool(_SDK_AVAILABLE and self._api_key and self._client)

    def complete(self, task: Task) -> TaskResult:
        """
        Send *task* to Gemini and return a TaskResult.

        Parameters
        ----------
        task : Task

        Returns
        -------
        TaskResult
        """
        if not self.is_available():
            return self._make_result(
                task, self.PROVIDER_KEY, self._model_name, "",
                error="Gemini adapter not available — check GEMINI_API_KEY",
            )

        start = self._start_timer()
        try:
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=task.full_prompt(),
                config=genai_types.GenerateContentConfig(
                    max_output_tokens=task.max_tokens,
                ),
            )
            content = response.text or ""

            # Token counts are available in usage_metadata when reported.
            # They may be absent on some free-tier responses.
            tokens = 0
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                tokens = (
                    (response.usage_metadata.prompt_token_count or 0)
                    + (response.usage_metadata.candidates_token_count or 0)
                )

            return self._make_result(
                task, self.PROVIDER_KEY, self._model_name,
                content=content,
                tokens_used=tokens,
                latency_ms=self._elapsed_ms(start),
                cost_usd=None,
            )

        except GeminiAPIError as exc:
            # Catch Gemini-specific API errors (auth, quota, invalid request).
            logger.warning("Gemini API error: %s", exc)
            return self._make_result(
                task, self.PROVIDER_KEY, self._model_name, "",
                latency_ms=self._elapsed_ms(start),
                error=str(exc),
            )
        except Exception as exc:  # noqa: BLE001
            # Catch remaining errors (network timeouts, unexpected SDK behaviour).
            # Log at warning level — these are usually transient.
            logger.warning("Gemini unexpected error: %s", exc)
            return self._make_result(
                task, self.PROVIDER_KEY, self._model_name, "",
                latency_ms=self._elapsed_ms(start),
                error=str(exc),
            )
