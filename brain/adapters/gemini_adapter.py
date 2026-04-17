"""
adapters/gemini_adapter.py — Adapter for Google Gemini models.

Gemini 1.5 Flash is FREE up to generous rate limits and has a massive
1M-token context window — ideal for long-document summarisation and
extraction tasks that would be expensive elsewhere.

Required env var:
    GEMINI_API_KEY  — obtain from https://aistudio.google.com/app/apikey

Optional env var:
    GEMINI_MODEL    — defaults to "gemini-1.5-flash" (free tier, large context)
"""

from __future__ import annotations

import os
from typing import List

try:
    import google.generativeai as genai
    _SDK_AVAILABLE = True
except ImportError:
    _SDK_AVAILABLE = False

from brain.adapters.base import BaseAdapter
from brain.task import Task, TaskResult, TaskType


class GeminiAdapter(BaseAdapter):
    """Wraps the `google-generativeai` SDK."""

    PROVIDER_KEY = "gemini"
    TIER = "free"          # 1.5 Flash is free up to rate limits
    COST_PER_1K_TOKENS = None  # Free tier — no per-token charge

    # Flash's large context window makes it the best choice for long inputs.
    SUPPORTED_TASK_TYPES: List[TaskType] = [
        TaskType.SUMMARIZATION,
        TaskType.EXTRACTION,
        TaskType.TRANSLATION,
        TaskType.CLASSIFICATION,
        TaskType.FACTUAL_QA,
        TaskType.GENERAL,
    ]

    DEFAULT_MODEL = "gemini-1.5-flash"

    def __init__(self) -> None:
        self._api_key = os.getenv("GEMINI_API_KEY", "")
        self._model_name = os.getenv("GEMINI_MODEL", self.DEFAULT_MODEL)
        self._model = None

        if _SDK_AVAILABLE and self._api_key:
            genai.configure(api_key=self._api_key)
            self._model = genai.GenerativeModel(self._model_name)

    # ------------------------------------------------------------------
    # BaseAdapter contract
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        return bool(_SDK_AVAILABLE and self._api_key and self._model)

    def complete(self, task: Task) -> TaskResult:
        if not self.is_available():
            return self._make_result(
                task, self.PROVIDER_KEY, self._model_name, "",
                error="Gemini adapter not available — check GEMINI_API_KEY"
            )

        start = self._start_timer()
        try:
            # GenerationConfig caps the output length.
            gen_config = genai.types.GenerationConfig(
                max_output_tokens=task.max_tokens,
            )
            response = self._model.generate_content(
                task.full_prompt(),
                generation_config=gen_config,
            )
            content = response.text if response.parts else ""

            # Gemini's SDK doesn't always expose token counts on free tier.
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
                cost_usd=None,  # Free tier
            )

        except Exception as exc:
            return self._make_result(
                task, self.PROVIDER_KEY, self._model_name, "",
                latency_ms=self._elapsed_ms(start),
                error=str(exc),
            )
