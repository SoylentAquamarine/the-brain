"""
adapters/huggingface_adapter.py — Adapter for Hugging Face Inference API.

Hugging Face hosts thousands of open-source models on free serverless
inference endpoints.  The free tier rate-limits by model but costs nothing.
Acts as a reliable general fallback when other providers are unavailable.

Uses the `huggingface_hub` InferenceClient (OpenAI-compatible interface).

Required env var:
    HUGGINGFACE_API_KEY  — obtain from https://huggingface.co/settings/tokens
                           Create a "Read" token — that's all that's needed.

Optional env var:
    HUGGINGFACE_MODEL    — defaults to "meta-llama/Meta-Llama-3-8B-Instruct"
"""

from __future__ import annotations

import logging
import os
from typing import List

try:
    from huggingface_hub import InferenceClient
    from huggingface_hub.errors import HfHubHTTPError
    _SDK_AVAILABLE = True
except ImportError:
    _SDK_AVAILABLE = False

from brain.adapters.base import BaseAdapter
from brain.task import Task, TaskResult, TaskType

logger = logging.getLogger(__name__)


class HuggingFaceAdapter(BaseAdapter):
    """Wraps the `huggingface_hub` InferenceClient."""

    PROVIDER_KEY = "huggingface"
    TIER = "free"
    COST_PER_1K_TOKENS = None

    SUPPORTED_TASK_TYPES: List[TaskType] = [
        TaskType.GENERAL,
        TaskType.SUMMARIZATION,
        TaskType.TRANSLATION,
        TaskType.CLASSIFICATION,
        TaskType.FACTUAL_QA,
        TaskType.CREATIVE,
    ]

    # Qwen2.5-72B is openly accessible on the free serverless tier.
    # Meta-Llama models are gated and may 400 if terms not accepted.
    DESCRIPTION   = "HuggingFace serverless inference — broad model selection, last-resort fallback"
    QUALITY_SCORE = 5
    SPEED_TIER    = "slow"

    DEFAULT_MODEL = "Qwen/Qwen2.5-72B-Instruct"
    MODELS = [
        "Qwen/Qwen2.5-72B-Instruct",
        "Qwen/Qwen2.5-Coder-32B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "HuggingFaceH4/zephyr-7b-beta",
    ]

    def __init__(self) -> None:
        """Initialise the HuggingFace client from environment variables."""
        self._api_key    = os.getenv("HUGGINGFACE_API_KEY", "")
        self._model_name = os.getenv("HUGGINGFACE_MODEL", self.DEFAULT_MODEL)
        self._client     = None

        if _SDK_AVAILABLE and self._api_key:
            self._client = InferenceClient(token=self._api_key)

    def is_available(self) -> bool:
        """Return True if the SDK is installed and an API key is configured."""
        return bool(_SDK_AVAILABLE and self._api_key and self._client)

    def complete(self, task: Task) -> TaskResult:
        """
        Send *task* to HuggingFace and return a TaskResult.

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
                error="HuggingFace adapter not available — check HUGGINGFACE_API_KEY",
            )

        start = self._start_timer()
        try:
            response = self._client.chat_completion(
                model=self._model_name,
                messages=[{"role": "user", "content": task.full_prompt()}],
                max_tokens=task.max_tokens,
            )
            content = response.choices[0].message.content or ""
            tokens  = response.usage.total_tokens if response.usage else 0

            return self._make_result(
                task, self.PROVIDER_KEY, self._model_name,
                content=content,
                tokens_used=tokens,
                latency_ms=self._elapsed_ms(start),
                cost_usd=None,
            )

        except HfHubHTTPError as exc:
            # Catches 401 (bad token), 429 (rate limit), 503 (model loading), etc.
            logger.warning("HuggingFace API error: %s", exc)
            return self._make_result(
                task, self.PROVIDER_KEY, self._model_name, "",
                latency_ms=self._elapsed_ms(start),
                error=str(exc),
            )
        except Exception as exc:  # noqa: BLE001
            # Catches network timeouts and unexpected SDK changes.
            logger.warning("HuggingFace unexpected error: %s", exc)
            return self._make_result(
                task, self.PROVIDER_KEY, self._model_name, "",
                latency_ms=self._elapsed_ms(start),
                error=str(exc),
            )
