"""
adapters/cloudflare_adapter.py — Adapter for Cloudflare Workers AI.

Cloudflare runs open-source models on their edge network.
Free tier available — no credit card required.

Required env var:
    CLOUDFLARE_API_KEY  — format: accountID:apiToken
                          Get your account ID from dash.cloudflare.com
                          Create an API token at dash.cloudflare.com/profile/api-tokens
                          (use the "Workers AI" template)

Optional env var:
    CLOUDFLARE_MODEL    — defaults to "@cf/meta/llama-3.1-8b-instruct"
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


class CloudflareAdapter(BaseAdapter):
    """Cloudflare Workers AI — edge-hosted open models, free tier."""

    PROVIDER_KEY = "cloudflare"
    TIER = "free"
    COST_PER_1K_TOKENS = None

    SUPPORTED_TASK_TYPES: List[TaskType] = [
        TaskType.GENERAL,
        TaskType.FACTUAL_QA,
        TaskType.SUMMARIZATION,
        TaskType.CLASSIFICATION,
        TaskType.EXTRACTION,
    ]

    DESCRIPTION   = "Edge-hosted open models on Cloudflare's global network, no credit card"
    QUALITY_SCORE = 5
    SPEED_TIER    = "fast"

    DEFAULT_MODEL = "@cf/meta/llama-3.1-8b-instruct"
    MODELS = [
        "@cf/meta/llama-3.1-8b-instruct",
        "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        "@cf/google/gemma-3-12b-it",
    ]

    def __init__(self) -> None:
        raw_key = os.getenv("CLOUDFLARE_API_KEY", "")
        self._model = os.getenv("CLOUDFLARE_MODEL", self.DEFAULT_MODEL)
        self._client = None

        if raw_key and ":" in raw_key:
            account_id, api_token = raw_key.split(":", 1)
            base_url = (
                f"https://api.cloudflare.com/client/v4/accounts"
                f"/{account_id}/ai/v1"
            )
            if _SDK_AVAILABLE:
                self._client = OpenAI(
                    base_url=base_url,
                    api_key=api_token,
                )

    def is_available(self) -> bool:
        return bool(_SDK_AVAILABLE and self._client)

    def complete(self, task: Task) -> TaskResult:
        if not self.is_available():
            return self._make_result(
                task, self.PROVIDER_KEY, self._model, "",
                error="Cloudflare adapter not available — check CLOUDFLARE_API_KEY (format: accountID:apiToken)"
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
                task, self.PROVIDER_KEY, self._model,
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
