"""
adapters/ollama/adapter.py — Adapter for a local Ollama instance.

Ollama exposes an OpenAI-compatible REST API on port 11434, so we reuse
the openai SDK with a custom base_url — no extra dependency needed.

On startup, the adapter queries /api/tags to discover which models are
actually installed and picks the best available one automatically.
This means a model that is still downloading will be skipped until it
appears in the tags list, and no manual config changes are needed once
a new model finishes installing.

Required env vars:  none — defaults work out of the box for the configured host.
Optional env vars:
    OLLAMA_HOST   — IP or hostname of the Ollama machine (default: 10.197.1.211)
    OLLAMA_PORT   — port Ollama listens on            (default: 11434)
    OLLAMA_MODEL  — force a specific model (skips auto-select when set)
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from typing import List, Optional

try:
    from openai import OpenAI, APIError, APIConnectionError
    _SDK_AVAILABLE = True
except ImportError:
    _SDK_AVAILABLE = False

from brain.adapters.base import BaseAdapter
from brain.task import Task, TaskResult, TaskType

logger = logging.getLogger(__name__)

_DEFAULT_HOST = "10.197.1.211"
_DEFAULT_PORT = "11434"

# All models installed on the remote machine, ordered best-first.
# The adapter picks the first one that appears in /api/tags at startup.
_PREFERRED_ORDER = [
    "llama3.1",
    "mistral",
    "llama3.2:3b",
]


class OllamaAdapter(BaseAdapter):
    """Wraps a remote Ollama instance via its OpenAI-compatible API."""

    PROVIDER_KEY = "ollama"
    TIER = "free"
    COST_PER_1K_TOKENS = None

    SUPPORTED_TASK_TYPES: List[TaskType] = [
        TaskType.GENERAL,
        TaskType.CODING,
        TaskType.FACTUAL_QA,
        TaskType.SUMMARIZATION,
        TaskType.CLASSIFICATION,
        TaskType.REASONING,
        TaskType.CREATIVE,
        TaskType.EXTRACTION,
        TaskType.TRANSLATION,
    ]

    DESCRIPTION   = "Local Ollama (LAN) — private, zero-cost, no rate limits"
    QUALITY_SCORE = 6
    SPEED_TIER    = "fast"

    DEFAULT_MODEL = _PREFERRED_ORDER[0]
    MODELS        = _PREFERRED_ORDER

    def __init__(self) -> None:
        host = os.getenv("OLLAMA_HOST", _DEFAULT_HOST)
        port = os.getenv("OLLAMA_PORT", _DEFAULT_PORT)
        self._base_url = f"http://{host}:{port}/v1"
        self._tags_url = f"http://{host}:{port}/api/tags"

        # _model is resolved after probing; may be overridden by env var.
        self._env_model: Optional[str] = os.getenv("OLLAMA_MODEL")
        self._model = self._env_model or self.DEFAULT_MODEL

        self._client = None
        self._reachable: Optional[bool] = None
        self._installed_models: List[str] = []

        if _SDK_AVAILABLE:
            # Ollama's OpenAI-compat layer ignores the api_key but the SDK requires a non-empty value.
            self._client = OpenAI(base_url=self._base_url, api_key="ollama")

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def _probe(self) -> bool:
        """
        Query /api/tags, populate _installed_models, and pick the best model.

        Returns True if the host is reachable and at least one known model
        is installed.
        """
        try:
            with urllib.request.urlopen(self._tags_url, timeout=2) as resp:
                if resp.status != 200:
                    return False
                data = json.loads(resp.read().decode())
        except Exception:
            return False

        # /api/tags returns {"models": [{"name": "llama3.1:latest", ...}, ...]}
        raw_names = [m.get("name", "") for m in data.get("models", [])]
        # Normalise: strip ":latest" suffix so "llama3.1:latest" matches "llama3.1"
        def _base(name: str) -> str:
            return name[: -len(":latest")] if name.endswith(":latest") else name

        installed_bases = {_base(n) for n in raw_names}
        self._installed_models = [m for m in _PREFERRED_ORDER if m in installed_bases]

        if not self._installed_models:
            logger.warning(
                "Ollama reachable but none of %s are installed yet. Installed: %s",
                _PREFERRED_ORDER, raw_names,
            )
            return False

        # Respect explicit env override; otherwise pick best available.
        if self._env_model:
            self._model = self._env_model
        else:
            self._model = self._installed_models[0]

        logger.info(
            "Ollama available. Installed: %s. Using: %s",
            self._installed_models, self._model,
        )
        return True

    def is_available(self) -> bool:
        if not _SDK_AVAILABLE:
            return False
        if self._reachable is None:
            self._reachable = self._probe()
            if not self._reachable:
                logger.warning(
                    "Ollama not reachable or no models ready at %s", self._base_url
                )
        return self._reachable

    # ------------------------------------------------------------------
    # Completion
    # ------------------------------------------------------------------

    def complete(self, task: Task) -> TaskResult:
        if not self.is_available():
            return self._make_result(
                task, self.PROVIDER_KEY, self._model, "",
                error=f"Ollama not reachable at {self._base_url}",
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

        except (APIError, APIConnectionError) as exc:
            logger.warning("Ollama API error: %s", exc)
            return self._make_result(
                task, self.PROVIDER_KEY, self._model, "",
                latency_ms=self._elapsed_ms(start),
                error=str(exc),
            )
