"""
adapters/ollama/adapter.py — Adapter for local Ollama inference.

Ollama serves open-source models (Llama, Qwen, Mistral, Phi, etc.) locally
or on a LAN machine.  Completely free — no API key, no rate limits, no data
leaving your network.  Best for coding and general tasks when privacy or
cost matters.  Speed depends on local hardware.

Required env vars:
    OLLAMA_HOST   — IP or hostname of the Ollama server (default: localhost)
    OLLAMA_PORT   — Port (default: 11434)

Optional env var:
    OLLAMA_MODEL  — Model to use.  If unset, auto-selects the best available
                    model from the server's installed list.

Model priority order (auto-select picks the first match found on server):
    qwen2.5:7b → qwen2.5:latest → llama3.1:latest → llama3.2:latest →
    mistral:latest → phi3:latest → llama3.2:3b → (first model listed)
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import List, Optional

from brain.adapters.base import BaseAdapter
from brain.task import Task, TaskResult, TaskType

logger = logging.getLogger(__name__)

# Models tried in order when OLLAMA_MODEL is not set.
# Coding-specialised models lead; general models follow; small/weak models last.
_AUTO_SELECT_PRIORITY = [
    "qwen2.5-coder:7b",     # best overall coding model
    "deepseek-coder:6.7b",  # fast coding + debugging
    "codellama:13b",         # large, strong reasoning
    "qwen2.5:7b",            # general reasoning + some coding
    "llama3.1:latest",       # best general assistant
    "llama3.1:8b",
    "mistral:latest",        # general-purpose
    "phi3:latest",           # small, efficient
    "llama3.2:latest",
    "llama3.2:3b",           # small, fast, weak reasoning
]

_TIMEOUT = 60  # seconds — local inference can be slow on CPU


class OllamaAdapter(BaseAdapter):
    """Calls a local or LAN Ollama server via its native REST API."""

    PROVIDER_KEY = "ollama"
    TIER = "free"
    COST_PER_1K_TOKENS = None

    SUPPORTED_TASK_TYPES: List[TaskType] = [
        TaskType.CODING,
        TaskType.GENERAL,
        TaskType.FACTUAL_QA,
        TaskType.SUMMARIZATION,
        TaskType.EXTRACTION,
        TaskType.REASONING,
        TaskType.CREATIVE,
    ]

    DESCRIPTION   = "Local Ollama — LAN/CPU inference, fully private, no API key"
    QUALITY_SCORE = 6   # qwen2.5:7b / llama3.1 are competitive with hosted 8B models
    SPEED_TIER    = "standard"

    DEFAULT_MODEL = "qwen2.5-coder:7b"

    def __init__(self) -> None:
        host = os.getenv("OLLAMA_HOST", "localhost").rstrip("/")
        port = os.getenv("OLLAMA_PORT", "11434")
        self._base_url = f"http://{host}:{port}"

        explicit_model = os.getenv("OLLAMA_MODEL", "").strip()
        self._installed_models: List[str] = []
        self._model: str = explicit_model or self.DEFAULT_MODEL
        self._ready: bool = False   # True once _probe() has run

    # ------------------------------------------------------------------
    # BaseAdapter contract
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        if not self._ready:
            self._probe()
        return bool(self._installed_models)

    def complete(self, task: Task) -> TaskResult:
        if not self.is_available():
            return self._make_result(
                task, self.PROVIDER_KEY, self._model, "",
                error=f"Ollama not reachable at {self._base_url}",
            )

        start = self._start_timer()
        url = f"{self._base_url}/api/chat"
        payload = json.dumps({
            "model": self._model,
            "messages": [{"role": "user", "content": task.full_prompt()}],
            "stream": False,
            "options": {"num_predict": task.max_tokens},
        }).encode()

        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                body = json.loads(resp.read().decode())

            content = body.get("message", {}).get("content", "")
            usage = body.get("prompt_eval_count", 0) + body.get("eval_count", 0)

            return self._make_result(
                task, self.PROVIDER_KEY, self._model,
                content=content,
                tokens_used=usage,
                latency_ms=self._elapsed_ms(start),
            )

        except urllib.error.URLError as exc:
            logger.warning("Ollama request failed: %s", exc)
            return self._make_result(
                task, self.PROVIDER_KEY, self._model, "",
                latency_ms=self._elapsed_ms(start),
                error=str(exc),
            )
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Ollama response parse error: %s", exc)
            return self._make_result(
                task, self.PROVIDER_KEY, self._model, "",
                latency_ms=self._elapsed_ms(start),
                error=f"Parse error: {exc}",
            )

    def list_models(self) -> List[str]:
        if not self._ready:
            self._probe()
        return list(self._installed_models) if self._installed_models else [self._model]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _probe(self) -> None:
        """
        Query /api/tags to get installed models, then auto-select one.
        Sets self._installed_models and self._model.  Marks _ready=True.
        """
        self._ready = True
        try:
            req = urllib.request.Request(
                f"{self._base_url}/api/tags",
                headers={"Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())

            self._installed_models = [
                m["name"] for m in data.get("models", [])
                if not m["name"].startswith("nomic-embed")  # skip embedding-only models
            ]
            logger.info("Ollama: found %d models: %s", len(self._installed_models), self._installed_models)

            # If a model was explicitly set via env var, trust it.
            if os.getenv("OLLAMA_MODEL", "").strip():
                return

            # Auto-select: walk priority list, pick first installed match.
            installed_set = set(self._installed_models)
            for candidate in _AUTO_SELECT_PRIORITY:
                if candidate in installed_set:
                    self._model = candidate
                    logger.info("Ollama: auto-selected model %s", self._model)
                    return

            # Fallback: use whatever is first in the list.
            if self._installed_models:
                self._model = self._installed_models[0]
                logger.info("Ollama: fallback model %s", self._model)

        except urllib.error.URLError as exc:
            logger.debug("Ollama not reachable at %s: %s", self._base_url, exc)
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Ollama /api/tags parse error: %s", exc)
