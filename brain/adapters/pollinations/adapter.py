"""
adapters/pollinations_adapter.py — Adapter for Pollinations.ai

Pollinations.ai provides completely FREE image generation with NO API key,
NO sign-up, and NO credit card.  Just an HTTP GET request.

It also offers a text inference endpoint (powered by various open models)
that is equally free and keyless.

No env vars required — works out of the box.
"""

from __future__ import annotations

import os
import time
import urllib.parse
from pathlib import Path
from typing import List

try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

from brain.adapters.base import BaseAdapter
from brain.task import Task, TaskResult, TaskType

_TEXT_URL  = "https://text.pollinations.ai/{prompt}"
_IMAGE_URL = "https://image.pollinations.ai/prompt/{prompt}"


class PollinationsAdapter(BaseAdapter):
    """
    Completely free, keyless AI via Pollinations.ai.

    Supports both text generation and image generation.
    Set task.metadata['output'] = 'image' to generate an image instead of text.
    Image is saved to assets/ and the file path is returned as content.
    """

    PROVIDER_KEY = "pollinations"
    TIER = "free"
    COST_PER_1K_TOKENS = None

    SUPPORTED_TASK_TYPES: List[TaskType] = [
        TaskType.GENERAL,
        TaskType.CREATIVE,
        TaskType.FACTUAL_QA,
        TaskType.SUMMARIZATION,
    ]

    DESCRIPTION   = "Pollinations.ai — completely keyless, no sign-up; text and image generation"
    QUALITY_SCORE = 3
    SPEED_TIER    = "slow"

    DEFAULT_MODEL = "openai"   # Pollinations text uses openai-compatible model names
    MODELS = ["openai", "mistral", "llama"]  # text models only

    def __init__(self) -> None:
        self._api_key = os.getenv("POLLINATIONS_API_KEY", "")
        self._model   = self.DEFAULT_MODEL

    def is_available(self) -> bool:
        return _REQUESTS_AVAILABLE

    def complete(self, task: Task) -> TaskResult:
        if not self.is_available():
            return self._make_result(
                task, self.PROVIDER_KEY, self.DEFAULT_MODEL, "",
                error="requests library not installed"
            )

        # Image generation mode — triggered by metadata flag
        if task.metadata.get("output") == "image":
            return self._generate_image(task)
        return self._generate_text(task)

    def _generate_text(self, task: Task) -> TaskResult:
        start  = self._start_timer()
        model  = self._get_active_model() or self.DEFAULT_MODEL
        try:
            encoded = urllib.parse.quote(task.full_prompt())
            url     = f"{_TEXT_URL.format(prompt=encoded)}?model={model}"
            headers = {}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            r = requests.get(url, headers=headers, timeout=60)
            r.raise_for_status()
            return self._make_result(
                task, self.PROVIDER_KEY, model,
                content=r.text,
                latency_ms=self._elapsed_ms(start),
            )
        except Exception as exc:
            return self._make_result(
                task, self.PROVIDER_KEY, model, "",
                latency_ms=self._elapsed_ms(start),
                error=str(exc),
            )

    def _generate_image(self, task: Task) -> TaskResult:
        """Generate an image and save it to assets/. Returns the file path."""
        start   = self._start_timer()
        width   = task.metadata.get("width",  800)
        height  = task.metadata.get("height", 400)
        seed    = task.metadata.get("seed",   42)
        outfile = task.metadata.get("outfile", "assets/generated.png")

        try:
            encoded = urllib.parse.quote(task.prompt)
            url = (
                f"{_IMAGE_URL.format(prompt=encoded)}"
                f"?width={width}&height={height}&seed={seed}&nologo=true"
            )
            r = requests.get(url, timeout=90)
            r.raise_for_status()

            Path(outfile).parent.mkdir(parents=True, exist_ok=True)
            Path(outfile).write_bytes(r.content)

            return self._make_result(
                task, self.PROVIDER_KEY, "flux",
                content=outfile,
                latency_ms=self._elapsed_ms(start),
            )
        except Exception as exc:
            return self._make_result(
                task, self.PROVIDER_KEY, "flux", "",
                latency_ms=self._elapsed_ms(start),
                error=str(exc),
            )
