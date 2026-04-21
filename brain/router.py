"""
router.py — Smart routing logic for The Brain.

The Router decides WHICH provider adapter should handle each Task, aiming to:
  1. Use free-tier models whenever the task doesn't require frontier quality.
  2. Prefer the fastest model (Cerebras) for low-complexity work.
  3. Fall back through a priority chain until a working adapter is found.
  4. Never call a paid model when a free one is genuinely sufficient.

Two routing modes
-----------------
STATIC (default)
    Rule-based: looks up the task type in a priority table.
    Fast, deterministic, zero extra tokens spent.

DYNAMIC (opt-in)
    Claude-assisted: sends a short routing prompt so Claude can read the
    actual content and override the static choice.
    Costs ~100 tokens but catches edge cases the rules miss.
    Enable with env var BRAIN_DYNAMIC_ROUTING=1 or pass dynamic=True
    to the Router constructor.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

from brain.constants import ROUTING_MAX_TOKENS, ROUTING_PROMPT_SNIPPET_LEN
from brain.task import Task, TaskType, Priority

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Static routing table
# ---------------------------------------------------------------------------
#
# Format: { TaskType: [provider_key_in_preference_order] }
#
# Design principles encoded here:
#   - cerebras leads most lists — it is the fastest free model (~1500 tok/s).
#   - gemini leads summarisation — its 1M-token context window is unique.
#   - mistral leads coding/reasoning — best quality among free models.
#   - huggingface always anchors the end as a reliable final fallback.
#   - Priority.LOW tasks skip to the first available provider (see _static_route).

STATIC_ROUTING_TABLE: Dict[TaskType, List[str]] = {
    # Speed-first: cerebras is fastest for quick binary/label tasks.
    # cloudflare added as a fast edge option after groq.
    TaskType.CLASSIFICATION: ["cerebras", "groq",       "cloudflare", "mistral",    "gemini",     "openrouter", "huggingface"],

    # Factual Q&A: speed over depth.
    TaskType.FACTUAL_QA:     ["cerebras", "groq",       "cloudflare", "gemini",     "mistral",    "openrouter", "huggingface"],

    # Gemini's 1M context window makes it ideal for long documents.
    TaskType.SUMMARIZATION:  ["gemini",   "mistral",    "sambanova",  "cerebras",   "groq",       "openrouter", "huggingface"],

    # Mistral is the strongest free model for structured extraction.
    TaskType.EXTRACTION:     ["mistral",  "sambanova",  "gemini",     "groq",       "cerebras",   "openrouter", "huggingface"],

    # Gemini handles multilingual best among the free providers.
    TaskType.TRANSLATION:    ["gemini",   "mistral",    "cerebras",   "groq",       "openrouter", "huggingface"],

    # Fireworks (DeepSeek V3) and SambaNova 70B are strong on code.
    TaskType.CODING:         ["mistral",  "fireworks",  "sambanova",  "cerebras",   "groq",       "gemini",     "openrouter", "huggingface"],

    # SambaNova's 70B leads on deep reasoning; mistral second.
    TaskType.REASONING:      ["sambanova","mistral",    "gemini",     "fireworks",  "cerebras",   "groq",       "openrouter", "huggingface"],

    # Creative writing: mistral and sambanova have the most nuance.
    TaskType.CREATIVE:       ["mistral",  "sambanova",  "gemini",     "huggingface","cerebras",   "groq",       "openrouter"],

    # General catch-all: fast cerebras/groq first, escalate for quality.
    TaskType.GENERAL:        ["cerebras", "groq",       "cloudflare", "gemini",     "mistral",    "fireworks",  "openrouter", "huggingface"],
}

# ---------------------------------------------------------------------------
# Dynamic routing prompt
# ---------------------------------------------------------------------------
#
# Sent to Claude when dynamic routing is enabled.  Kept deliberately short
# to minimise the token cost of the routing decision itself.
# The model should reply with exactly one provider key word.

_ROUTING_PROMPT_TEMPLATE = """\
You are a routing controller for an AI orchestration system.
Available providers and their strengths:
  cerebras    : FREE, ultra-fast (~1500 tok/s)  — classification, short Q&A
  groq        : FREE, very fast (~400 tok/s)    — classification, factual Q&A
  cloudflare  : FREE, edge-hosted, fast         — general, classification
  gemini      : FREE, 1M context window         — long docs, summarisation, translation
  mistral     : FREE, high quality              — coding, extraction, creative
  sambanova   : FREE, 70B model, best quality   — reasoning, coding, creative
  fireworks   : FREE, DeepSeek V3               — coding, general
  openrouter  : FREE tier available             — general fallback, many models
  huggingface : FREE, reliable fallback         — general tasks

Task type   : {task_type}
Priority    : {priority}
Prompt (first {snippet_len} chars): {prompt_snippet}

Reply with ONLY the provider key (one lowercase word).
Choose the cheapest free option that can handle this task adequately.\
"""


class Router:
    """
    Selects the best available adapter for a given Task.

    Parameters
    ----------
    registry : dict
        Map of { provider_key: adapter_instance } from adapters/__init__.py.
    dynamic : bool
        When True, consult Claude for non-obvious routing decisions.
    """

    def __init__(self, registry: dict, dynamic: bool = False) -> None:
        self._registry = registry
        # Allow env var to override the constructor flag so callers don't
        # need to change code to enable dynamic routing.
        self._dynamic = dynamic or bool(os.getenv("BRAIN_DYNAMIC_ROUTING", ""))

        # Pre-compute which providers are available so route() doesn't
        # repeatedly call is_available() on every request.
        self._available: List[str] = [
            key for key, adapter in registry.items()
            if adapter.is_available()
        ]
        logger.info("Router ready. Available providers: %s", self._available)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(self, task: Task) -> Optional[str]:
        """
        Return the provider key that should handle *task*.

        Decision flow (in order):
          1. If task.preferred_model is set and available → use it.
          2. If dynamic routing is enabled → ask Claude.
          3. Fall back to the static routing table.

        Parameters
        ----------
        task : Task

        Returns
        -------
        str or None
            Provider key, or None if no provider is available at all.
        """
        # Honour explicit caller override first — no routing logic needed.
        if task.preferred_model:
            if task.preferred_model in self._available:
                logger.debug("Using caller-preferred provider: %s", task.preferred_model)
                return task.preferred_model
            # Warn but don't raise — fall through to automatic routing.
            logger.warning(
                "Preferred provider '%s' unavailable, falling back to routing.",
                task.preferred_model,
            )

        # Dynamic routing: Claude reads the content and can override the table.
        if self._dynamic and "anthropic" in self._available:
            provider = self._dynamic_route(task)
            if provider and provider in self._available:
                logger.info("Dynamic routing selected: %s", provider)
                return provider

        return self._static_route(task)

    def available_providers(self) -> List[str]:
        """
        Return the list of provider keys that are ready to accept calls.

        Returns
        -------
        list[str]
        """
        return list(self._available)

    def status(self) -> dict:
        """
        Return a full status dict for every registered provider.

        Returns
        -------
        dict  { provider_key: provider_info_dict }
        """
        return {
            key: adapter.provider_info()
            for key, adapter in self._registry.items()
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _static_route(self, task: Task) -> Optional[str]:
        """
        Walk the static preference list and return the first available provider.

        For Priority.LOW tasks we restrict candidates to free-tier providers
        first, only falling back to paid ones if no free provider is available.
        This prevents a cheap classification task from accidentally hitting
        a paid model just because it appears higher in the fallback chain.

        Parameters
        ----------
        task : Task

        Returns
        -------
        str or None
        """
        preference_list = STATIC_ROUTING_TABLE.get(task.task_type, [])

        # LOW priority: only use free providers to minimise cost.
        if task.priority == Priority.LOW:
            free_available = [
                p for p in preference_list
                if p in self._available
                and self._registry[p].TIER == "free"
            ]
            if free_available:
                logger.debug("LOW priority → free provider: %s", free_available[0])
                return free_available[0]
            # No free providers found — fall through to full list below.

        # Normal / HIGH priority: first available in preference order wins.
        for provider_key in preference_list:
            if provider_key in self._available:
                logger.debug("Static routing → %s", provider_key)
                return provider_key

        # Nothing in the preference list is available — use whatever we have.
        logger.warning(
            "No preferred provider available for %s; using first available.",
            task.task_type,
        )
        return self._available[0] if self._available else None

    def _dynamic_route(self, task: Task) -> Optional[str]:
        """
        Ask Claude to pick a provider based on the actual task content.

        This is intentionally a minimal call (tiny max_tokens, short prompt)
        so that the routing decision itself costs nearly nothing.

        Parameters
        ----------
        task : Task

        Returns
        -------
        str or None
            A provider key if Claude returned a valid one, otherwise None
            (caller falls back to static routing).
        """
        try:
            anthropic_adapter = self._registry["anthropic"]

            prompt = _ROUTING_PROMPT_TEMPLATE.format(
                task_type=task.task_type.value,
                priority=task.priority.value,
                snippet_len=ROUTING_PROMPT_SNIPPET_LEN,
                prompt_snippet=task.prompt[:ROUTING_PROMPT_SNIPPET_LEN],
            )

            # Import here to avoid a circular import at module load time.
            from brain.task import Task as _Task, TaskType as _TT  # noqa: PLC0415

            routing_task = _Task(
                prompt=prompt,
                task_type=_TT.CLASSIFICATION,
                max_tokens=ROUTING_MAX_TOKENS,
            )

            result = anthropic_adapter.complete(routing_task)
            if result.succeeded:
                # Extract the first word and normalise to lowercase.
                chosen = result.content.strip().lower().split()[0]
                logger.debug(
                    "Dynamic routing raw='%s' → chosen='%s'",
                    result.content.strip(),
                    chosen,
                )
                # Only accept the response if it matches a known provider key.
                return chosen if chosen in self._registry else None

        except Exception as exc:  # noqa: BLE001
            # Dynamic routing is best-effort — never let it crash the pipeline.
            logger.warning("Dynamic routing error (falling back to static): %s", exc)

        return None
