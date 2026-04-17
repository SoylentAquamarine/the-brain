"""
router.py — Smart routing logic for The Brain.

The Router decides WHICH provider adapter should handle each Task, aiming to:
  1. Use free-tier models whenever the task doesn't require frontier quality.
  2. Prefer fast models (Groq) for low-complexity work.
  3. Fall back through a priority chain until a working adapter is found.
  4. Never call Claude (paid) when a free model is genuinely sufficient.

Two routing modes
-----------------
STATIC   — Rule-based: looks up the task type against a priority table.
           Fast, deterministic, zero extra tokens spent.

DYNAMIC  — Claude-assisted: sends a tiny routing prompt to Claude so it can
           read the actual content and override the static choice.
           Costs ~100 tokens but catches edge cases the rules miss.
           Enabled by setting BRAIN_DYNAMIC_ROUTING=1 in your env.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

from brain.task import Task, TaskType, Priority

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Static routing table
# ---------------------------------------------------------------------------
#
# Format: { TaskType: [provider_key_in_preference_order] }
#
# Rules of thumb encoded here:
#   - Free models (groq, gemini, cohere) go first.
#   - Paid models (openai, anthropic) serve as high-quality fallbacks.
#   - For CODING and REASONING, the paid models lead because accuracy matters.
#   - Priority.LOW tasks will only walk as far as the first available free model.

STATIC_ROUTING_TABLE: Dict[TaskType, List[str]] = {
    TaskType.CLASSIFICATION:  ["groq",     "cohere",     "gemini", "openai", "anthropic"],
    TaskType.FACTUAL_QA:      ["groq",     "gemini",     "cohere", "openai", "anthropic"],
    TaskType.SUMMARIZATION:   ["gemini",   "groq",       "cohere", "openai", "anthropic"],
    TaskType.EXTRACTION:      ["cohere",   "gemini",     "groq",   "openai", "anthropic"],
    TaskType.TRANSLATION:     ["gemini",   "groq",       "cohere", "openai", "anthropic"],
    TaskType.CODING:          ["openai",   "anthropic",  "groq",   "gemini", "cohere"],
    TaskType.REASONING:       ["anthropic","openai",     "gemini", "groq",   "cohere"],
    TaskType.CREATIVE:        ["anthropic","openai",     "gemini", "groq",   "cohere"],
    TaskType.GENERAL:         ["groq",     "gemini",     "cohere", "openai", "anthropic"],
}

# ---------------------------------------------------------------------------
# Routing prompt used in DYNAMIC mode
# ---------------------------------------------------------------------------

_ROUTING_PROMPT_TEMPLATE = """You are a routing controller for an AI orchestration system.
Available providers and their strengths:
  - groq      : FREE, ultra-fast, Llama 3 8B — best for classification, short Q&A
  - gemini    : FREE, 1M context window — best for long docs, summarisation, translation
  - cohere    : FREE, Command-R — best for extraction and classification
  - openai    : PAID, GPT-4o-mini — best for coding, structured tasks
  - anthropic : PAID, Claude — best for complex reasoning, nuanced writing, fallback

Task type   : {task_type}
Priority    : {priority}
Prompt (first 300 chars): {prompt_snippet}

Reply with ONLY the provider key (one word, lowercase) that should handle this task.
Choose the cheapest free option that can handle it adequately.
Only choose a paid provider when quality is critical."""


class Router:
    """
    Selects the best available adapter for a given Task.

    Parameters
    ----------
    registry    : Dict of { provider_key: adapter_instance } from adapters/__init__.py
    dynamic     : When True, consult Claude for non-obvious routing decisions.
    """

    def __init__(self, registry: dict, dynamic: bool = False) -> None:
        self._registry  = registry
        self._dynamic   = dynamic or bool(os.getenv("BRAIN_DYNAMIC_ROUTING", ""))

        # Pre-filter to available adapters so we don't waste time trying dead ones.
        self._available: List[str] = [
            key for key, adapter in registry.items()
            if adapter.is_available()
        ]
        logger.info("Router initialised. Available providers: %s", self._available)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(self, task: Task) -> Optional[str]:
        """
        Return the provider key that should handle *task*, or None if no
        suitable provider is available.

        Decision flow:
          1. If task.preferred_model is set and available, use it immediately.
          2. If dynamic routing is enabled, ask Claude.
          3. Fall back to the static routing table.
        """
        # Caller forced a specific provider.
        if task.preferred_model:
            if task.preferred_model in self._available:
                logger.debug("Using caller-preferred provider: %s", task.preferred_model)
                return task.preferred_model
            logger.warning(
                "Preferred provider '%s' is not available, falling back to routing.",
                task.preferred_model,
            )

        # Dynamic routing (optional, costs ~100 tokens on Claude).
        if self._dynamic and "anthropic" in self._available:
            provider = self._dynamic_route(task)
            if provider and provider in self._available:
                logger.info("Dynamic routing selected: %s", provider)
                return provider

        # Static rule-based routing.
        return self._static_route(task)

    def available_providers(self) -> List[str]:
        """Return the list of provider keys that are ready to accept calls."""
        return list(self._available)

    def status(self) -> dict:
        """Return a full status dict for all registered providers."""
        return {
            key: adapter.provider_info()
            for key, adapter in self._registry.items()
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _static_route(self, task: Task) -> Optional[str]:
        """Walk the static preference list and return the first available provider."""
        preference_list = STATIC_ROUTING_TABLE.get(task.task_type, [])

        # For LOW priority tasks, only consider free providers unless none are available.
        if task.priority == Priority.LOW:
            free_providers = [
                p for p in preference_list
                if p in self._available
                and self._registry[p].TIER == "free"
            ]
            if free_providers:
                logger.debug("LOW priority — chose free provider: %s", free_providers[0])
                return free_providers[0]

        for provider_key in preference_list:
            if provider_key in self._available:
                logger.debug("Static routing selected: %s", provider_key)
                return provider_key

        # Nothing in the preference list is available.
        logger.warning("No preferred provider available for task type %s", task.task_type)
        return self._available[0] if self._available else None

    def _dynamic_route(self, task: Task) -> Optional[str]:
        """
        Ask Claude to pick a provider based on the actual task content.

        This is intentionally a lightweight call (short prompt, tiny max_tokens)
        to keep its own cost negligible.
        """
        try:
            anthropic_adapter = self._registry["anthropic"]
            prompt = _ROUTING_PROMPT_TEMPLATE.format(
                task_type=task.task_type.value,
                priority=task.priority.value,
                prompt_snippet=task.prompt[:300],
            )
            # Minimal token usage — we only need one word back.
            from brain.task import Task as _Task, TaskType as _TT
            routing_task = _Task(
                prompt=prompt,
                task_type=_TT.CLASSIFICATION,
                max_tokens=20,
            )
            result = anthropic_adapter.complete(routing_task)
            if result.succeeded:
                chosen = result.content.strip().lower().split()[0]
                logger.debug("Dynamic routing raw response: '%s' → chosen: '%s'", result.content, chosen)
                return chosen if chosen in self._registry else None
        except Exception as exc:
            logger.warning("Dynamic routing failed, falling back to static: %s", exc)
        return None
