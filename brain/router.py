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
        # Set for O(1) membership checks inside _preference_order.
        self._available_set: set = set(self._available)
        logger.info("Router ready. Available providers: %s", self._available)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(self, task: Task) -> Optional[str]:
        """
        Return the single best provider key for *task*.

        Delegates to route_ordered() and returns the first entry.

        Returns
        -------
        str or None
            Provider key, or None if no provider is available at all.
        """
        ordered = self.route_ordered(task)
        return ordered[0] if ordered else None

    def route_ordered(self, task: Task) -> List[str]:
        """
        Return the full provider fallback chain for *task*, in attempt order.

        Resolves which provider (if any) should be forced to the front, then
        delegates entirely to _preference_order() for list construction.

        Parameters
        ----------
        task : Task

        Returns
        -------
        list[str]
            All available providers ordered for this task. Empty only when
            no providers are configured at all.
        """
        return self._preference_order(task, top=self._resolve_top(task))

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

    def _resolve_top(self, task: Task) -> Optional[str]:
        """
        Return the provider that should be forced to position 0, or None.

        Checks preferred_model first, then dynamic routing. Does not build
        the full list — that is _preference_order's job.
        """
        if task.preferred_model:
            if task.preferred_model in self._available_set:
                logger.debug("Using caller-preferred provider: %s", task.preferred_model)
                return task.preferred_model
            logger.warning(
                "Preferred provider '%s' unavailable, falling back to routing.",
                task.preferred_model,
            )

        if self._dynamic and "anthropic" in self._available_set:
            provider = self._dynamic_route(task)
            if provider and provider in self._available_set:
                logger.info("Dynamic routing selected: %s", provider)
                return provider

        return None

    def _preference_order(self, task: Task, top: Optional[str] = None) -> List[str]:
        """
        Build and return the complete ordered provider list for *task*.

        This is the single place where available-provider filtering and
        ordering logic live. All other methods call this; none re-filter.

        Steps (all in one pass over available providers):
          1. Routing-table providers first, in table order.
          2. Any available provider not in the table appended at the end.
          3. For Priority.LOW tasks, free-tier providers promoted to front.
          4. If *top* is given, it is moved to position 0.

        Parameters
        ----------
        task : Task
        top  : str or None
            Provider that must appear first (from preferred_model or dynamic
            routing). Must already be validated as available by _resolve_top.

        Returns
        -------
        list[str]
        """
        preference_list = STATIC_ROUTING_TABLE.get(task.task_type, [])
        pref_set        = set(preference_list)

        # Single filtering pass — each available provider lands in exactly one list.
        ordered = [p for p in preference_list if p in self._available_set]
        extras  = [p for p in self._available  if p not in pref_set]
        full    = ordered + extras

        if not full:
            logger.warning("No providers available for %s.", task.task_type)
            return []

        # LOW priority: free-tier providers promoted to front.
        if task.priority == Priority.LOW:
            free = [p for p in full if self._registry[p].TIER == "free"]
            paid = [p for p in full if self._registry[p].TIER != "free"]
            full = free + paid

        # Promote top to position 0 without re-filtering the rest.
        if top and top != full[0]:
            full = [top] + [p for p in full if p != top]

        logger.debug("Routing order for %s: %s", task.task_type, full)
        return full

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
