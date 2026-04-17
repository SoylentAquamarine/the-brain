"""
orchestrator.py — The Brain's central command.

The Orchestrator is the only public-facing entry point for callers.
It wires together the adapter registry, the router, and the retry/fallback
logic.  Think of it as the general who issues orders but never fires a rifle.

Design goals:
  - Single .run(task) call for callers — complexity lives here, not in main.py.
  - Automatic fallback: if the chosen provider fails, try the next available one.
  - Full audit trail: every attempt is logged with provider, latency, and cost.
  - Cost summary: .session_stats() gives a running total after a batch of calls.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional

from brain.adapters import REGISTRY
from brain.router import Router
from brain.stats import tracker
from brain.task import Task, TaskResult, TaskStatus

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    The central dispatcher for The Brain.

    Instantiate once and reuse for the lifetime of the application so that
    session stats accumulate across calls.

    Parameters
    ----------
    dynamic_routing : Forward to Router — enables Claude-assisted routing.
    max_fallbacks   : How many providers to try before giving up. Default 3.
    """

    def __init__(
        self,
        dynamic_routing: bool = False,
        max_fallbacks: int = 3,
    ) -> None:
        self._registry     = REGISTRY
        self._router       = Router(self._registry, dynamic=dynamic_routing)
        self._max_fallbacks = max_fallbacks

        # Session-level accounting.
        self._total_calls:   int   = 0
        self._total_tokens:  int   = 0
        self._total_cost:    float = 0.0
        self._failed_calls:  int   = 0

        available = self._router.available_providers()
        logger.info(
            "Orchestrator ready. %d provider(s) available: %s",
            len(available), available
        )

    # ------------------------------------------------------------------
    # Primary public API
    # ------------------------------------------------------------------

    def run(self, task: Task) -> TaskResult:
        """
        Execute *task* and return a TaskResult.

        The method tries providers in routing-priority order, falling back
        automatically on errors until `max_fallbacks` attempts are exhausted.
        """
        task.status = TaskStatus.ROUTING
        logger.info("Received task %s type=%s priority=%s", task.id, task.task_type.value, task.priority.value)

        # Build an ordered list of providers to attempt.
        provider_order = self._build_provider_order(task)

        if not provider_order:
            task.status = TaskStatus.FAILED
            return TaskResult(
                task_id=task.id,
                provider="none",
                model="none",
                content="",
                error="No providers are available. Check your API keys.",
            )

        last_result: Optional[TaskResult] = None

        for attempt, provider_key in enumerate(provider_order[:self._max_fallbacks], start=1):
            adapter = self._registry[provider_key]
            task.status = TaskStatus.RUNNING
            logger.info(
                "Attempt %d/%d — routing to %s",
                attempt, min(len(provider_order), self._max_fallbacks), provider_key
            )

            result = adapter.complete(task)
            self._account(result, task)

            if result.succeeded:
                task.status = TaskStatus.COMPLETED
                logger.info("Task %s completed. %s", task.id, result.summary())
                return result

            # Non-fatal error — log and try next provider.
            logger.warning(
                "Provider %s failed (attempt %d): %s — trying fallback",
                provider_key, attempt, result.error
            )
            last_result = result

        # All attempts exhausted.
        task.status = TaskStatus.FAILED
        self._failed_calls += 1
        logger.error("Task %s failed after %d attempt(s).", task.id, attempt)
        return last_result or TaskResult(
            task_id=task.id,
            provider="none",
            model="none",
            content="",
            error="All providers failed.",
        )

    def run_batch(self, tasks: List[Task]) -> List[TaskResult]:
        """
        Execute a list of tasks sequentially and return all results.

        For true parallelism, call run() from a ThreadPoolExecutor in your
        application layer — this method exists for simple scripting convenience.
        """
        return [self.run(task) for task in tasks]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def session_stats(self) -> dict:
        """Return cumulative stats for all calls made through this orchestrator."""
        return {
            "total_calls":   self._total_calls,
            "failed_calls":  self._failed_calls,
            "total_tokens":  self._total_tokens,
            "estimated_cost_usd": round(self._total_cost, 6),
        }

    def provider_status(self) -> dict:
        """Delegate to router for a full provider availability snapshot."""
        return self._router.status()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_provider_order(self, task: Task) -> List[str]:
        """
        Return an ordered list of provider keys to attempt.

        The primary choice comes from the router; remaining available providers
        are appended as automatic fallbacks so the caller never writes retry logic.
        """
        primary = self._router.route(task)
        available = self._router.available_providers()

        if primary is None:
            return available  # Everything is a fallback if routing fails.

        # Put the primary choice first, then the rest in availability order.
        return [primary] + [p for p in available if p != primary]

    def _account(self, result: TaskResult, task: Task) -> None:
        """Update session counters and persistent stats from a completed TaskResult."""
        self._total_calls  += 1
        self._total_tokens += result.tokens_used
        if result.cost_usd is not None:
            self._total_cost += result.cost_usd
        tracker.record(result, task)
