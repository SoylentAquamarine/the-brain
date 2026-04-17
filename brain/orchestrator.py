"""
orchestrator.py — The Brain's central command.

The Orchestrator is the only public-facing entry point for callers.
It wires together the adapter registry, the router, and the retry/fallback
logic into a single .run(task) call.

Design goals
------------
- Single .run(task) call for callers — complexity lives here, not in main.py.
- Automatic fallback: if the chosen provider fails, try the next available one.
- Full audit trail: every attempt is logged with provider, latency, and cost.
- Cost summary: .session_stats() gives a running total after any batch of calls.

Typical call flow
-----------------
  caller → Orchestrator.run(task)
              → Router.route(task)          # picks best provider
              → adapter.complete(task)      # calls the AI
              → stats tracker records result
              → TaskResult returned to caller
"""

from __future__ import annotations

import logging
from typing import List, Optional

from brain.adapters import REGISTRY
from brain.constants import DEFAULT_MAX_FALLBACKS
from brain.router import Router
from brain.stats import tracker
from brain.task import Task, TaskResult, TaskStatus

# Module-level logger — all orchestrator events flow through here.
# Callers can silence orchestrator noise with logging.getLogger("brain.orchestrator").
logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Central dispatcher for The Brain.

    Instantiate once and reuse for the lifetime of the application so that
    session stats accumulate correctly across multiple calls.

    Parameters
    ----------
    dynamic_routing : bool
        When True, Claude analyses each task's content and can override the
        static routing table.  Costs ~100 tokens per call but improves
        accuracy on ambiguous requests.  Default False.
    max_fallbacks : int
        Maximum number of providers to try before declaring a task failed.
        Increase if you want more resilience; decrease to fail fast.
    """

    def __init__(
        self,
        dynamic_routing: bool = False,
        max_fallbacks: int = DEFAULT_MAX_FALLBACKS,
    ) -> None:
        self._registry      = REGISTRY
        self._router        = Router(self._registry, dynamic=dynamic_routing)
        self._max_fallbacks = max_fallbacks

        # Session-level accounting — resets when the object is recreated.
        self._total_calls:  int   = 0
        self._total_tokens: int   = 0
        self._total_cost:   float = 0.0
        self._failed_calls: int   = 0

        available = self._router.available_providers()
        logger.info(
            "Orchestrator ready. %d provider(s) available: %s",
            len(available),
            available,
        )

    # ------------------------------------------------------------------
    # Primary public API
    # ------------------------------------------------------------------

    def run(self, task: Task) -> TaskResult:
        """
        Execute *task* and return a TaskResult.

        Tries providers in routing-priority order.  On error it falls back
        automatically — the caller never writes retry logic.

        Parameters
        ----------
        task : Task
            Fully constructed task.  Set task.preferred_model to force a
            specific provider; leave it None for automatic routing.

        Returns
        -------
        TaskResult
            Always returns a TaskResult — check .succeeded or .error to
            detect failure rather than catching exceptions.
        """
        task.status = TaskStatus.ROUTING
        logger.info(
            "Task %s  type=%-14s  priority=%s",
            task.id[:8],
            task.task_type.value,
            task.priority.value,
        )

        provider_order = self._build_provider_order(task)

        # Guard: no providers configured at all.
        if not provider_order:
            task.status = TaskStatus.FAILED
            return TaskResult(
                task_id=task.id,
                provider="none",
                model="none",
                content="",
                error="No providers are available. Check your API keys in .env",
            )

        last_result: Optional[TaskResult] = None
        attempt = 0

        for provider_key in provider_order[: self._max_fallbacks]:
            attempt += 1
            adapter    = self._registry[provider_key]
            task.status = TaskStatus.RUNNING

            logger.info(
                "Attempt %d/%d — provider: %s",
                attempt,
                min(len(provider_order), self._max_fallbacks),
                provider_key,
            )

            result = adapter.complete(task)
            self._account(result, task)

            if result.succeeded:
                task.status = TaskStatus.COMPLETED
                logger.info("Task %s done.  %s", task.id[:8], result.summary())
                return result

            # Non-fatal failure — log and try the next provider.
            logger.warning(
                "Provider %s failed (attempt %d): %s — trying fallback",
                provider_key,
                attempt,
                result.error,
            )
            last_result = result

        # Every attempt exhausted.
        task.status = TaskStatus.FAILED
        self._failed_calls += 1
        logger.error("Task %s failed after %d attempt(s).", task.id[:8], attempt)

        # Return the last failure result so the caller can inspect the error.
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
        application layer.  This method exists for simple scripting convenience.

        Parameters
        ----------
        tasks : list[Task]
            Tasks to process in order.

        Returns
        -------
        list[TaskResult]
            Results in the same order as the input tasks.
        """
        return [self.run(task) for task in tasks]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def session_stats(self) -> dict:
        """
        Return cumulative counters for all calls made through this instance.

        Returns
        -------
        dict with keys: total_calls, failed_calls, total_tokens,
                        estimated_cost_usd
        """
        return {
            "total_calls":        self._total_calls,
            "failed_calls":       self._failed_calls,
            "total_tokens":       self._total_tokens,
            "estimated_cost_usd": round(self._total_cost, 6),
        }

    def provider_status(self) -> dict:
        """
        Return a snapshot of every provider's availability and capabilities.

        Delegates to the router so callers don't need to import the registry
        directly.
        """
        return self._router.status()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_provider_order(self, task: Task) -> List[str]:
        """
        Return an ordered list of provider keys to attempt for *task*.

        The primary choice comes from the router; the remaining available
        providers are appended as automatic fallbacks.  This guarantees that
        even if the router's top pick fails, the orchestrator keeps trying
        without the caller having to do anything.

        Parameters
        ----------
        task : Task

        Returns
        -------
        list[str]
            Provider keys in attempt order.  May be empty if no providers
            are configured.
        """
        primary   = self._router.route(task)
        available = self._router.available_providers()

        if primary is None:
            # Routing failed entirely — try everything in availability order.
            return available

        # Primary first, then everything else as ordered fallbacks.
        return [primary] + [p for p in available if p != primary]

    def _account(self, result: TaskResult, task: Task) -> None:
        """
        Update in-memory session counters and write to the persistent stats file.

        Called after every attempt (success or failure) so partial usage is
        always recorded — even if the overall task ultimately fails.

        Parameters
        ----------
        result : TaskResult
        task   : Task
        """
        self._total_calls  += 1
        self._total_tokens += result.tokens_used

        # Only add cost if the provider reported one (free providers return None).
        if result.cost_usd is not None:
            self._total_cost += result.cost_usd

        # Persist to stats/usage.json so report.py can read across sessions.
        tracker.record(result, task)
