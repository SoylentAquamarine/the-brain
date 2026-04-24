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
import re
from enum import Enum
from typing import Dict, List, Optional, Set

from brain.adapters import REGISTRY
from brain.constants import DEFAULT_MAX_FALLBACKS
from brain.router import Router
from brain.stats import tracker
from brain.task import Task, TaskResult, TaskStatus


# ---------------------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------------------

class FailureType(Enum):
    TIMEOUT     = "timeout"
    RATE_LIMIT  = "rate_limit"
    AUTH_ERROR  = "auth_error"
    MODEL_ERROR = "model_error"
    UNKNOWN     = "unknown"


_RE_TIMEOUT     = re.compile(r"timeout|timed.?out|connection.?reset|read.?error", re.I)
_RE_RATE_LIMIT  = re.compile(r"429|rate.?limit|quota|too.?many.?request", re.I)
_RE_AUTH        = re.compile(r"401|403|invalid.?api.?key|unauthorized|authentication failed|invalid.?key", re.I)
_RE_MODEL       = re.compile(r"400|404|invalid.?request|context.?length|max.?token|model.?not.?found", re.I)


def _classify_failure(error: str) -> FailureType:
    """Classify a provider error string into a FailureType."""
    if _RE_TIMEOUT.search(error):
        return FailureType.TIMEOUT
    if _RE_RATE_LIMIT.search(error):
        return FailureType.RATE_LIMIT
    if _RE_AUTH.search(error):
        return FailureType.AUTH_ERROR
    if _RE_MODEL.search(error):
        return FailureType.MODEL_ERROR
    return FailureType.UNKNOWN

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

        # Providers disabled this session: {provider: reason}
        self._session_disabled: Set[str] = set()
        self._session_disabled_reasons: Dict[str, str] = {}

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

        # Per-request skip set: providers rate-limited this call only.
        request_skipped: Set[str] = set()
        # Latency ceiling set on first timeout; providers slower than this are skipped.
        slow_ceiling_ms: Optional[float] = None
        # Historical latency cache — loaded once lazily on first timeout.
        hist_latency: Optional[Dict[str, float]] = None

        for provider_key in provider_order:
            if attempt >= self._max_fallbacks:
                break

            # --- Skip checks (do not count toward attempt budget) ---
            if provider_key in self._session_disabled:
                logger.debug("Skipping %s — auth-disabled for session.", provider_key)
                continue

            if provider_key in request_skipped:
                logger.debug("Skipping %s — rate-limited this request.", provider_key)
                continue

            if slow_ceiling_ms is not None:
                avg = (hist_latency or {}).get(provider_key, 0.0)
                if avg > slow_ceiling_ms:
                    logger.debug(
                        "Skipping %s — historically slow (%.0fms > %.0fms ceiling).",
                        provider_key, avg, slow_ceiling_ms,
                    )
                    continue

            # --- Attempt ---
            attempt += 1
            adapter = self._registry[provider_key]
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

            # --- Classify failure and update skip state ---
            failure_type = _classify_failure(result.error or "")
            logger.warning(
                "Provider %s failed (attempt %d, type=%s): %s — trying fallback",
                provider_key, attempt, failure_type.value, result.error,
            )

            if failure_type == FailureType.AUTH_ERROR:
                self._session_disabled.add(provider_key)
                self._session_disabled_reasons[provider_key] = f"auth error: {result.error}"
                logger.warning("Provider %s disabled for this session (auth error).", provider_key)

            elif failure_type == FailureType.RATE_LIMIT:
                self._session_disabled.add(provider_key)
                self._session_disabled_reasons[provider_key] = f"rate limit: {result.error}"
                logger.warning("Provider %s disabled for this session (rate limit).", provider_key)

            elif failure_type == FailureType.MODEL_ERROR:
                self._session_disabled.add(provider_key)
                self._session_disabled_reasons[provider_key] = f"model/request error: {result.error}"
                logger.warning("Provider %s disabled for this session (model error).", provider_key)

            elif failure_type == FailureType.TIMEOUT:
                # Load historical latencies once, then set a ceiling so that
                # providers historically slower than this one are skipped.
                if hist_latency is None:
                    hist_latency = self._load_provider_latencies()
                slow_ceiling_ms = result.latency_ms
                logger.debug(
                    "Timeout on %s (%.0fms) — skipping providers slower than %.0fms.",
                    provider_key, result.latency_ms, slow_ceiling_ms,
                )

            # MODEL_ERROR / UNKNOWN — fall through to next provider normally.
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

    def provider_report(self) -> str:
        """Return a human-readable session provider health report."""
        available = self._router.available_providers()
        lines = ["Provider Report", "=" * 40]
        ok = [p for p in available if p not in self._session_disabled]
        lines.append(f"Active ({len(ok)}): {', '.join(ok) or 'none'}")
        if self._session_disabled_reasons:
            lines.append(f"\nDisabled ({len(self._session_disabled_reasons)}):")
            for p, reason in self._session_disabled_reasons.items():
                lines.append(f"  {p}: {reason}")
        else:
            lines.append("No providers disabled this session.")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_provider_order(self, task: Task) -> List[str]:
        """
        Return the provider attempt order for *task* as decided by the Router.

        The Orchestrator does not modify or rebuild this list — all ordering
        logic lives exclusively in Router.route_ordered().

        Parameters
        ----------
        task : Task

        Returns
        -------
        list[str]
            Provider keys in attempt order.  May be empty if no providers
            are configured.
        """
        return self._router.route_ordered(task)

    def _load_provider_latencies(self) -> Dict[str, float]:
        """
        Return a snapshot of historical avg latency (ms) per provider from stats.

        Called at most once per run() — only when a timeout is first observed.
        Providers with no recorded calls return 0.0 (treated as fast / unknown).
        """
        try:
            return {
                key: ps.avg_latency_ms
                for key, ps in tracker.get().providers.items()
            }
        except Exception:  # noqa: BLE001
            return {}

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
