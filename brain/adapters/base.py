"""
adapters/base.py — Abstract base class that every provider adapter must implement.

By coding to this interface the orchestrator and router never need to know
anything about individual SDKs.  Adding a new provider = subclass + register.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import List, Optional

from brain.task import Task, TaskResult, TaskType


class BaseAdapter(ABC):
    """
    Contract that all AI provider adapters must satisfy.

    Subclasses should:
      1. Accept API credentials in __init__ (read from env vars, not hard-coded).
      2. Implement `complete()` to call the provider and return a TaskResult.
      3. Set class-level `PROVIDER_KEY`, `SUPPORTED_TASK_TYPES`, and `TIER`.
    """

    # -----------------------------------------------------------------------
    # Class-level metadata — set in every subclass
    # -----------------------------------------------------------------------

    PROVIDER_KEY: str = "base"
    """Short identifier used by the router, e.g. "openai", "groq"."""

    SUPPORTED_TASK_TYPES: List[TaskType] = list(TaskType)
    """Task types this provider handles well (used for routing hints)."""

    TIER: str = "paid"
    """'free' or 'paid' — lets the router prefer free models when cost matters."""

    COST_PER_1K_TOKENS: Optional[float] = None
    """Approximate USD cost per 1 000 tokens (None = free / unknown)."""

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    @abstractmethod
    def complete(self, task: Task) -> TaskResult:
        """
        Send *task* to the provider and return a populated TaskResult.

        Implementations must:
          - Record latency (use `_timed_call` helper below).
          - Catch provider-specific exceptions and surface them via
            TaskResult.error rather than raising, so the orchestrator can
            attempt fallbacks gracefully.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """
        Return True if the adapter has valid credentials and can be called.

        Used at startup to skip providers that aren't configured yet.
        """
        ...

    def provider_info(self) -> dict:
        """Return a dict describing this provider (useful for /status endpoints)."""
        return {
            "provider":    self.PROVIDER_KEY,
            "tier":        self.TIER,
            "cost_per_1k": self.COST_PER_1K_TOKENS,
            "task_types":  [t.value for t in self.SUPPORTED_TASK_TYPES],
            "available":   self.is_available(),
        }

    # -----------------------------------------------------------------------
    # Helpers available to subclasses
    # -----------------------------------------------------------------------

    @staticmethod
    def _make_result(
        task: Task,
        provider: str,
        model: str,
        content: str,
        tokens_used: int = 0,
        latency_ms: float = 0.0,
        cost_usd: Optional[float] = None,
        error: Optional[str] = None,
        **metadata,
    ) -> TaskResult:
        """Convenience factory so subclasses don't import TaskResult directly."""
        return TaskResult(
            task_id=task.id,
            provider=provider,
            model=model,
            content=content,
            tokens_used=tokens_used,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            error=error,
            metadata=metadata,
        )

    @staticmethod
    def _start_timer() -> float:
        """Return the current high-resolution time for latency measurement."""
        return time.perf_counter()

    @staticmethod
    def _elapsed_ms(start: float) -> float:
        """Return milliseconds elapsed since `start`."""
        return (time.perf_counter() - start) * 1000.0
