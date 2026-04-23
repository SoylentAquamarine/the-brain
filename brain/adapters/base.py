"""
adapters/base.py — Abstract base class that every provider adapter must implement.

By coding to this interface the orchestrator and router never need to know
anything about individual SDKs.  Adding a new provider means:
  1. Subclass BaseAdapter
  2. Set the class-level metadata constants
  3. Implement is_available() and complete()
  4. Place adapter.py in a subfolder under brain/adapters/ — auto-discovered.

Extension rules (enforced by convention):
  - Adapters must not import from each other.
  - Adapters must not modify router or registry logic.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import List, Optional

from brain.task import Task, TaskResult, TaskType


class BaseAdapter(ABC):
    """
    Contract that all AI provider adapters must satisfy.

    Class-level constants
    ---------------------
    PROVIDER_KEY          : Short string ID used by the router ("groq", "gemini", …).
    SUPPORTED_TASK_TYPES  : Task types this provider handles well — used as routing hints.
    TIER                  : "free" or "paid" — lets the router prefer free models.
    COST_PER_1K_TOKENS    : Approximate USD per 1 000 tokens; None for free/unknown.
    """

    PROVIDER_KEY:           str             = "base"
    SUPPORTED_TASK_TYPES:   List[TaskType]  = list(TaskType)
    TIER:                   str             = "paid"
    COST_PER_1K_TOKENS:     Optional[float] = None
    ENABLED:                bool            = True

    # ------------------------------------------------------------------
    # Abstract methods — every subclass must implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def complete(self, task: Task) -> TaskResult:
        """
        Send *task* to the provider and return a populated TaskResult.

        Implementation requirements
        ---------------------------
        - Always return a TaskResult — never raise.  Surface errors via
          TaskResult.error so the orchestrator can attempt fallbacks.
        - Record wall-clock latency using _start_timer() / _elapsed_ms().
        - If the SDK raises, catch the specific SDK exception type (not bare
          Exception) and map it to a failed TaskResult.

        Parameters
        ----------
        task : Task
            The work to perform.  Use task.full_prompt() to get the prompt
            with any context block prepended.

        Returns
        -------
        TaskResult
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """
        Return True if this adapter has valid credentials and can be called.

        Used at startup to skip unconfigured providers gracefully, and by
        the router to build the available-provider list.

        Returns
        -------
        bool
        """
        ...

    # ------------------------------------------------------------------
    # Non-abstract helpers — available to all subclasses
    # ------------------------------------------------------------------

    def validate_contract(self) -> bool:
        """
        Return True if this adapter satisfies the plugin contract.

        Default implementation always returns True.  Override to add
        adapter-specific self-checks (e.g. verify required env vars are
        set, SDK is importable, metadata fields are populated).

        Called by the plugin loader after instantiation; a False return
        allows the loader to skip registration and log a warning.

        Returns
        -------
        bool
        """
        return True

    def provider_info(self) -> dict:
        """
        Return a serialisable dict describing this provider.

        Used by Orchestrator.provider_status() and report.py to display
        provider capabilities without importing each adapter directly.

        Returns
        -------
        dict
        """
        return {
            "provider":    self.PROVIDER_KEY,
            "tier":        self.TIER,
            "cost_per_1k": self.COST_PER_1K_TOKENS,
            "task_types":  [t.value for t in self.SUPPORTED_TASK_TYPES],
            "available":   self.is_available(),
        }

    @staticmethod
    def _make_result(
        task:        Task,
        provider:    str,
        model:       str,
        content:     str,
        tokens_used: int            = 0,
        latency_ms:  float          = 0.0,
        cost_usd:    Optional[float] = None,
        error:       Optional[str]  = None,
        **metadata,
    ) -> TaskResult:
        """
        Convenience factory so subclasses don't import TaskResult directly.

        Pass keyword arguments beyond the named ones as **metadata and they
        will be stored in TaskResult.metadata for provider-specific extras
        (e.g. finish_reason, stop_reason).

        Parameters
        ----------
        task        : Originating Task (for task_id linkage).
        provider    : Provider key string.
        model       : Model name as returned by the SDK.
        content     : Response text (empty string on error).
        tokens_used : Total tokens (prompt + completion).
        latency_ms  : Wall-clock time from send to response in milliseconds.
        cost_usd    : Estimated cost, or None for free-tier calls.
        error       : Error message string, or None on success.
        **metadata  : Provider-specific extras stored in TaskResult.metadata.

        Returns
        -------
        TaskResult
        """
        return TaskResult(
            task_id=task.id,
            provider=provider,
            model=model,
            content=content,
            tokens_used=tokens_used,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            error=error,
            metadata=dict(metadata),
        )

    @staticmethod
    def _start_timer() -> float:
        """
        Return the current high-resolution monotonic time.

        Use this at the start of a provider call, then pass the result to
        _elapsed_ms() to get the wall-clock latency.

        Returns
        -------
        float  (seconds, from perf_counter)
        """
        return time.perf_counter()

    @staticmethod
    def _elapsed_ms(start: float) -> float:
        """
        Return milliseconds elapsed since *start* (from _start_timer()).

        Parameters
        ----------
        start : float
            Value returned by a previous _start_timer() call.

        Returns
        -------
        float  (milliseconds)
        """
        return (time.perf_counter() - start) * 1000.0
