"""
task.py — Core data structures for The Brain.

Every request flowing through the system is a Task; every response is a TaskResult.
Keeping these as plain dataclasses means they are serialisable, loggable, and
easy to pass between the orchestrator and any adapter without coupling.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class TaskType(str, Enum):
    """
    Broad categories used by the router to select the best model.

    Why categories matter:
      - Different models excel at different task types.
      - Routing by category lets us prefer fast/cheap models for simple work
        and reserve powerful (expensive) models for tasks that need them.
    """
    CLASSIFICATION  = "classification"   # label / yes-no / short answer
    SUMMARIZATION   = "summarization"    # condense long text
    CODING          = "coding"           # write, explain, or debug code
    CREATIVE        = "creative"         # stories, marketing copy, brainstorming
    REASONING       = "reasoning"        # multi-step logic, math, planning
    FACTUAL_QA      = "factual_qa"       # direct knowledge questions
    EXTRACTION      = "extraction"       # pull structured data from text
    TRANSLATION     = "translation"      # language conversion
    GENERAL         = "general"          # catch-all when type is unknown


class Priority(str, Enum):
    """
    Caller-supplied urgency hint.

    HIGH   → prefer speed and quality, cost is secondary.
    NORMAL → balanced routing (default).
    LOW    → prefer cheapest available model.
    """
    HIGH   = "high"
    NORMAL = "normal"
    LOW    = "low"


class TaskStatus(str, Enum):
    """Lifecycle state of a Task."""
    PENDING    = "pending"
    ROUTING    = "routing"    # orchestrator deciding which model to use
    RUNNING    = "running"    # delegated to a provider adapter
    COMPLETED  = "completed"
    FAILED     = "failed"


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

@dataclass
class Task:
    """
    A single unit of work submitted to The Brain.

    Attributes
    ----------
    prompt          : The user-facing instruction or question.
    task_type       : Hint about what kind of work this is.
    priority        : Speed/cost trade-off preference.
    context         : Optional surrounding text (e.g. document to summarise).
    max_tokens      : Hard ceiling on response length passed to the provider.
    preferred_model : Bypass routing and force a specific provider key, e.g. "groq".
    metadata        : Free-form dict for caller-supplied tags, trace IDs, etc.
    id              : Auto-generated UUID for correlation / logging.
    created_at      : Unix timestamp set at construction.
    status          : Mutable lifecycle state.
    """
    prompt:          str
    task_type:       TaskType       = TaskType.GENERAL
    priority:        Priority       = Priority.NORMAL
    context:         Optional[str]  = None
    max_tokens:      int            = 1024
    preferred_model: Optional[str]  = None   # e.g. "openai", "gemini", "groq"
    metadata:        dict           = field(default_factory=dict)

    # Set automatically — callers should not pass these.
    id:         str        = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float      = field(default_factory=time.time)
    status:     TaskStatus = field(default=TaskStatus.PENDING)

    def full_prompt(self) -> str:
        """Return the prompt with any context block prepended."""
        if self.context:
            return f"Context:\n{self.context}\n\n{self.prompt}"
        return self.prompt


# ---------------------------------------------------------------------------
# TaskResult
# ---------------------------------------------------------------------------

@dataclass
class TaskResult:
    """
    The outcome of executing a Task.

    Attributes
    ----------
    task_id        : Links back to the originating Task.id.
    provider       : Which adapter actually ran this (e.g. "groq", "gemini").
    model          : The specific model string reported by the provider.
    content        : The textual response.
    tokens_used    : Total tokens consumed (prompt + completion) if available.
    latency_ms     : Wall-clock time from request send to response received.
    cost_usd       : Estimated cost in USD; None if the tier is free/unknown.
    error          : Non-None if the call failed; content will be empty string.
    metadata       : Provider-specific extras (finish reason, etc.).
    """
    task_id:     str
    provider:    str
    model:       str
    content:     str
    tokens_used: int            = 0
    latency_ms:  float          = 0.0
    cost_usd:    Optional[float] = None
    error:       Optional[str]  = None
    metadata:    dict           = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.error is None

    def summary(self) -> str:
        """One-line human-readable result summary for logging."""
        status = "OK" if self.succeeded else f"ERR: {self.error}"
        return (
            f"[{status}] provider={self.provider} model={self.model} "
            f"tokens={self.tokens_used} latency={self.latency_ms:.0f}ms"
        )
