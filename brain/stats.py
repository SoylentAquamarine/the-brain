"""
stats.py — Persistent usage tracking for The Brain.

Every completed task is recorded to stats/usage.json so you can see:
  - Which providers were used and how often
  - Total tokens per provider
  - Estimated cost per provider
  - Claude tokens SAVED (tokens handled by non-Claude workers)
  - Average latency per provider

The stats file survives restarts and accumulates over the lifetime of the
project.  Run `python report.py` at any time for a formatted summary, or
`python update_readme_stats.py --push` to publish stats to GitHub.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from brain.constants import CLAUDE_COST_PER_1K_TOKENS, STATS_FILE_PATH
from brain.task import TaskResult, Task

logger = logging.getLogger(__name__)

# Absolute path to the stats file, anchored to the project root.
# Using __file__ means this works regardless of where Python is invoked.
_STATS_FILE = Path(__file__).parent.parent / STATS_FILE_PATH


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ProviderStats:
    """
    Accumulated usage numbers for a single AI provider.

    All counters are additive — they grow monotonically and are never reset
    unless the user explicitly calls StatsTracker.reset().

    Attributes
    ----------
    provider      : Provider key string, e.g. "groq", "gemini".
    calls         : Total number of complete() calls made to this provider.
    tokens        : Total tokens consumed (prompt + completion combined).
    cost_usd      : Cumulative cost in USD (0.0 for free-tier providers).
    total_latency : Sum of all call latencies in ms (divide by calls for avg).
    errors        : Number of calls that returned a non-None error.
    """

    provider:      str
    calls:         int   = 0
    tokens:        int   = 0
    cost_usd:      float = 0.0
    total_latency: float = 0.0
    errors:        int   = 0

    @property
    def avg_latency_ms(self) -> float:
        """Average wall-clock latency per call in milliseconds."""
        # Guard against division by zero on the first call.
        return (self.total_latency / self.calls) if self.calls else 0.0

    @property
    def success_rate(self) -> float:
        """Percentage of calls that succeeded (0–100)."""
        if not self.calls:
            return 0.0
        return (self.calls - self.errors) / self.calls * 100.0


@dataclass
class UsageStats:
    """
    Full lifetime stats for the project, spanning all sessions.

    Attributes
    ----------
    providers      : Per-provider breakdown.
    total_calls    : Grand total across all providers.
    total_tokens   : Grand total tokens across all providers.
    total_cost_usd : Grand total cost in USD.
    claude_calls   : Calls handled by the Anthropic/Claude adapter.
    claude_tokens  : Tokens consumed by Claude specifically.
    worker_tokens  : Tokens handled by non-Claude (free) providers.
    first_call_ts  : Unix timestamp of the very first recorded call.
    last_call_ts   : Unix timestamp of the most recent call.
    """

    providers:      Dict[str, ProviderStats] = field(default_factory=dict)
    total_calls:    int   = 0
    total_tokens:   int   = 0
    total_cost_usd: float = 0.0
    claude_calls:   int   = 0
    claude_tokens:  int   = 0
    worker_tokens:  int   = 0
    first_call_ts:  float = field(default_factory=time.time)
    last_call_ts:   float = field(default_factory=time.time)
    call_log:       List[dict] = field(default_factory=list)

    @property
    def claude_tokens_saved(self) -> int:
        """
        Tokens that were handled by free workers instead of Claude.

        This is the primary "savings" metric — every token here is one that
        did NOT consume your Claude subscription quota.
        """
        return self.worker_tokens

    @property
    def estimated_savings_usd(self) -> float:
        """
        Dollar value of saved tokens at Claude Sonnet's published rate.

        Even if you're on a flat subscription, this shows what the equivalent
        API cost would have been — a useful proxy for demonstrating ROI.
        """
        return (self.worker_tokens / 1000) * CLAUDE_COST_PER_1K_TOKENS


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class StatsTracker:
    """
    Records task results and persists them to a JSON file.

    The file is read-modify-write on every record() call, which keeps it
    consistent across sessions without needing a background process or
    in-memory cache beyond a single session.

    Parameters
    ----------
    stats_path : Path
        Path to the JSON file.  Created automatically on first write.
    """

    def __init__(self, stats_path: Path = _STATS_FILE) -> None:
        self._path  = Path(stats_path)
        # Ensure the parent directory exists before we ever try to write.
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._stats = self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, result: TaskResult, task: Task) -> None:
        """
        Add a completed TaskResult to the persistent stats.

        Called by the Orchestrator after every attempt (success or failure)
        so even errored calls contribute to the error-rate metric.

        Parameters
        ----------
        result : TaskResult
            The outcome of the adapter call.
        task   : Task
            The originating task (currently unused but kept for future
            enrichment, e.g. tagging by task_type in per-provider breakdown).
        """
        self._stats.total_calls  += 1
        self._stats.last_call_ts  = time.time()

        provider = result.provider

        # Create a new ProviderStats entry on first encounter.
        if provider not in self._stats.providers:
            self._stats.providers[provider] = ProviderStats(provider=provider)

        ps = self._stats.providers[provider]
        ps.calls         += 1
        ps.tokens        += result.tokens_used
        ps.total_latency += result.latency_ms

        if result.cost_usd is not None:
            ps.cost_usd            += result.cost_usd
            self._stats.total_cost_usd += result.cost_usd

        if result.error:
            ps.errors += 1

        self._stats.total_tokens += result.tokens_used

        # Append a timestamped entry to the activity log (keep last 100).
        self._stats.call_log.append({
            "ts":       self._stats.last_call_ts,
            "provider": provider,
            "type":     task.task_type.value,
            "tokens":   result.tokens_used,
            "ms":       int(result.latency_ms),
        })
        if len(self._stats.call_log) > 100:
            self._stats.call_log = self._stats.call_log[-100:]

        # Track Claude vs free-worker token split for the savings metric.
        if provider == "anthropic":
            self._stats.claude_calls  += 1
            self._stats.claude_tokens += result.tokens_used
        else:
            # Every token here is one NOT spent on Claude.
            self._stats.worker_tokens += result.tokens_used

        self._save()

    def get(self) -> UsageStats:
        """
        Return the current stats, reloading from disk to pick up any
        changes made by other sessions.

        Returns
        -------
        UsageStats
        """
        self._stats = self._load()
        return self._stats

    def reset(self) -> None:
        """
        Wipe all recorded stats and start fresh.

        Use with caution — this is irreversible unless you have a git backup
        of stats/usage.json.
        """
        self._stats = UsageStats()
        self._save()
        logger.info("Stats reset.")

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _load(self) -> UsageStats:
        """
        Read and deserialise stats/usage.json.

        Returns a blank UsageStats if the file is missing or corrupt,
        so the first run always succeeds cleanly.
        """
        if not self._path.exists():
            return UsageStats()

        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))

            # Deserialise nested ProviderStats objects from plain dicts.
            providers = {
                k: ProviderStats(**v)
                for k, v in raw.get("providers", {}).items()
            }
            raw["providers"] = providers
            raw.setdefault("call_log", [])
            return UsageStats(**raw)

        except (json.JSONDecodeError, TypeError, KeyError) as exc:
            # Corrupt file — log a warning and return blank stats so we
            # don't crash the orchestrator just because of a bad JSON file.
            logger.warning(
                "Could not load stats from %s (%s). Starting fresh.",
                self._path,
                exc,
            )
            return UsageStats()

    def _save(self) -> None:
        """
        Serialise current stats to stats/usage.json.

        Uses a plain dict rather than dataclasses.asdict on the top-level
        object so we can control exactly what gets written.
        """
        data = {
            "providers":      {k: asdict(v) for k, v in self._stats.providers.items()},
            "total_calls":    self._stats.total_calls,
            "total_tokens":   self._stats.total_tokens,
            "total_cost_usd": self._stats.total_cost_usd,
            "claude_calls":   self._stats.claude_calls,
            "claude_tokens":  self._stats.claude_tokens,
            "worker_tokens":  self._stats.worker_tokens,
            "first_call_ts":  self._stats.first_call_ts,
            "last_call_ts":   self._stats.last_call_ts,
            "call_log":       self._stats.call_log,
        }
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

# Import and use this directly rather than instantiating StatsTracker yourself.
# All sessions within the same process share one tracker instance.
tracker = StatsTracker()
