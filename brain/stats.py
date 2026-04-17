"""
stats.py — Persistent usage tracking for The Brain.

Every completed task is recorded to stats/usage.json so you can see:
  - Which providers were used and how often
  - Total tokens per provider
  - Total cost per provider
  - Claude tokens SAVED (tokens handled by non-Claude workers)
  - Average latency per provider

The stats file survives restarts and accumulates over time.
Run `python report.py` at any time for a formatted summary.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict

from brain.task import TaskResult, Task

# Stats file lives in stats/ directory at the project root.
_DEFAULT_STATS_PATH = Path(__file__).parent.parent / "stats" / "usage.json"

# Claude Sonnet pricing used to calculate "tokens saved" value.
# Even if you're not paying for the API, this shows what it WOULD have cost.
_CLAUDE_COST_PER_1K = 0.003


@dataclass
class ProviderStats:
    """Accumulated stats for one provider."""
    provider:      str
    calls:         int   = 0
    tokens:        int   = 0
    cost_usd:      float = 0.0
    total_latency: float = 0.0   # ms, for computing average
    errors:        int   = 0

    @property
    def avg_latency_ms(self) -> float:
        return (self.total_latency / self.calls) if self.calls else 0.0

    @property
    def success_rate(self) -> float:
        return ((self.calls - self.errors) / self.calls * 100) if self.calls else 0.0


@dataclass
class UsageStats:
    """Full session + lifetime stats."""
    providers:         Dict[str, ProviderStats] = field(default_factory=dict)
    total_calls:       int   = 0
    total_tokens:      int   = 0
    total_cost_usd:    float = 0.0
    claude_calls:      int   = 0
    claude_tokens:     int   = 0
    worker_tokens:     int   = 0   # tokens handled by non-Claude providers
    first_call_ts:     float = field(default_factory=time.time)
    last_call_ts:      float = field(default_factory=time.time)

    @property
    def claude_tokens_saved(self) -> int:
        """Tokens that would have gone to Claude but were handled by workers."""
        return self.worker_tokens

    @property
    def estimated_savings_usd(self) -> float:
        """Dollar value of tokens saved at Claude Sonnet rates."""
        return (self.worker_tokens / 1000) * _CLAUDE_COST_PER_1K


class StatsTracker:
    """
    Records task results and persists them to a JSON file.

    Thread-safe for single-process use (file is read-modify-write on each call).
    """

    def __init__(self, stats_path: Path = _DEFAULT_STATS_PATH) -> None:
        self._path = Path(stats_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._stats = self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, result: TaskResult, task: Task) -> None:
        """Record a completed TaskResult into persistent stats."""
        self._stats.total_calls  += 1
        self._stats.last_call_ts  = time.time()

        p = result.provider
        if p not in self._stats.providers:
            self._stats.providers[p] = ProviderStats(provider=p)

        ps = self._stats.providers[p]
        ps.calls         += 1
        ps.tokens        += result.tokens_used
        ps.total_latency += result.latency_ms
        if result.cost_usd:
            ps.cost_usd  += result.cost_usd
            self._stats.total_cost_usd += result.cost_usd
        if result.error:
            ps.errors    += 1

        self._stats.total_tokens += result.tokens_used

        # Track Claude vs worker token split.
        if p == "anthropic":
            self._stats.claude_calls  += 1
            self._stats.claude_tokens += result.tokens_used
        else:
            self._stats.worker_tokens += result.tokens_used

        self._save()

    def get(self) -> UsageStats:
        """Return the current stats (reloads from disk)."""
        self._stats = self._load()
        return self._stats

    def reset(self) -> None:
        """Wipe all stats and start fresh."""
        self._stats = UsageStats()
        self._save()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> UsageStats:
        if not self._path.exists():
            return UsageStats()
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            providers = {
                k: ProviderStats(**v)
                for k, v in raw.get("providers", {}).items()
            }
            raw["providers"] = providers
            return UsageStats(**raw)
        except Exception:
            return UsageStats()

    def _save(self) -> None:
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
        }
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# Module-level singleton — import and use directly.
tracker = StatsTracker()
