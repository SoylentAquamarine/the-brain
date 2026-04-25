"""
cache.py — Prompt-response cache for The Brain.

Identical prompts to the same provider return instantly from disk instead
of spending tokens on a real API call.

Cache key  : SHA-256 of (prompt, task_type, preferred_model, max_tokens)
Cache store: stats/cache.json  (human-readable, survives restarts)
TTL        : CACHE_TTL_SECONDS (default 24 h) — stale entries are skipped
             and lazily evicted on the next write.
Eviction   : LRU-style — when the store exceeds CACHE_MAX_ENTRIES the
             oldest (by cached_at) entries are dropped.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Optional

from brain.constants import CACHE_FILE_PATH, CACHE_MAX_ENTRIES, CACHE_TTL_SECONDS
from brain.task import Task, TaskResult

logger = logging.getLogger(__name__)

_CACHE_FILE = Path(__file__).parent.parent / CACHE_FILE_PATH


class PromptCache:
    """
    Persistent file-backed cache for TaskResults.

    Thread-safety: each read/write reloads and re-saves the JSON file, so
    concurrent processes stay in sync at the cost of extra I/O.  For a
    single-process CLI this is fine.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, task: Task) -> Optional[TaskResult]:
        """
        Return a cached TaskResult for *task*, or None on a miss/expiry.

        Parameters
        ----------
        task : Task

        Returns
        -------
        TaskResult or None
        """
        key = _make_key(task)
        store = self._load()
        entry = store.get(key)

        if entry is None:
            return None

        age = time.time() - entry.get("cached_at", 0)
        if age > CACHE_TTL_SECONDS:
            logger.debug("Cache expired for key %s (age %.0fs)", key[:8], age)
            return None

        logger.info(
            "Cache hit [%s] provider=%s age=%.0fs",
            key[:8], entry.get("provider"), age,
        )
        return TaskResult(
            task_id=task.id,
            provider=entry["provider"],
            model=entry["model"],
            content=entry["content"],
            tokens_used=0,       # cached — no tokens consumed
            latency_ms=0.0,
            cost_usd=None,
            error=None,
            metadata={"cache_hit": True, "cached_age_s": round(age)},
        )

    def put(self, task: Task, result: TaskResult) -> None:
        """
        Store a successful *result* for *task*.

        No-ops on failed results so we never cache errors.

        Parameters
        ----------
        task   : Task
        result : TaskResult
        """
        if not result.succeeded:
            return

        key = _make_key(task)
        store = self._load()

        store[key] = {
            "provider":   result.provider,
            "model":      result.model,
            "content":    result.content,
            "cached_at":  time.time(),
            "prompt_prefix": task.prompt[:80],   # for human inspection only
        }

        self._evict(store)
        self._save(store)
        logger.debug("Cache stored key %s (provider=%s)", key[:8], result.provider)

    def clear(self) -> int:
        """Delete all cache entries. Returns the number of entries removed."""
        store = self._load()
        count = len(store)
        self._save({})
        logger.info("Cache cleared (%d entries removed).", count)
        return count

    def stats(self) -> dict:
        """Return a summary dict for display in report.py / status.py."""
        store = self._load()
        now = time.time()
        fresh = sum(1 for e in store.values() if now - e.get("cached_at", 0) <= CACHE_TTL_SECONDS)
        return {
            "total_entries": len(store),
            "fresh_entries": fresh,
            "stale_entries": len(store) - fresh,
            "cache_file":    str(_CACHE_FILE),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> dict:
        if not _CACHE_FILE.exists():
            return {}
        try:
            return json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Cache load error (returning empty): %s", exc)
            return {}

    def _save(self, store: dict) -> None:
        try:
            _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            _CACHE_FILE.write_text(
                json.dumps(store, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Cache save error: %s", exc)

    def _evict(self, store: dict) -> None:
        if len(store) <= CACHE_MAX_ENTRIES:
            return
        # Drop oldest entries by cached_at timestamp.
        sorted_keys = sorted(store, key=lambda k: store[k].get("cached_at", 0))
        for k in sorted_keys[:len(store) - CACHE_MAX_ENTRIES]:
            del store[k]


def _make_key(task: Task) -> str:
    """
    Return a stable SHA-256 hex key for *task*.

    Includes prompt, task_type, preferred_model, and max_tokens so that
    the same prompt to a different provider gets its own cache slot.
    """
    payload = json.dumps(
        {
            "prompt":    task.prompt,
            "type":      task.task_type.value,
            "provider":  task.preferred_model or "",
            "max_tokens": task.max_tokens,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


# Module-level singleton — import and use this everywhere.
cache = PromptCache()
