#!/usr/bin/env python3
"""
health_check.py — Ping every provider and append results to stats/health_log.json.

Runs hourly via Windows Task Scheduler. Each call appends one entry per
provider:model tested — no data is ever overwritten or averaged.

    python health_check.py
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from brain.adapters import REGISTRY
from brain.task import Task, TaskType

LOG_FILE    = Path(__file__).parent / "stats" / "health_log.json"
PING_PROMPT = "What is 2+2? Reply with only the number."


def load() -> list:
    if LOG_FILE.exists():
        raw = json.loads(LOG_FILE.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            return raw
    return []


def save(entries: list) -> None:
    LOG_FILE.parent.mkdir(exist_ok=True)
    LOG_FILE.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def ping(adapter, model: str) -> tuple[int, int]:
    """Return (ok, latency_ms). ok=1 if response contains '4', else 0."""
    task = Task(PING_PROMPT, TaskType.FACTUAL_QA, max_tokens=10)

    old = adapter._get_active_model()
    adapter._set_active_model(model)

    try:
        start = time.perf_counter()
        res = adapter.complete(task)
        latency = int((time.perf_counter() - start) * 1000)
    finally:
        adapter._set_active_model(old)

    ok = 1 if (res.succeeded and "4" in (res.content or "")) else 0
    return ok, latency


def main() -> None:
    entries = load()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    added = 0
    for provider, adapter in REGISTRY.items():
        if not adapter.is_available():
            continue

        for model in adapter.list_models():
            ok, latency = ping(adapter, model)
            entries.append({
                "ts":         now,
                "provider":   provider,
                "model":      model,
                "ok":         ok,
                "latency_ms": latency,
            })
            added += 1

    save(entries)
    print(f"Health check complete: {added} probes appended ({len(entries)} total in log).")


if __name__ == "__main__":
    main()
