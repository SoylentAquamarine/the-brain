#!/usr/bin/env python3
"""
health_check.py — Hourly provider health check for The Brain.

Pings every configured AI provider with a simple question, measures
latency, scores the response, and appends results to stats/health_log.json.

Scoring:
  - Response contains the correct answer ("4") → quality 1.0
  - Response is non-empty but wrong answer    → quality 0.5
  - Request failed / timed out               → quality 0.0

Run manually:   python health_check.py
Run via CI:     called by .github/workflows/health-check.yml every hour
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from brain.adapters import REGISTRY
from brain.task import Task, TaskType

HEALTH_LOG = Path(__file__).parent / "stats" / "health_log.json"
PING_PROMPT = "What is 2+2? Reply with only the number, nothing else."
CORRECT_ANSWER = "4"
MAX_LOG_ENTRIES = 2016   # 9 providers × 24 hours × 7 days = keep one week


def score_response(content: str) -> float:
    """Return quality score 0.0 – 1.0 based on whether the answer is correct."""
    if not content:
        return 0.0
    cleaned = content.strip().strip(".")
    if cleaned == CORRECT_ANSWER:
        return 1.0
    if CORRECT_ANSWER in content:
        return 0.8   # correct but noisy
    return 0.5       # responded but wrong


def ping_provider(key: str, adapter) -> dict:
    """Call one provider and return a health record."""
    task = Task(
        prompt=PING_PROMPT,
        task_type=TaskType.FACTUAL_QA,
        max_tokens=10,
    )
    start = time.perf_counter()
    result = adapter.complete(task)
    latency_ms = round((time.perf_counter() - start) * 1000)

    if result.succeeded:
        quality = score_response(result.content)
        status = "ok" if quality >= 0.8 else "degraded"
    else:
        quality = 0.0
        status = "error"

    return {
        "timestamp":  datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "provider":   key,
        "model":      result.model,
        "status":     status,
        "latency_ms": latency_ms,
        "quality":    quality,
        "response":   result.content[:80] if result.succeeded else result.error[:80],
    }


def load_log() -> list:
    if HEALTH_LOG.exists():
        try:
            return json.loads(HEALTH_LOG.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def save_log(entries: list) -> None:
    HEALTH_LOG.parent.mkdir(parents=True, exist_ok=True)
    # Trim to rolling window
    trimmed = entries[-MAX_LOG_ENTRIES:]
    HEALTH_LOG.write_text(json.dumps(trimmed, indent=2), encoding="utf-8")


def main() -> None:
    print(f"\n=== Health Check — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} ===\n")

    log = load_log()
    new_entries = []

    print(f"  {'Provider':<14} {'Status':<10} {'Latency':>8}  {'Quality':>7}  Response")
    print(f"  {'-' * 70}")

    for key, adapter in REGISTRY.items():
        if not adapter.is_available():
            record = {
                "timestamp":  datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "provider":   key,
                "model":      "n/a",
                "status":     "no_key",
                "latency_ms": 0,
                "quality":    0.0,
                "response":   "API key not configured",
            }
        else:
            try:
                record = ping_provider(key, adapter)
            except Exception as exc:
                record = {
                    "timestamp":  datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "provider":   key,
                    "model":      "n/a",
                    "status":     "error",
                    "latency_ms": 0,
                    "quality":    0.0,
                    "response":   str(exc)[:80],
                }

        new_entries.append(record)
        status_icon = {"ok": "[OK]", "degraded": "[WARN]", "error": "[FAIL]", "no_key": "[SKIP]"}.get(record["status"], "?")
        print(
            f"  {key:<14} {status_icon:<10} {record['latency_ms']:>7}ms"
            f"  {record['quality']:>6.1f}  {record['response'][:40]}"
        )

    log.extend(new_entries)
    save_log(log)
    print(f"\n  Logged {len(new_entries)} entries to {HEALTH_LOG}\n")


if __name__ == "__main__":
    main()
