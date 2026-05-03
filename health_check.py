#!/usr/bin/env python3
"""
health_check.py — Ping every provider/model and append results to stats/health_log.json.

Runs hourly via Windows Task Scheduler. Each call appends one entry per
provider:model tested — no data is ever overwritten or averaged.

Model state rules
-----------------
- disabled : skipped entirely — never tested.
- paused   : tested but not routed. Auto-unpaused on a passing probe.
- 3+ consecutive hard failures → disabled permanently.
- timeout / rate-limit / 5xx → paused (temporary).

Spacing
-------
Models of the same provider are spaced 90 seconds apart to avoid rate limits.
Models of different providers run without delay.

    python health_check.py
"""
from __future__ import annotations

import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from brain.adapters import REGISTRY
from brain import model_state
from brain.task import Task, TaskType

LOG_FILE = Path(__file__).parent / "stats" / "health_log.json"

_DISABLE_AFTER_FAILURES = 3

# Randomized ping prompts — varied so caching can't mask a broken model.
_PROMPTS = [
    ("What is 2+2?",       "4"),
    ("What is 3+3?",       "6"),
    ("What is 5+5?",       "10"),
    ("What is 7+7?",       "14"),
    ("What is 8+8?",       "16"),
    ("What is 9+9?",       "18"),
    ("What is 6+4?",       "10"),
    ("What is 12+3?",      "15"),
    ("What is 20-5?",      "15"),
    ("What is 4 times 3?", "12"),
]


def load() -> list:
    if LOG_FILE.exists():
        raw = json.loads(LOG_FILE.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            return raw
    return []


def save(entries: list) -> None:
    LOG_FILE.parent.mkdir(exist_ok=True)
    LOG_FILE.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def ping(adapter, model: str) -> tuple[int, int, str]:
    """
    Return (ok, latency_ms, failure_type).

    ok=1 if response contains the expected answer, else 0.
    failure_type: '' | 'timeout' | 'rate_limit' | 'http_error' | 'wrong_answer' | 'error'
    """
    prompt_text, expected = random.choice(_PROMPTS)
    task = Task(f"{prompt_text} Reply with only the number.", TaskType.FACTUAL_QA, max_tokens=10)

    old = adapter._get_active_model()
    adapter._set_active_model(model)

    try:
        start = time.perf_counter()
        res   = adapter.complete(task)
        latency = int((time.perf_counter() - start) * 1000)
    except Exception as exc:
        latency = int((time.perf_counter() - start) * 1000)
        adapter._set_active_model(old)
        err = str(exc).lower()
        ftype = "timeout" if "timeout" in err else "error"
        return 0, latency, ftype
    finally:
        adapter._set_active_model(old)

    if not res.succeeded:
        err = (res.error or "").lower()
        if "timeout" in err or "timed out" in err:
            return 0, latency, "timeout"
        if "rate" in err or "429" in err:
            return 0, latency, "rate_limit"
        if any(c in err for c in ("500", "502", "503", "504")):
            return 0, latency, "http_error"
        return 0, latency, "error"

    ok = 1 if expected in (res.content or "") else 0
    return ok, latency, ("" if ok else "wrong_answer")


def _is_temporary(failure_type: str) -> bool:
    return failure_type in ("timeout", "rate_limit", "http_error")


def main() -> None:
    entries = load()
    now     = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    added   = 0

    for provider, adapter in REGISTRY.items():
        if not adapter.is_available():
            continue

        models         = adapter.list_models()
        last_model_time: float = 0.0  # track when we last pinged this provider

        for model in models:
            state = model_state.get_state(provider, model)

            if state == "disabled":
                print(f"  [{provider}/{model}] DISABLED — skipping")
                continue

            # 90-second spacing between models of the same provider
            elapsed = time.perf_counter() - last_model_time
            if last_model_time and elapsed < 90:
                wait = 90 - elapsed
                print(f"  [{provider}] spacing — waiting {wait:.0f}s before next model...")
                time.sleep(wait)

            ok, latency, failure_type = ping(adapter, model)
            last_model_time = time.perf_counter()

            # Update model state
            if ok:
                model_state.record_success(provider, model)
                if state == "paused":
                    model_state.unpause(provider, model)
                    print(f"  [{provider}/{model}] UNPAUSED — probe passing again ({latency}ms)")
                else:
                    print(f"  [{provider}/{model}] OK ({latency}ms)")
            else:
                if _is_temporary(failure_type):
                    model_state.pause(provider, model, reason=failure_type)
                    print(f"  [{provider}/{model}] PAUSED ({failure_type}, {latency}ms)")
                else:
                    model_state.pause(provider, model, reason=failure_type)
                    fails = model_state.consecutive_failures(provider, model)
                    if fails >= _DISABLE_AFTER_FAILURES:
                        model_state.disable(provider, model, reason=f"{fails} consecutive failures")
                        print(f"  [{provider}/{model}] DISABLED after {fails} failures")
                    else:
                        print(f"  [{provider}/{model}] FAIL #{fails} ({failure_type}, {latency}ms)")

            entries.append({
                "ts":           now,
                "provider":     provider,
                "model":        model,
                "ok":           ok,
                "latency_ms":   latency,
                "failure_type": failure_type,
            })
            added += 1

    save(entries)
    print(f"\nHealth check complete: {added} probes appended ({len(entries)} total in log).")


if __name__ == "__main__":
    main()
