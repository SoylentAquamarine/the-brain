#!/usr/bin/env python3
"""
Crash-safe AI provider health checker.
Never stops on errors. Always logs everything.
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

_PROMPTS = [
    ("What is 2+2?", "4"),
    ("What is 3+3?", "6"),
    ("What is 5+5?", "10"),
    ("What is 7+7?", "14"),
    ("What is 8+8?", "16"),
    ("What is 9+9?", "18"),
    ("What is 6+4?", "10"),
    ("What is 12+3?", "15"),
    ("What is 20-5?", "15"),
    ("What is 4 times 3?", "12"),
]


def load():
    if LOG_FILE.exists():
        try:
            return json.loads(LOG_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def save(entries):
    LOG_FILE.parent.mkdir(exist_ok=True)
    LOG_FILE.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def safe_str(x):
    if x is None:
        return ""
    return str(x)


def ping(adapter, model):
    prompt_text, expected = random.choice(_PROMPTS)
    task = Task(
        f"{prompt_text} Reply with only the number.",
        TaskType.FACTUAL_QA,
        max_tokens=10
    )

    old = adapter._get_active_model()
    adapter._set_active_model(model)

    start = time.perf_counter()

    try:
        res = adapter.complete(task)
        latency = int((time.perf_counter() - start) * 1000)
    except Exception as exc:
        adapter._set_active_model(old)
        return 0, int((time.perf_counter() - start) * 1000), f"crash:{str(exc)[:80]}"

    finally:
        adapter._set_active_model(old)

    # --- failure cases from adapter ---
    if not res.succeeded:
        err = safe_str(res.error).lower()

        if "timeout" in err:
            return 0, latency, "timeout"
        if "rate" in err or "429" in err:
            return 0, latency, "rate_limit"
        if any(c in err for c in ("500", "502", "503", "504")):
            return 0, latency, "http_error"

        return 0, latency, "error"

    # --- SAFE content handling (FIXED CRASH HERE) ---
    content = safe_str(res.content)

    ok = 1 if expected in content else 0
    return ok, latency, ("" if ok else "wrong_answer")


def is_temporary(failure_type):
    return failure_type in ("timeout", "rate_limit", "http_error")


def main():
    entries = load()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for provider, adapter in REGISTRY.items():
        if not adapter.is_available():
            continue

        last_time = 0

        for model in adapter.list_models():

            state = model_state.get_state(provider, model)

            if state == "disabled":
                entries.append({
                    "ts": now,
                    "provider": provider,
                    "model": model,
                    "ok": 0,
                    "latency_ms": 0,
                    "failure_type": "disabled_skip"
                })
                continue

            # spacing
            elapsed = time.perf_counter() - last_time
            if last_time and elapsed < 90:
                time.sleep(90 - elapsed)

            try:
                ok, latency, failure_type = ping(adapter, model)
            except Exception as e:
                ok, latency, failure_type = 0, 0, f"outer_crash:{str(e)[:80]}"

            last_time = time.perf_counter()

            # ---- STATE HANDLING ----
            if ok:
                model_state.record_success(provider, model)

                if state == "paused":
                    model_state.unpause(provider, model)

            else:
                if is_temporary(failure_type):
                    model_state.pause(provider, model, reason=failure_type)
                else:
                    model_state.pause(provider, model, reason=failure_type)

                    fails = model_state.consecutive_failures(provider, model)
                    if fails >= _DISABLE_AFTER_FAILURES:
                        model_state.disable(provider, model, reason=f"{fails} failures")

            # ---- ALWAYS LOG ----
            entries.append({
                "ts": now,
                "provider": provider,
                "model": model,
                "ok": ok,
                "latency_ms": latency,
                "failure_type": failure_type
            })

    save(entries)
    print(f"Done. Logged {len(entries)} total entries.")


if __name__ == "__main__":
    main()
