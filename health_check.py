#!/usr/bin/env python3
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from brain.adapters import REGISTRY
from brain.task import Task, TaskType

LOG_FILE = Path(__file__).parent / "stats" / "health_log.json"
PING_PROMPT = "What is 2+2? Reply with only the number."


def load():
    if LOG_FILE.exists():
        return json.loads(LOG_FILE.read_text())
    return {}


def save(data):
    LOG_FILE.parent.mkdir(exist_ok=True)
    LOG_FILE.write_text(json.dumps(data, indent=2))


def key(provider, model):
    return f"{provider}:{model}"


def ping(adapter, model):
    task = Task(PING_PROMPT, TaskType.FACTUAL_QA, max_tokens=10)

    old = adapter._get_active_model()
    adapter._set_active_model(model)

    try:
        start = time.perf_counter()
        res = adapter.complete(task)
        latency = int((time.perf_counter() - start) * 1000)
    finally:
        adapter._set_active_model(old)

    ok = res.succeeded and "4" in (res.content or "")
    return ok, latency


def main():
    data = load()
    now = datetime.now(timezone.utc).isoformat()

    for provider, adapter in REGISTRY.items():
        if not adapter.is_available():
            continue

        for model in adapter.list_models():
            k = key(provider, model)
            ok, latency = ping(adapter, model)

            bucket = data.setdefault(k, [])
            bucket.append({
                "ts": now,
                "ok": 1 if ok else 0,
                "latency": latency
            })

            data[k] = bucket[-24:]  # keep last 24 checks only

    save(data)


if __name__ == "__main__":
    main()