"""
brain/model_state.py — Persistent pause/disable state for individual provider models.

States
------
active   : Normal — included in health checks and routing.
paused   : Temporary issue (timeout, rate-limit). Skipped by orchestrator but
           re-tested every health cycle. Auto-unpaused on a passing probe.
disabled : Hard failure confirmed. Never tested or routed again unless manually
           re-enabled by editing stats/model_state.json.

Backed by stats/model_state.json — survives restarts.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_STATE_FILE = Path(__file__).parent.parent / "stats" / "model_state.json"


def _key(provider: str, model: str) -> str:
    return f"{provider}::{model}"


def _load() -> dict:
    if not _STATE_FILE.exists():
        return {}
    try:
        return json.loads(_STATE_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not load model_state.json: %s", exc)
        return {}


def _save(data: dict) -> None:
    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _STATE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_state(provider: str, model: str) -> str:
    """Return 'active', 'paused', or 'disabled'."""
    data = _load()
    return data.get(_key(provider, model), {}).get("state", "active")


def is_active(provider: str, model: str) -> bool:
    return get_state(provider, model) == "active"


def is_paused(provider: str, model: str) -> bool:
    return get_state(provider, model) == "paused"


def pause(provider: str, model: str, reason: str = "") -> None:
    data = _load()
    k = _key(provider, model)
    entry = data.get(k, {})
    entry["state"]  = "paused"
    entry["reason"] = reason
    entry["since"]  = _now()
    entry.setdefault("consecutive_failures", 0)
    entry["consecutive_failures"] += 1
    data[k] = entry
    _save(data)
    logger.info("Paused %s/%s: %s", provider, model, reason)


def disable(provider: str, model: str, reason: str = "") -> None:
    data = _load()
    k = _key(provider, model)
    entry = data.get(k, {})
    entry["state"]  = "disabled"
    entry["reason"] = reason
    entry["since"]  = _now()
    data[k] = entry
    _save(data)
    logger.warning("Disabled %s/%s: %s", provider, model, reason)


def unpause(provider: str, model: str) -> None:
    """Mark a paused model active again after a passing probe."""
    data = _load()
    k = _key(provider, model)
    if data.get(k, {}).get("state") != "paused":
        return
    entry = data[k]
    entry["state"] = "active"
    entry["consecutive_failures"] = 0
    entry["recovered_at"] = _now()
    data[k] = entry
    _save(data)
    logger.info("Unpaused %s/%s — probe passing again", provider, model)


def consecutive_failures(provider: str, model: str) -> int:
    data = _load()
    return data.get(_key(provider, model), {}).get("consecutive_failures", 0)


def record_success(provider: str, model: str) -> None:
    """Reset failure counter on a successful probe (even if already active)."""
    data = _load()
    k = _key(provider, model)
    if k not in data:
        return
    data[k]["consecutive_failures"] = 0
    _save(data)
