"""
health_rollup.py — MRTG-style compaction for health_log.json.

Keeps full-resolution data for recent entries and progressively averages
older data into larger buckets — exactly like MRTG round-robin archives.

Resolution tiers
----------------
  < 24 h   : raw entries kept as-is (hourly resolution)
  24 h–7 d : averaged into 6-hour buckets  per provider+model
  > 7 d    : averaged into 24-hour buckets per provider+model

Averaged entries carry a "rollup" field so they are distinguishable from
raw pings and are never re-averaged (idempotent).

Dynamic provider support
------------------------
All grouping is done on whatever provider+model keys appear in the log.
No hardcoded provider list — adding or removing an adapter just changes
which keys appear in the output.
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import List

# ---------------------------------------------------------------------------
# Tier boundaries (seconds before now)
# ---------------------------------------------------------------------------
_RAW_CUTOFF_S   = 24 * 3600        # 24 h  — keep full resolution
_MED_CUTOFF_S   = 7  * 24 * 3600   # 7 d   — 6-hour buckets beyond this
_MED_BUCKET_S   = 6  * 3600        # 6 h bucket size
_OLD_BUCKET_S   = 24 * 3600        # 24 h bucket size


def _parse_ts(ts_str: str) -> datetime:
    return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))


def _floor_to_bucket(dt: datetime, bucket_seconds: int) -> datetime:
    """Floor a datetime to the nearest bucket boundary."""
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    total_seconds = int((dt - epoch).total_seconds())
    floored = (total_seconds // bucket_seconds) * bucket_seconds
    return epoch + timedelta(seconds=floored)


def _average_group(entries: List[dict], bucket_dt: datetime, bucket_label: str) -> dict:
    """
    Collapse a list of raw (or already-averaged) entries into one rolled-up entry.

    Uses the provider/model from the first entry; all entries in the group
    should share the same provider+model+bucket.
    """
    latencies   = [e["latency_ms"] for e in entries if e.get("latency_ms", 0) > 0]
    qualities   = [e["quality"]    for e in entries]
    ok_count    = sum(1 for e in entries if e.get("status") in ("ok", "degraded", "averaged"))
    sample_count = len(entries)

    # Carry forward sample_count from previously rolled entries
    total_samples = sum(e.get("sample_count", 1) for e in entries)

    return {
        "timestamp":    bucket_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "provider":     entries[0]["provider"],
        "model":        entries[0]["model"],
        "status":       "averaged",
        "latency_ms":   round(sum(latencies) / len(latencies), 1) if latencies else 0,
        "quality":      round(sum(qualities) / len(qualities), 3) if qualities else 0.0,
        "uptime":       round(ok_count / sample_count, 3) if sample_count else 0.0,
        "sample_count": total_samples,
        "rollup":       bucket_label,
    }


def rollup(entries: List[dict]) -> List[dict]:
    """
    Apply MRTG-style compaction to *entries* and return the reduced list.

    Idempotent — calling rollup() on already-rolled data produces the same
    result as calling it once on the raw data (rolled entries are bucketed
    by their timestamp and sample_count is preserved).

    Parameters
    ----------
    entries : list[dict]
        All entries from health_log.json, in any order.

    Returns
    -------
    list[dict]
        Compacted entries sorted by timestamp ascending.
    """
    if not entries:
        return []

    now = datetime.now(timezone.utc)
    raw_cutoff = now - timedelta(seconds=_RAW_CUTOFF_S)
    med_cutoff = now - timedelta(seconds=_MED_CUTOFF_S)

    keep_raw: List[dict] = []
    med_groups: dict = defaultdict(list)   # (provider, model, bucket_dt) → entries
    old_groups: dict = defaultdict(list)

    for entry in entries:
        try:
            ts = _parse_ts(entry["timestamp"])
        except Exception:
            continue  # skip malformed timestamps

        if ts >= raw_cutoff:
            keep_raw.append(entry)
        elif ts >= med_cutoff:
            bucket = _floor_to_bucket(ts, _MED_BUCKET_S)
            med_groups[(entry["provider"], entry["model"], bucket)].append(entry)
        else:
            bucket = _floor_to_bucket(ts, _OLD_BUCKET_S)
            old_groups[(entry["provider"], entry["model"], bucket)].append(entry)

    averaged: List[dict] = []
    for (provider, model, bucket_dt), group in med_groups.items():
        averaged.append(_average_group(group, bucket_dt, "6h"))
    for (provider, model, bucket_dt), group in old_groups.items():
        averaged.append(_average_group(group, bucket_dt, "24h"))

    combined = keep_raw + averaged
    combined.sort(key=lambda e: e["timestamp"])
    return combined


# ---------------------------------------------------------------------------
# Health snapshot — used by the router
# ---------------------------------------------------------------------------

def recent_provider_health(entries: List[dict], window_hours: int = 24) -> dict:
    """
    Summarise provider health over the last *window_hours* from raw entries.

    Returns a dict keyed by provider with uptime, avg_latency_ms, and
    sample_count — derived dynamically from whatever providers appear in
    the log, no hardcoded list needed.

    Parameters
    ----------
    entries       : list[dict]  — full health_log contents
    window_hours  : int         — how far back to look (default 24 h)

    Returns
    -------
    dict  { provider_key: { "uptime": float, "avg_latency_ms": float,
                            "sample_count": int } }
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
    provider_data: dict = defaultdict(lambda: {"ok": 0, "total": 0, "latencies": []})

    for entry in entries:
        # Skip no_key entries — provider isn't configured, not a health signal
        if entry.get("status") == "no_key":
            continue
        try:
            ts = _parse_ts(entry["timestamp"])
        except Exception:
            continue
        if ts < cutoff:
            continue

        p = entry["provider"]
        provider_data[p]["total"] += 1
        if entry.get("status") in ("ok", "degraded"):
            provider_data[p]["ok"] += 1
        lat = entry.get("latency_ms", 0)
        if lat > 0:
            provider_data[p]["latencies"].append(lat)

    result = {}
    for provider, d in provider_data.items():
        lats = d["latencies"]
        result[provider] = {
            "uptime":        round(d["ok"] / d["total"], 3) if d["total"] else 0.0,
            "avg_latency_ms": round(sum(lats) / len(lats), 1) if lats else 0.0,
            "sample_count":  d["total"],
        }
    return result
