from datetime import datetime, timezone, timedelta


def recent_provider_health(entries: list, window_hours: int = 24) -> dict:
    """
    Aggregate health_log.json entries from the last window_hours into a
    per-provider summary.

    Accepts the append-only list format written by health_check.py:
        [{"ts": "...", "provider": "...", "model": "...", "ok": 1, "latency_ms": 330}, ...]

    Also handles the old averaged/rollup format for backward compatibility.

    Returns:
        {provider: {"uptime": float, "avg_latency_ms": float, "sample_count": int}}
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
    out: dict = {}

    for e in entries:
        # Support both field name conventions
        ts_raw = e.get("ts") or e.get("timestamp", "")
        try:
            ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
        except Exception:
            continue
        if ts < cutoff:
            continue

        provider = e.get("provider", "unknown")
        if provider not in out:
            out[provider] = {"ok": 0.0, "total": 0, "latencies": []}

        latency = e.get("latency_ms", e.get("latency", 0))

        if "ok" in e:
            # New format: ok=1/0 per individual probe
            out[provider]["ok"]    += e["ok"]
            out[provider]["total"] += 1
        else:
            # Old averaged format: uptime=float, sample_count=int
            sample_count = e.get("sample_count", 1)
            uptime       = e.get("uptime", 1.0)
            out[provider]["ok"]    += uptime * sample_count
            out[provider]["total"] += sample_count

        if latency:
            out[provider]["latencies"].append(latency)

    final = {}
    for p, d in out.items():
        if d["total"] == 0:
            continue
        final[p] = {
            "uptime":         round(d["ok"] / d["total"], 3),
            "avg_latency_ms": round(sum(d["latencies"]) / len(d["latencies"]), 1) if d["latencies"] else 0,
            "sample_count":   d["total"],
        }

    return final
