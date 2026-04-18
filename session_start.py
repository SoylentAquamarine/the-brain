#!/usr/bin/env python3
"""
session_start.py — Session briefing for Claude Code.

Reads stats/health_log.json and prints a concise heads-up on how each
AI provider has been performing in the last 24 hours.

Claude Code runs this automatically at the start of every session
(as instructed in CLAUDE.md) so routing decisions are informed by
recent provider health.

Run manually:   python session_start.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

HEALTH_LOG = Path(__file__).parent / "stats" / "health_log.json"
HOURS = 24


def load_recent(hours: int = HOURS) -> list:
    if not HEALTH_LOG.exists():
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    entries = json.loads(HEALTH_LOG.read_text(encoding="utf-8"))
    result = []
    for e in entries:
        try:
            ts = datetime.fromisoformat(e["timestamp"].replace("Z", "+00:00"))
            if ts >= cutoff:
                result.append(e)
        except Exception:
            pass
    return result


def summarise(entries: list) -> dict:
    """Return {provider: {avg_latency, uptime_pct, last_status, checks}} """
    by_provider = defaultdict(list)
    for e in entries:
        by_provider[e["provider"]].append(e)

    summary = {}
    for provider, checks in by_provider.items():
        valid_latencies = [c["latency_ms"] for c in checks if c["quality"] > 0 and c["latency_ms"] > 0]
        ok_count = sum(1 for c in checks if c["quality"] >= 0.8)
        last = checks[-1]
        summary[provider] = {
            "checks":      len(checks),
            "uptime_pct":  round(ok_count / len(checks) * 100) if checks else 0,
            "avg_latency": round(sum(valid_latencies) / len(valid_latencies)) if valid_latencies else 0,
            "last_status": last["status"],
            "last_latency": last["latency_ms"],
        }
    return summary


def health_icon(uptime: int, last_status: str) -> str:
    if last_status in ("error", "no_key"):
        return "[DOWN]"
    if uptime >= 95:
        return "[ OK ]"
    if uptime >= 75:
        return "[WARN]"
    return "[POOR]"


def main() -> None:
    entries = load_recent(HOURS)

    print()
    print("=" * 60)
    print("   THE BRAIN  -  Session Briefing")
    print("=" * 60)
    print(f"   Provider health over the last {HOURS} hours")
    print()

    if not entries:
        print("   No health data yet.")
        print("   Run: python health_check.py")
        print("   Or trigger the GitHub Action manually.")
        print()
        return

    summary = summarise(entries)

    # Sort: healthy first, then by uptime desc
    order = sorted(summary.items(), key=lambda x: (-x[1]["uptime_pct"], x[0]))

    print(f"   {'Provider':<14} {'Status':<8} {'Uptime':>7} {'Avg ms':>8} {'Checks':>7}")
    print(f"   {'-' * 50}")

    warnings = []
    for provider, s in order:
        icon = health_icon(s["uptime_pct"], s["last_status"])
        avg  = f"{s['avg_latency']}ms" if s["avg_latency"] else "—"
        print(
            f"   {provider:<14} {icon:<8} {s['uptime_pct']:>6}%"
            f"  {avg:>7}  {s['checks']:>6}"
        )
        if s["uptime_pct"] < 75 or s["last_status"] == "error":
            warnings.append(f"{provider} ({s['uptime_pct']}% uptime, last: {s['last_status']})")

    print()

    if warnings:
        print("   ROUTING ADVISORY:")
        for w in warnings:
            print(f"   - Avoid or use as fallback: {w}")
        print()
    else:
        print("   All providers healthy — no routing changes needed.")
        print()

    # Timestamp of last check
    last_ts = entries[-1].get("timestamp", "unknown")
    print(f"   Last check: {last_ts}")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
