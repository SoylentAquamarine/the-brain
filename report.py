"""
report.py — Usage report for The Brain.

Shows which AI providers were used, how much compute each consumed,
and how many Claude tokens were saved by routing to free workers.

Run at any time:
    python report.py
    python report.py --json       # machine-readable output
    python report.py --reset      # wipe stats and start fresh
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from brain.stats import tracker

# Claude Sonnet retail rate for the "saved" calculation.
_CLAUDE_COST_PER_1K = 0.003

# Tier labels for display
_TIERS = {
    "anthropic": "PAID",
    "openai":    "PAID",
    "gemini":    "FREE",
    "groq":      "FREE",
    "cohere":    "FREE",
    "mistral":   "FREE",
    "deepseek":  "FREE",
    "cerebras":  "FREE",
    "together":  "FREE",
}


def main() -> None:
    p = argparse.ArgumentParser(description="The Brain — usage report")
    p.add_argument("--json",  action="store_true", help="Output as JSON")
    p.add_argument("--reset", action="store_true", help="Reset all stats")
    args = p.parse_args()

    if args.reset:
        tracker.reset()
        print("Stats reset.")
        return

    stats = tracker.get()

    if args.json:
        print(json.dumps({
            "total_calls":         stats.total_calls,
            "total_tokens":        stats.total_tokens,
            "total_cost_usd":      round(stats.total_cost_usd, 6),
            "claude_tokens_saved": stats.claude_tokens_saved,
            "estimated_savings":   round(stats.estimated_savings_usd, 6),
            "providers": {
                k: {
                    "calls":       v.calls,
                    "tokens":      v.tokens,
                    "cost_usd":    round(v.cost_usd, 6),
                    "avg_latency": round(v.avg_latency_ms),
                    "success_pct": round(v.success_rate, 1),
                }
                for k, v in sorted(stats.providers.items(), key=lambda x: -x[1].calls)
            }
        }, indent=2))
        return

    # ── Human-readable report ────────────────────────────────────────────

    since = datetime.fromtimestamp(stats.first_call_ts).strftime("%Y-%m-%d %H:%M")
    last  = datetime.fromtimestamp(stats.last_call_ts).strftime("%Y-%m-%d %H:%M")

    print()
    print("=" * 60)
    print("           THE BRAIN  --  Usage Report")
    print("=" * 60)
    print(f"  Tracking since : {since}")
    print(f"  Last call      : {last}")
    print(f"  Total calls    : {stats.total_calls}")
    print(f"  Total tokens   : {stats.total_tokens:,}")
    print(f"  Total cost     : ${stats.total_cost_usd:.4f}")
    print()

    if not stats.providers:
        print("  No calls recorded yet.")
        print()
        return

    # Per-provider table
    header = f"  {'Provider':<13} {'Tier':<6} {'Calls':>7} {'Tokens':>12} {'Cost':>10} {'Avg ms':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    sorted_providers = sorted(stats.providers.items(), key=lambda x: -x[1].tokens)
    for key, ps in sorted_providers:
        tier     = _TIERS.get(key, "?")
        cost_str = f"${ps.cost_usd:.4f}" if ps.cost_usd else "free"
        print(
            f"  {key:<13} {tier:<6} {ps.calls:>7} {ps.tokens:>12,} "
            f"{cost_str:>10} {ps.avg_latency_ms:>8.0f}"
        )

    print()

    # Claude savings callout
    saved_tokens = stats.claude_tokens_saved
    saved_usd    = stats.estimated_savings_usd
    free_calls   = sum(v.calls for k, v in stats.providers.items() if k != "anthropic")
    total_pct    = (saved_tokens / stats.total_tokens * 100) if stats.total_tokens else 0

    print("-" * 60)
    print("  Claude Token Savings")
    print("-" * 60)
    print(f"  Calls handled by free workers  : {free_calls:,}")
    print(f"  Tokens offloaded from Claude   : {saved_tokens:,}")
    print(f"  % of total tokens saved        : {total_pct:.1f}%")
    print(f"  Estimated savings (Sonnet rate): ${saved_usd:.4f}")
    print()

    # Usage breakdown bar
    if stats.total_tokens:
        print("  Token distribution:")
        for key, ps in sorted_providers:
            pct = ps.tokens / stats.total_tokens * 100
            bar = "#" * int(pct / 2)
            tier = _TIERS.get(key, "?")
            print(f"    {key:<12} [{tier}]  {bar:<50} {pct:5.1f}%")
    print()


if __name__ == "__main__":
    main()
