#!/usr/bin/env python3
"""
status.py — Live dashboard for The Brain.

Shows:
  - Which AI providers are reachable right now
  - Usage stats per provider (calls, tokens, cost, avg latency)
  - Total Claude tokens saved
  - Nightly report schedule

Run at any time:
    python status.py
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from brain.adapters import REGISTRY
from brain.stats import tracker

# ANSI colours — degrade gracefully on terminals that don't support them
_GREEN  = "\033[32m"
_RED    = "\033[31m"
_YELLOW = "\033[33m"
_CYAN   = "\033[36m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"

def _tick(ok: bool) -> str:
    return f"{_GREEN}✓ online {_RESET}" if ok else f"{_RED}✗ offline{_RESET}"


def main() -> None:
    print()
    print(f"{_BOLD}{'=' * 62}{_RESET}")
    print(f"{_BOLD}           THE BRAIN  —  Status Dashboard{_RESET}")
    print(f"{_BOLD}{'=' * 62}{_RESET}")
    print(f"  {_CYAN}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{_RESET}")
    print()

    # ── Provider connectivity ─────────────────────────────────────────────
    print(f"  {_BOLD}Provider Connectivity{_RESET}")
    print(f"  {'─' * 58}")
    print(f"  {'Provider':<14} {'Status':<18} {'Tier':<6}  Best for")
    print(f"  {'─' * 58}")

    best_for = {
        "cerebras":    "classification, scoring, yes/no",
        "groq":        "factual Q&A, general tasks",
        "gemini":      "summarisation, long text, translation",
        "mistral":     "coding, creative writing, extraction",
        "huggingface": "fallback",
        "pollinations":"image generation (no key needed)",
    }

    available_count = 0
    for key, adapter in REGISTRY.items():
        ok = adapter.is_available()
        if ok:
            available_count += 1
        status = _tick(ok)
        tier   = adapter.TIER.upper()
        tip    = best_for.get(key, "")
        print(f"  {key:<14} {status:<27} {tier:<6}  {tip}")

    print(f"  {'─' * 58}")
    print(f"  {available_count}/{len(REGISTRY)} providers online")
    print()

    # ── Usage stats ───────────────────────────────────────────────────────
    stats = tracker.get()

    if stats.total_calls == 0:
        print(f"  {_YELLOW}No usage recorded yet.{_RESET}")
        print()
    else:
        since = datetime.fromtimestamp(stats.first_call_ts).strftime("%Y-%m-%d %H:%M")
        last  = datetime.fromtimestamp(stats.last_call_ts).strftime("%Y-%m-%d %H:%M")

        print(f"  {_BOLD}Usage Stats{_RESET}  (since {since})")
        print(f"  {'─' * 58}")
        print(f"  {'Provider':<14} {'Calls':>6}  {'Tokens':>10}  {'Avg ms':>7}  {'Success':>7}  Cost")
        print(f"  {'─' * 58}")

        sorted_providers = sorted(
            stats.providers.items(), key=lambda x: -x[1].calls
        )
        for key, ps in sorted_providers:
            cost_str = f"${ps.cost_usd:.4f}" if ps.cost_usd else "free"
            print(
                f"  {key:<14} {ps.calls:>6}  {ps.tokens:>10,}  "
                f"{ps.avg_latency_ms:>7.0f}  {ps.success_rate:>6.1f}%  {cost_str}"
            )

        print(f"  {'─' * 58}")
        print(f"  {'TOTAL':<14} {stats.total_calls:>6}  {stats.total_tokens:>10,}")
        print()

        # Claude savings callout
        saved   = stats.claude_tokens_saved
        savings = stats.estimated_savings_usd
        pct     = (saved / stats.total_tokens * 100) if stats.total_tokens else 0
        print(f"  {_BOLD}Claude Token Savings{_RESET}")
        print(f"  {'─' * 58}")
        print(f"  Tokens offloaded from Claude : {saved:,}")
        print(f"  % of total tokens saved      : {pct:.1f}%")
        print(f"  Estimated savings (Sonnet)   : ${savings:.4f}")
        print(f"  Last call                    : {last}")
        print()

    # ── Stats file info ───────────────────────────────────────────────────
    stats_file = Path(__file__).parent / "stats" / "usage.json"
    if stats_file.exists():
        size = stats_file.stat().st_size
        mtime = datetime.fromtimestamp(stats_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        print(f"  {_BOLD}Stats File{_RESET}")
        print(f"  {'─' * 58}")
        print(f"  Path     : {stats_file}")
        print(f"  Size     : {size:,} bytes")
        print(f"  Modified : {mtime}")
        print(f"  {_YELLOW}Remember to push stats/usage.json after each session{_RESET}")
        print(f"  so the nightly GitHub Action picks them up (runs midnight UTC).")
        print()

    print(f"{'=' * 62}")
    print()


if __name__ == "__main__":
    main()
