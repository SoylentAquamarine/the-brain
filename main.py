"""
main.py — Command-line entry point for The Brain.

Run interactively:
    python main.py

Or pipe a prompt:
    echo "Summarise the Apollo 11 mission" | python main.py --type summarization

Flags:
    --type      TaskType value (classification, summarization, coding, ...)
    --priority  high | normal | low
    --provider  Force a specific provider (groq, gemini, openai, anthropic, cohere)
    --dynamic   Enable Claude-assisted routing
    --commit    Write output to outputs/<task_id>.md and git commit it
    --push      Push to GitHub after committing (requires --commit)
    --status    Print provider availability and exit
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from dotenv import load_dotenv

# Load .env before importing brain so adapters can read keys at init time.
load_dotenv()

from brain import Orchestrator, Task, TaskType, Priority
from brain.git_ops import GitOps

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("brain.main")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="The Brain — Claude-orchestrated AI delegation system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("prompt", nargs="?", help="Prompt to send (or pipe via stdin)")
    p.add_argument("--type",     default="general",  help="TaskType (default: general)")
    p.add_argument("--priority", default="normal",   help="Priority: high|normal|low")
    p.add_argument("--provider", default=None,       help="Force a specific provider")
    p.add_argument("--dynamic",  action="store_true", help="Enable dynamic routing")
    p.add_argument("--commit",   action="store_true", help="Git-commit the output")
    p.add_argument("--push",     action="store_true", help="Push to GitHub after commit")
    p.add_argument("--status",   action="store_true", help="Show provider status and exit")
    p.add_argument("--json",     action="store_true", help="Output result as JSON")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    orchestrator = Orchestrator(dynamic_routing=args.dynamic)

    # ── Status mode ──────────────────────────────────────────────────────
    if args.status:
        status = orchestrator.provider_status()
        for key, info in status.items():
            tick = "✓" if info["available"] else "✗"
            tier = info["tier"].upper()
            print(f"  {tick} {key:<12} [{tier}]  types: {', '.join(info['task_types'][:3])}...")
        return 0

    # ── Resolve prompt ────────────────────────────────────────────────────
    prompt = args.prompt
    if not prompt and not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()
    if not prompt:
        print("Error: provide a prompt as an argument or via stdin.", file=sys.stderr)
        return 1

    # ── Build task ────────────────────────────────────────────────────────
    try:
        task_type = TaskType(args.type)
    except ValueError:
        valid = [t.value for t in TaskType]
        print(f"Invalid --type '{args.type}'. Valid values: {valid}", file=sys.stderr)
        return 1

    try:
        priority = Priority(args.priority)
    except ValueError:
        print(f"Invalid --priority '{args.priority}'. Use: high, normal, low", file=sys.stderr)
        return 1

    task = Task(
        prompt=prompt,
        task_type=task_type,
        priority=priority,
        preferred_model=args.provider,
    )

    # ── Execute ───────────────────────────────────────────────────────────
    result = orchestrator.run(task)

    if not result.succeeded:
        print(f"\n[ERROR] {result.error}", file=sys.stderr)
        return 1

    # ── Output ────────────────────────────────────────────────────────────
    if args.json:
        print(json.dumps({
            "task_id":  result.task_id,
            "provider": result.provider,
            "model":    result.model,
            "content":  result.content,
            "tokens":   result.tokens_used,
            "latency_ms": result.latency_ms,
            "cost_usd": result.cost_usd,
        }, indent=2))
    else:
        print(f"\n{'─'*60}")
        print(f"Provider : {result.provider}  Model : {result.model}")
        print(f"Tokens   : {result.tokens_used}   Latency: {result.latency_ms:.0f}ms")
        cost_str = f"${result.cost_usd:.6f}" if result.cost_usd else "free"
        print(f"Cost     : {cost_str}")
        print(f"{'─'*60}\n")
        print(result.content)

    # ── Optional git commit ───────────────────────────────────────────────
    if args.commit:
        git = GitOps(repo_path=".")
        output_path = f"outputs/{result.task_id[:8]}.md"
        committed = git.write_and_commit(result, task, output_path)
        if committed:
            print(f"\n[git] Committed output → {output_path}")
            if args.push:
                pushed = git.push()
                print(f"[git] Push {'succeeded' if pushed else 'FAILED'}")

    stats = orchestrator.session_stats()
    logger.info("Session stats: %s", stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
