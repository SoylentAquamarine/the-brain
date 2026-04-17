"""
delegate.py — Called by Claude Code to offload a task to a worker AI.

Usage (Claude calls this internally):
    python delegate.py --provider groq --prompt "..." --type classification
    python delegate.py --provider gemini --prompt "..." --type summarization
    python delegate.py --provider openai --prompt "..." --type coding

Prints a clean result block that Claude reads and relays to the user.
"""

from __future__ import annotations

import argparse
import sys
from dotenv import load_dotenv
load_dotenv()

from brain.adapters import REGISTRY
from brain.task import Task, TaskType


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--provider", required=True, help="groq | gemini | openai | cohere")
    p.add_argument("--prompt",   required=True, help="The task prompt")
    p.add_argument("--type",     default="general", help="TaskType value")
    p.add_argument("--context",  default=None, help="Optional context block")
    p.add_argument("--tokens",   type=int, default=1024, help="Max output tokens")
    args = p.parse_args()

    adapter = REGISTRY.get(args.provider)
    if not adapter:
        print(f"ERROR: Unknown provider '{args.provider}'. Available: {list(REGISTRY.keys())}")
        sys.exit(1)
    if not adapter.is_available():
        print(f"ERROR: Provider '{args.provider}' has no API key configured.")
        sys.exit(1)

    try:
        task_type = TaskType(args.type)
    except ValueError:
        task_type = TaskType.GENERAL

    task = Task(
        prompt=args.prompt,
        task_type=task_type,
        context=args.context,
        max_tokens=args.tokens,
        preferred_model=args.provider,
    )

    result = adapter.complete(task)

    if result.error:
        print(f"ERROR: {result.error}")
        sys.exit(1)

    # Clean output block for Claude to read
    cost_str = f"${result.cost_usd:.6f}" if result.cost_usd else "free"
    print(f"[{result.provider} / {result.model} | {result.tokens_used} tokens | {result.latency_ms:.0f}ms | {cost_str}]")
    print(result.content)


if __name__ == "__main__":
    main()
