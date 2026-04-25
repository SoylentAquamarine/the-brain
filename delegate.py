"""
delegate.py — Called by Claude Code to offload a task to a worker AI.

This is the bridge between Claude (orchestrator, running in the desktop app)
and the free AI workers.  Claude calls this script via a Bash tool call,
reads the printed result, and relays it to the user.

Usage
-----
    python delegate.py --provider groq     --type classification --prompt "..."
    python delegate.py --provider gemini   --type summarization  --prompt "..."
    python delegate.py --provider mistral  --type coding         --prompt "..."
    python delegate.py --provider cerebras --type factual_qa     --prompt "..."

The script prints a single-line metadata header followed by the response body.
Claude reads both and incorporates the answer into its reply.

Exit codes
----------
    0 — success
    1 — provider unavailable or API call failed
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from dotenv import load_dotenv

# Load .env BEFORE importing brain modules so adapters pick up API keys.
load_dotenv()

from brain.constants import DEFAULT_MAX_TOKENS
from brain.orchestrator import Orchestrator
from brain.task import Task, TaskType


def parse_args() -> argparse.Namespace:
    """
    Parse and return command-line arguments.

    Returns
    -------
    argparse.Namespace
    """
    p = argparse.ArgumentParser(
        description="Delegate a task to a free AI worker.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--provider", required=True,
        help="Provider key: groq | gemini | mistral | cerebras | huggingface | pollinations",
    )
    p.add_argument(
        "--prompt", required=True,
        help="The task prompt to send to the provider.",
    )
    p.add_argument(
        "--type", default="general",
        dest="task_type",
        help=f"TaskType value (default: general). Options: {[t.value for t in TaskType]}",
    )
    p.add_argument(
        "--context", default=None,
        help="Optional context block prepended to the prompt (e.g. document text).",
    )
    p.add_argument(
        "--tokens", type=int, default=DEFAULT_MAX_TOKENS,
        help=f"Maximum output tokens (default: {DEFAULT_MAX_TOKENS}).",
    )
    p.add_argument(
        "--parallel", type=int, default=0, metavar="N",
        help="Race N providers in parallel and return the first success (default: off).",
    )
    p.add_argument(
        "--no-cache", action="store_true",
        help="Bypass the prompt cache and always call the provider.",
    )
    return p.parse_args()


def resolve_task_type(raw: str) -> TaskType:
    """
    Convert a string to a TaskType enum value.

    Falls back to TaskType.GENERAL if the string is not recognised so that
    delegate.py never exits with a confusing enum error.

    Parameters
    ----------
    raw : str
        Value passed via --type flag.

    Returns
    -------
    TaskType
    """
    try:
        return TaskType(raw)
    except ValueError:
        # Unknown type — log to stderr and default gracefully.
        print(
            f"Warning: unknown task type '{raw}', defaulting to 'general'.",
            file=sys.stderr,
        )
        return TaskType.GENERAL


def main() -> int:
    """
    Entry point — parse args, call provider, print result.

    Returns
    -------
    int
        Exit code: 0 = success, 1 = failure.
    """
    args = parse_args()

    task_type = resolve_task_type(args.task_type)

    task = Task(
        prompt=args.prompt,
        task_type=task_type,
        context=args.context,
        max_tokens=args.tokens,
        preferred_model=args.provider,
    )

    orchestrator = Orchestrator(use_cache=not args.no_cache)

    if args.parallel and args.parallel > 1:
        result = orchestrator.run_parallel(task, n=args.parallel)
    else:
        result = orchestrator.run(task)

    if not result.succeeded:
        print(f"ERROR: {result.error}", file=sys.stderr)
        return 1

    def _out(text: str) -> None:
        sys.stdout.buffer.write((text + "\n").encode("utf-8", errors="replace"))

    cost_str = f"${result.cost_usd:.6f}" if result.cost_usd else "free"
    _out(
        f"[{result.provider} / {result.model} | "
        f"{result.tokens_used} tokens | "
        f"{result.latency_ms:.0f}ms | {cost_str}]"
    )
    _out(result.content)
    return 0


if __name__ == "__main__":
    sys.exit(main())
