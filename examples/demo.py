"""
examples/demo.py — End-to-end demonstration of The Brain.

Runs five tasks that exercise different routing paths:
  1. Classification  → Groq (free, fast)
  2. Summarisation   → Gemini (free, large context)
  3. Extraction      → Cohere (free, structured)
  4. Coding          → OpenAI (paid, best for code)
  5. Reasoning       → Claude (paid, best for logic)

Each result is git-committed so you can inspect the full audit trail with:
    git log --oneline

Run:
    python examples/demo.py
"""

from __future__ import annotations

import os
import sys

# Ensure the project root is on the path when running the example directly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from brain import Orchestrator, Task, TaskType, Priority
from brain.git_ops import GitOps

SAMPLE_ARTICLE = """
The James Webb Space Telescope (JWST) has delivered its first full-colour images,
revealing distant galaxies with unprecedented clarity.  Launched on 25 December 2021,
the telescope operates at the second Lagrange point (L2), roughly 1.5 million km from Earth.
Its 6.5-metre gold-plated mirror collects infrared light, allowing astronomers to observe
objects formed just a few hundred million years after the Big Bang.
Among the early targets: the Carina Nebula, Stephan's Quintet, and the atmospheric
composition of exoplanet WASP-96b.  The data has already prompted revisions to models
of galaxy formation and early cosmic structure.
"""

SAMPLE_RESUME_SNIPPET = """
Jane Smith — Senior Software Engineer
Skills: Python, Kubernetes, AWS, PostgreSQL, GraphQL
Experience: 8 years at tech companies
Certifications: AWS Solutions Architect, CKA
"""


def run_demo() -> None:
    orchestrator = Orchestrator(dynamic_routing=False)
    git = GitOps(repo_path=os.path.join(os.path.dirname(__file__), ".."))

    print("\n" + "=" * 65)
    print("  THE BRAIN — Live Demo")
    print("  Claude orchestrates; free AIs do the heavy lifting")
    print("=" * 65 + "\n")

    print("Available providers:")
    for key, info in orchestrator.provider_status().items():
        tick = "✓" if info["available"] else "✗"
        print(f"  {tick} {key:<12} [{info['tier'].upper()}]")
    print()

    # ------------------------------------------------------------------ 1
    _run_task(
        orchestrator, git,
        label="1 / 5 — Classification (→ Groq, free)",
        task=Task(
            prompt="Classify the sentiment of this review: 'The telescope images were absolutely breathtaking, truly a historic moment for humanity.'",
            task_type=TaskType.CLASSIFICATION,
            priority=Priority.LOW,   # LOW → prefer free models
        ),
        output_file="outputs/demo_classification.md",
    )

    # ------------------------------------------------------------------ 2
    _run_task(
        orchestrator, git,
        label="2 / 5 — Summarisation (→ Gemini, free)",
        task=Task(
            prompt="Summarise the following article in two sentences.",
            task_type=TaskType.SUMMARIZATION,
            context=SAMPLE_ARTICLE,
        ),
        output_file="outputs/demo_summary.md",
    )

    # ------------------------------------------------------------------ 3
    _run_task(
        orchestrator, git,
        label="3 / 5 — Extraction (→ Cohere, free)",
        task=Task(
            prompt="Extract: name, years_of_experience, top_3_skills as JSON.",
            task_type=TaskType.EXTRACTION,
            context=SAMPLE_RESUME_SNIPPET,
        ),
        output_file="outputs/demo_extraction.md",
    )

    # ------------------------------------------------------------------ 4
    _run_task(
        orchestrator, git,
        label="4 / 5 — Coding (→ OpenAI, paid)",
        task=Task(
            prompt="Write a Python function that retries a callable up to N times with exponential back-off.  Include a docstring and type hints.",
            task_type=TaskType.CODING,
            priority=Priority.HIGH,
        ),
        output_file="outputs/demo_code.md",
    )

    # ------------------------------------------------------------------ 5
    _run_task(
        orchestrator, git,
        label="5 / 5 — Reasoning (→ Claude, paid)",
        task=Task(
            prompt="A farmer has 17 sheep.  All but 9 die.  How many are left?  Explain your reasoning step by step.",
            task_type=TaskType.REASONING,
        ),
        output_file="outputs/demo_reasoning.md",
    )

    # ── Final stats ────────────────────────────────────────────────────
    stats = orchestrator.session_stats()
    print("\n" + "=" * 65)
    print("  Session Summary")
    print("=" * 65)
    print(f"  Total calls  : {stats['total_calls']}")
    print(f"  Total tokens : {stats['total_tokens']}")
    cost = stats['estimated_cost_usd']
    print(f"  Estimated $  : ${cost:.6f}")
    print(f"  Failed calls : {stats['failed_calls']}")
    print("\n  Git audit trail:")
    print(git.log(n=6))
    print()


def _run_task(
    orchestrator: Orchestrator,
    git: GitOps,
    label: str,
    task: Task,
    output_file: str,
) -> None:
    """Run one task, print its result, and commit to git."""
    print(f"── {label}")
    result = orchestrator.run(task)

    if result.succeeded:
        cost_str = f"${result.cost_usd:.6f}" if result.cost_usd else "free"
        print(f"   Provider: {result.provider}  Model: {result.model}")
        print(f"   Tokens: {result.tokens_used}  Cost: {cost_str}  Latency: {result.latency_ms:.0f}ms")
        preview = result.content[:200].replace("\n", " ")
        print(f"   Output: {preview}...")
        git.write_and_commit(result, task, output_file)
        print(f"   Committed → {output_file}")
    else:
        print(f"   FAILED: {result.error}")
    print()


if __name__ == "__main__":
    run_demo()
