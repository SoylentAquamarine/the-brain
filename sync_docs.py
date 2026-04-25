#!/usr/bin/env python3
"""
sync_docs.py — Regenerate the dynamic sections of CLAUDE.md from adapter metadata.

Run this after adding, removing, or editing any provider adapter to keep
CLAUDE.md, the routing guide, and the provider table in sync with the code.

    python sync_docs.py          # update CLAUDE.md in place, print diff summary
    python sync_docs.py --check  # exit 1 if CLAUDE.md is out of date (CI use)

CLAUDE.md sections updated
--------------------------
<!-- SYNC:providers:start --> … <!-- SYNC:providers:end -->
    Full provider table: key, default model, speed, quality, description.

<!-- SYNC:routing:start --> … <!-- SYNC:routing:end -->
    Routing guide: best provider per task type with one-line rationale.

Everything outside the markers is left completely untouched.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from brain.adapters import REGISTRY
from brain.router import build_routing_table
from brain.task import TaskType

CLAUDE_MD = Path(__file__).parent / "CLAUDE.md"

_SPEED_LABEL = {
    "ultra_fast": "ultra-fast",
    "fast":       "fast",
    "standard":   "standard",
    "slow":       "slow",
}

# Maps task type → short human label for the routing guide table
_TASK_LABEL: dict = {
    TaskType.CLASSIFICATION: "Classify / label / yes-no",
    TaskType.FACTUAL_QA:     "Quick factual Q&A",
    TaskType.SUMMARIZATION:  "Summarise long text",
    TaskType.EXTRACTION:     "Extract structured data",
    TaskType.TRANSLATION:    "Translation",
    TaskType.CODING:         "Code generation",
    TaskType.REASONING:      "Reasoning / analysis",
    TaskType.CREATIVE:       "Creative writing / drafting",
    TaskType.GENERAL:        "General / explanation",
}


# ---------------------------------------------------------------------------
# Section generators
# ---------------------------------------------------------------------------

def _providers_section() -> str:
    """Generate the ## Available providers table."""
    lines = [
        "## Available providers\n",
        "| Key | Default model | Speed | Quality | Description |",
        "|---|---|---|---|---|",
    ]
    # Sort: free first, then quality desc, then name
    ordered = sorted(
        REGISTRY.items(),
        key=lambda kv: (0 if kv[1].TIER == "free" else 1, -kv[1].QUALITY_SCORE, kv[0]),
    )
    for key, adapter in ordered:
        default = adapter.DEFAULT_MODEL if hasattr(adapter, "DEFAULT_MODEL") else "n/a"
        speed   = _SPEED_LABEL.get(adapter.SPEED_TIER, adapter.SPEED_TIER)
        quality = f"{adapter.QUALITY_SCORE}/10"
        desc    = adapter.DESCRIPTION or "—"
        lines.append(f"| `{key}` | {default} | {speed} | {quality} | {desc} |")
    return "\n".join(lines) + "\n"


def _routing_section() -> str:
    """Generate the ## Routing guide table."""
    routing_table = build_routing_table(REGISTRY)

    lines = [
        "## Routing guide — which worker for which task\n",
        "| Task | Best provider | Why |",
        "|---|---|---|",
    ]

    for task_type, label in _TASK_LABEL.items():
        chain = routing_table.get(task_type, [])
        # First available-registered key in chain
        best = next((k for k in chain if k in REGISTRY), None)
        if not best:
            continue
        adapter = REGISTRY[best]
        why = adapter.DESCRIPTION.split(" — ")[-1] if " — " in adapter.DESCRIPTION else adapter.DESCRIPTION
        lines.append(f"| {label} | `{best}` | {why} |")

    # Static rows not derived from task types
    lines += [
        "| Image generation | `pollinations` | Keyless, no sign-up, FLUX model |",
        "| Anything conversational | **You (Claude)** | No offload needed |",
        "| Planning which provider to use | **You (Claude)** | Orchestration is your job |",
    ]

    lines.append("")
    lines.append("**Full fallback chains** (first available wins):\n")
    for task_type, label in _TASK_LABEL.items():
        chain = routing_table.get(task_type, [])
        registered = [k for k in chain if k in REGISTRY]
        if registered:
            lines.append(f"- `{task_type.value}`: {' → '.join(f'`{k}`' for k in registered)}")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLAUDE.md update
# ---------------------------------------------------------------------------

def _replace_section(text: str, tag: str, new_content: str) -> tuple[str, bool]:
    """Replace content between <!-- SYNC:{tag}:start --> and <!-- SYNC:{tag}:end -->."""
    start_marker = f"<!-- SYNC:{tag}:start -->"
    end_marker   = f"<!-- SYNC:{tag}:end -->"

    start_idx = text.find(start_marker)
    end_idx   = text.find(end_marker)

    if start_idx == -1 or end_idx == -1:
        # Markers not present — append section at end of file
        appended = text.rstrip() + f"\n\n{start_marker}\n{new_content}\n{end_marker}\n"
        return appended, True

    inner_start = start_idx + len(start_marker)
    old_inner   = text[inner_start:end_idx]
    new_inner   = f"\n{new_content}\n"
    changed     = old_inner != new_inner
    new_text    = text[:inner_start] + new_inner + text[end_idx:]
    return new_text, changed


def sync(check_only: bool = False) -> int:
    """
    Regenerate dynamic sections in CLAUDE.md.

    Returns 0 if up-to-date (or successfully updated), 1 if --check and
    out-of-date.
    """
    if not CLAUDE_MD.exists():
        print(f"ERROR: {CLAUDE_MD} not found.", file=sys.stderr)
        return 1

    original = CLAUDE_MD.read_text(encoding="utf-8")
    text = original

    providers_content = _providers_section()
    routing_content   = _routing_section()

    text, providers_changed = _replace_section(text, "providers", providers_content)
    text, routing_changed   = _replace_section(text, "routing",   routing_content)

    any_changed = providers_changed or routing_changed

    if check_only:
        if any_changed:
            print("CLAUDE.md is OUT OF DATE — run: python sync_docs.py")
            return 1
        print("CLAUDE.md is up to date.")
        return 0

    if any_changed:
        CLAUDE_MD.write_text(text, encoding="utf-8")
        changes = []
        if providers_changed:
            changes.append("providers table")
        if routing_changed:
            changes.append("routing guide")
        print(f"CLAUDE.md updated: {', '.join(changes)}")
        print(f"  {len(REGISTRY)} providers, {sum(len(a.list_models()) for a in REGISTRY.values())} model slots")
    else:
        print("CLAUDE.md already up to date — no changes written.")

    return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true", help="Exit 1 if out of date (CI use)")
    args = p.parse_args()
    sys.exit(sync(check_only=args.check))


if __name__ == "__main__":
    main()
