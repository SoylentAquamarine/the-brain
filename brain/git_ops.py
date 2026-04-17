"""
git_ops.py — Git integration layer for The Brain.

Every AI-generated output can be written to disk and committed with a
structured message that records WHICH model produced it, WHY (task type),
and WHAT tokens were spent.  This turns Git into a full audit trail:
  - reproducible: replay any commit to see the exact AI output
  - auditable:    `git log --oneline` shows the complete AI action history
  - reversible:   `git revert` undoes any AI change cleanly
  - triggerable:  push to GitHub → Actions CI/CD pipeline runs automatically

Usage
-----
    from brain.git_ops import GitOps
    from brain.task import Task, TaskType

    git = GitOps(repo_path=".")
    git.write_and_commit(
        result=task_result,
        task=task,
        output_path="outputs/summary.md",
    )
"""

from __future__ import annotations

import logging
import os
import subprocess
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from brain.task import Task, TaskResult

logger = logging.getLogger(__name__)


class GitOps:
    """
    Thin wrapper around subprocess git calls.

    Why subprocess and not a library like GitPython?
      - Zero extra dependency for a core feature.
      - Git CLI output is already human-readable in logs.
      - Full access to every git flag without an abstraction layer.
    """

    def __init__(self, repo_path: str = ".") -> None:
        self._repo = Path(repo_path).resolve()
        self._git_available = self._check_git()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write_and_commit(
        self,
        result: TaskResult,
        task: Task,
        output_path: str,
        extra_message: str = "",
    ) -> bool:
        """
        Write *result.content* to *output_path* then `git add` + `git commit`.

        The commit message is structured so `git log` reads like a ledger:

            [brain] groq/llama3-8b-8192 — classification (42 tokens, 0ms)

            Task ID : 3f8a1b2c-...
            Prompt  : Classify the sentiment of...
            Provider: groq
            Model   : llama3-8b-8192
            Tokens  : 42
            Latency : 318ms
            Cost    : free

        Parameters
        ----------
        result       : The TaskResult to persist.
        task         : The originating Task (for metadata in the commit).
        output_path  : Relative path from repo root where content is written.
        extra_message: Optional extra paragraph appended to the commit body.

        Returns True on success, False if git is unavailable or commit failed.
        """
        if not self._git_available:
            logger.warning("Git not found — output written but not committed.")
            self._write_file(output_path, result.content)
            return False

        self._write_file(output_path, result.content)
        self._git("add", output_path)

        commit_msg = self._build_commit_message(result, task, extra_message)
        success = self._git("commit", "-m", commit_msg)

        if success:
            logger.info("Committed AI output to %s", output_path)
        else:
            logger.warning("git commit failed for %s", output_path)
        return success

    def push(self, remote: str = "origin", branch: str = "main") -> bool:
        """
        Push committed changes to GitHub.

        After pushing, any GitHub Actions workflows watching this branch
        will trigger automatically — completing the full pipeline.
        """
        return self._git("push", remote, branch)

    def status(self) -> str:
        """Return `git status --short` output as a string."""
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=self._repo,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def log(self, n: int = 10) -> str:
        """Return the last *n* commit one-liners."""
        result = subprocess.run(
            ["git", "log", f"--max-count={n}", "--oneline"],
            cwd=self._repo,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def is_repo(self) -> bool:
        """Return True if repo_path is inside a git repository."""
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=self._repo,
            capture_output=True,
            text=True,
        )
        return result.returncode == 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_file(self, rel_path: str, content: str) -> None:
        """Write content to a file, creating parent directories as needed."""
        full_path = self._repo / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")
        logger.debug("Wrote %d chars to %s", len(content), full_path)

    def _git(self, *args: str) -> bool:
        """Run a git command in the repo directory. Returns True on success."""
        cmd = ["git"] + list(args)
        result = subprocess.run(cmd, cwd=self._repo, capture_output=True, text=True)
        if result.returncode != 0:
            logger.debug("git %s stderr: %s", " ".join(args), result.stderr.strip())
        return result.returncode == 0

    def _check_git(self) -> bool:
        """Return True if git is installed and on PATH."""
        try:
            subprocess.run(["git", "--version"], capture_output=True, check=True)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            return False

    @staticmethod
    def _build_commit_message(
        result: TaskResult,
        task: Task,
        extra: str = "",
    ) -> str:
        """Build a structured, human-readable git commit message."""
        subject = (
            f"[brain] {result.provider}/{result.model} — "
            f"{task.task_type.value} "
            f"({result.tokens_used} tokens, {result.latency_ms:.0f}ms)"
        )
        cost_str = f"${result.cost_usd:.6f}" if result.cost_usd else "free"
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        prompt_snippet = textwrap.shorten(task.prompt, width=120, placeholder="…")

        body = textwrap.dedent(f"""
            Task ID  : {task.id}
            Time     : {timestamp}
            Prompt   : {prompt_snippet}
            Provider : {result.provider}
            Model    : {result.model}
            Tokens   : {result.tokens_used}
            Latency  : {result.latency_ms:.0f}ms
            Cost     : {cost_str}
        """).strip()

        if extra:
            body += f"\n\n{extra}"

        return f"{subject}\n\n{body}"
