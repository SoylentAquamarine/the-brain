"""
workflows/jobseeking.py — AI-powered job seeking workflow for The Brain.

Every heavy step is offloaded to the best free AI worker available:

  Step 1 — Extract requirements  →  Mistral   (best free extraction model)
  Step 2 — Score fit             →  Cerebras  (fastest classification)
  Step 3 — Summarise company     →  Gemini    (large context, summarisation)
  Step 4 — Draft cover letter    →  Mistral   (best free creative model)
  Step 5 — Tailor resume bullets →  Mistral   (instruction following)
  Step 6 — Generate interview Q  →  Groq      (fast factual generation)
  Step 7 — Commit all outputs    →  Git       (full audit trail)

Usage
-----
    # Full workflow — paste a job posting when prompted
    python workflows/jobseeking.py

    # Skip straight to a specific step
    python workflows/jobseeking.py --step cover_letter

    # Provide job posting as a file
    python workflows/jobseeking.py --job-file posting.txt

    # Provide your resume as a file for better tailoring
    python workflows/jobseeking.py --resume-file resume.txt

All outputs are saved to outputs/jobseeking/<company>/ and git-committed
so you have a searchable history of every application.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from brain import Orchestrator, Task, TaskType, Priority
from brain.git_ops import GitOps

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "WARNING"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ---------------------------------------------------------------------------
# Resume placeholder — paste your real resume here or pass --resume-file
# ---------------------------------------------------------------------------

DEFAULT_RESUME = """
[Your resume content here — or pass --resume-file path/to/resume.txt]
"""


# ---------------------------------------------------------------------------
# Workflow steps
# ---------------------------------------------------------------------------

def step_extract_requirements(orch: Orchestrator, job_posting: str) -> dict:
    """
    Step 1 — Extract structured requirements from the job posting.

    Offloads to Mistral (best free extraction model).
    Returns a dict with keys: title, company, required_skills,
    nice_to_have, responsibilities, experience_years.

    Parameters
    ----------
    orch        : Orchestrator
    job_posting : Raw job posting text

    Returns
    -------
    dict
    """
    print("\n[1/6] Extracting requirements from job posting...")

    result = orch.run(Task(
        prompt="""Extract the following from this job posting and return as plain text with clear labels:

TITLE: (job title)
COMPANY: (company name)
REQUIRED_SKILLS: (comma-separated list)
NICE_TO_HAVE: (comma-separated list)
EXPERIENCE_YEARS: (number or range)
KEY_RESPONSIBILITIES: (3-5 bullet points)
CULTURE_KEYWORDS: (values or culture words mentioned)""",
        task_type=TaskType.EXTRACTION,
        context=job_posting,
        max_tokens=600,
        preferred_model="mistral",
    ))

    if not result.succeeded:
        print(f"  Warning: extraction failed ({result.error}), continuing with raw posting")
        return {"raw": job_posting, "title": "unknown", "company": "unknown"}

    print(f"  Done [{result.provider} / {result.model} — {result.tokens_used} tokens, free]")

    # Parse the labeled output into a dict for use in later steps.
    # Keys are normalised to lowercase with underscores so callers can use
    # parsed["required_skills"] regardless of how the model capitalised the label.
    parsed: dict = {"raw_extraction": result.content}
    for line in result.content.splitlines():
        line = line.strip()
        if ":" not in line or line.startswith("#") or line.startswith("-"):
            continue
        key, _, value = line.partition(":")
        # Normalise: "Required Skills" → "required_skills"
        norm_key = key.strip().lower().replace(" ", "_").strip("*_")
        if norm_key and value.strip():
            parsed[norm_key] = value.strip()

    return parsed


def step_score_fit(orch: Orchestrator, requirements: dict, resume: str) -> str:
    """
    Step 2 — Score how well the resume matches the job requirements.

    Offloads to Cerebras (fastest classification, free).
    Returns a short fit-score summary string.

    Parameters
    ----------
    orch         : Orchestrator
    requirements : Output from step_extract_requirements
    resume       : Resume text

    Returns
    -------
    str — fit score and gap analysis
    """
    print("\n[2/6] Scoring resume fit...")

    req_summary = requirements.get("raw_extraction", str(requirements))

    result = orch.run(Task(
        prompt=f"""Score this resume's fit for the job requirements below.
Reply with:
SCORE: X/10
STRENGTHS: (2-3 matching strengths)
GAPS: (2-3 missing skills or experience)
RECOMMENDATION: (Apply / Apply with caveats / Skip)

JOB REQUIREMENTS:
{req_summary[:800]}

RESUME:
{resume[:1200]}""",
        task_type=TaskType.CLASSIFICATION,
        max_tokens=300,
        preferred_model="cerebras",
        priority=Priority.LOW,
    ))

    if not result.succeeded:
        return "Fit scoring unavailable."

    print(f"  Done [{result.provider} / {result.model} — {result.tokens_used} tokens, free]")
    return result.content


def step_summarise_company(orch: Orchestrator, company_name: str, job_context: str) -> str:
    """
    Step 3 — Summarise the company from context clues in the job posting.

    Offloads to Gemini (best free summarisation model).
    Returns a 3-5 sentence company summary.

    Parameters
    ----------
    orch         : Orchestrator
    company_name : Extracted company name
    job_context  : Full job posting (contains company description clues)

    Returns
    -------
    str — company summary
    """
    print(f"\n[3/6] Summarising {company_name}...")

    result = orch.run(Task(
        prompt=f"""Based only on the job posting below, write a 3-5 sentence summary of {company_name}:
what they do, their apparent culture, and what they seem to value in employees.
Focus on facts you can infer from the posting — do not invent anything.""",
        task_type=TaskType.SUMMARIZATION,
        context=job_context,
        max_tokens=250,
        preferred_model="gemini",
    ))

    if not result.succeeded:
        return f"Company summary for {company_name} unavailable."

    print(f"  Done [{result.provider} / {result.model} — {result.tokens_used} tokens, free]")
    return result.content


def step_draft_cover_letter(
    orch: Orchestrator,
    requirements: dict,
    resume: str,
    company_summary: str,
) -> str:
    """
    Step 4 — Draft a tailored cover letter.

    Offloads to Mistral (best free creative/instructed writing model).
    Returns a full cover letter as a string.

    Parameters
    ----------
    orch             : Orchestrator
    requirements     : Output from step_extract_requirements
    resume           : Resume text
    company_summary  : Output from step_summarise_company

    Returns
    -------
    str — cover letter draft
    """
    title   = requirements.get("title", "the position")
    company = requirements.get("company", "your company")
    skills  = requirements.get("required_skills", "")

    print(f"\n[4/6] Drafting cover letter for {title} at {company}...")

    result = orch.run(Task(
        prompt=f"""Write a professional, enthusiastic cover letter for the following:
- Position  : {title}
- Company   : {company}
- Key skills they want: {skills}
- Company context: {company_summary}

Requirements:
- 3 short paragraphs (opening, fit, closing)
- Highlight relevant experience from the resume
- Sound human and specific — not generic
- End with a clear call to action
- Do not use clichés like "I am writing to express my interest"

Resume context:
{resume[:1500]}""",
        task_type=TaskType.CREATIVE,
        max_tokens=600,
        preferred_model="mistral",
    ))

    if not result.succeeded:
        return "Cover letter generation failed — please draft manually."

    print(f"  Done [{result.provider} / {result.model} — {result.tokens_used} tokens, free]")
    return result.content


def step_tailor_resume_bullets(
    orch: Orchestrator,
    requirements: dict,
    resume: str,
) -> str:
    """
    Step 5 — Suggest resume bullet point rewrites tailored to this job.

    Offloads to Mistral (best free instruction-following model).
    Returns suggested rewritten bullets as a string.

    Parameters
    ----------
    orch         : Orchestrator
    requirements : Output from step_extract_requirements
    resume       : Resume text

    Returns
    -------
    str — suggested bullet rewrites
    """
    skills = requirements.get("required_skills", "")
    title  = requirements.get("title", "the role")

    print(f"\n[5/6] Tailoring resume bullets for {title}...")

    result = orch.run(Task(
        prompt=f"""Suggest 3-5 rewritten resume bullet points that better align with this job.
For each, show:
ORIGINAL: (existing bullet)
REWRITTEN: (improved version using job's keywords)
WHY: (one sentence explaining the change)

Target job skills: {skills}
Target role: {title}""",
        task_type=TaskType.CODING,   # instruction-following task
        context=resume,
        max_tokens=600,
        preferred_model="mistral",
    ))

    if not result.succeeded:
        return "Resume tailoring unavailable."

    print(f"  Done [{result.provider} / {result.model} — {result.tokens_used} tokens, free]")
    return result.content


def step_generate_interview_questions(
    orch: Orchestrator,
    requirements: dict,
) -> str:
    """
    Step 6 — Generate likely interview questions based on the job requirements.

    Offloads to Groq (fast, factual — ideal for predictable generation tasks).
    Returns 8-10 likely interview questions.

    Parameters
    ----------
    orch         : Orchestrator
    requirements : Output from step_extract_requirements

    Returns
    -------
    str — numbered list of interview questions
    """
    title       = requirements.get("title", "the role")
    skills      = requirements.get("required_skills", "")
    resp        = requirements.get("key_responsibilities", "")

    print(f"\n[6/6] Generating interview questions for {title}...")

    result = orch.run(Task(
        prompt=f"""Generate 8-10 likely interview questions for a {title} role.
Include a mix of:
- Technical questions about: {skills}
- Behavioural questions (STAR format prompts)
- Role-specific scenario questions based on: {resp}
- One culture/values question

Format as a numbered list.""",
        task_type=TaskType.FACTUAL_QA,
        max_tokens=500,
        preferred_model="groq",
        priority=Priority.LOW,
    ))

    if not result.succeeded:
        return "Interview question generation unavailable."

    print(f"  Done [{result.provider} / {result.model} — {result.tokens_used} tokens, free]")
    return result.content


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def save_and_commit(
    git:       GitOps,
    company:   str,
    outputs:   dict,
) -> None:
    """
    Save all workflow outputs to disk and commit them to git.

    Creates one file per output type under outputs/jobseeking/<company>/<date>/.
    The git commit message includes the company name and date so `git log`
    becomes a searchable application history.

    Parameters
    ----------
    git     : GitOps instance
    company : Company name (used as folder name)
    outputs : Dict mapping filename to content string
    """
    # Sanitise company name for use as a directory name.
    safe_company = re.sub(r"[^\w\-]", "_", company.lower())[:40]
    date_str     = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    folder       = f"outputs/jobseeking/{safe_company}_{date_str}"

    print(f"\n[Git] Saving outputs to {folder}/")

    for filename, content in outputs.items():
        path = f"{folder}/{filename}"
        git._write_file(path, content)
        git._git("add", path)

    git._git(
        "commit", "-m",
        f"jobseek: {company} application — {date_str}\n\n"
        f"Generated by workflows/jobseeking.py\n"
        "All AI calls offloaded to free providers via The Brain.\n\n"
        "Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>",
    )
    print(f"[Git] Committed all outputs for {company}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse and return CLI arguments."""
    p = argparse.ArgumentParser(
        description="AI-powered job seeking workflow — all heavy lifting offloaded to free AIs",
    )
    p.add_argument("--job-file",    default=None, help="Path to job posting text file")
    p.add_argument("--resume-file", default=None, help="Path to your resume text file")
    p.add_argument("--no-commit",   action="store_true", help="Skip git commit of outputs")
    return p.parse_args()


def main() -> int:
    """
    Run the full job-seeking workflow.

    Returns
    -------
    int  exit code (0 = success)
    """
    args   = parse_args()
    orch   = Orchestrator()
    git    = GitOps(repo_path=Path(__file__).parent.parent)

    print("\n" + "=" * 60)
    print("  THE BRAIN — Job Seeking Workflow")
    print("  All heavy lifting offloaded to free AI workers")
    print("=" * 60)

    # ── Load job posting ─────────────────────────────────────────────
    if args.job_file:
        job_posting = Path(args.job_file).read_text(encoding="utf-8")
        print(f"\nLoaded job posting from: {args.job_file}")
    else:
        print("\nPaste the job posting below.")
        print("Press Enter twice then Ctrl+Z (Windows) or Ctrl+D (Mac/Linux) when done:\n")
        lines = []
        try:
            while True:
                line = input()
                lines.append(line)
        except EOFError:
            pass
        job_posting = "\n".join(lines)

    if not job_posting.strip():
        print("ERROR: No job posting provided.", file=sys.stderr)
        return 1

    # ── Load resume ───────────────────────────────────────────────────
    if args.resume_file:
        resume = Path(args.resume_file).read_text(encoding="utf-8")
        print(f"Loaded resume from: {args.resume_file}")
    else:
        resume = DEFAULT_RESUME
        print("Using default resume placeholder — pass --resume-file for better tailoring")

    # ── Run all workflow steps ────────────────────────────────────────
    requirements    = step_extract_requirements(orch, job_posting)
    company         = requirements.get("company", "unknown_company")
    fit_score       = step_score_fit(orch, requirements, resume)
    company_summary = step_summarise_company(orch, company, job_posting)
    cover_letter    = step_draft_cover_letter(orch, requirements, resume, company_summary)
    resume_bullets  = step_tailor_resume_bullets(orch, requirements, resume)
    interview_qs    = step_generate_interview_questions(orch, requirements)

    # ── Print summary ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  Results for: {company}")
    print("=" * 60)
    print("\n--- FIT SCORE ---")
    print(fit_score)

    # ── Save & commit ─────────────────────────────────────────────────
    outputs = {
        "job_requirements.txt": requirements.get("raw_extraction", job_posting),
        "fit_score.txt":        fit_score,
        "company_summary.txt":  company_summary,
        "cover_letter.md":      cover_letter,
        "resume_bullets.md":    resume_bullets,
        "interview_questions.md": interview_qs,
    }

    if not args.no_commit:
        save_and_commit(git, company, outputs)

    # ── Stats ─────────────────────────────────────────────────────────
    stats = orch.session_stats()
    print(f"\n[Stats] {stats['total_calls']} calls | "
          f"{stats['total_tokens']} tokens | "
          f"cost: ${stats['estimated_cost_usd']:.4f}")
    print("Run `python report.py` for full usage history.\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
