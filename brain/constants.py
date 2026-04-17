"""
constants.py — Central home for every magic number and shared constant.

Keeping literals here means a single edit changes behaviour everywhere.
Never scatter raw numbers or strings through business logic files.
"""

# ---------------------------------------------------------------------------
# Token / cost defaults
# ---------------------------------------------------------------------------

# Default maximum tokens to request from any provider when the caller
# does not specify one.  1024 is large enough for most responses but
# small enough to avoid runaway bills on paid providers.
DEFAULT_MAX_TOKENS: int = 1024

# Token ceiling used for routing-only calls (e.g. dynamic routing prompt).
# We only need one word back, so keeping this tiny saves cost.
ROUTING_MAX_TOKENS: int = 20

# Claude Sonnet blended cost per 1 000 tokens (USD).
# Used to calculate "estimated savings" when free workers handle tasks
# that would otherwise have been sent to Claude.
# Update this if Anthropic changes pricing.
CLAUDE_COST_PER_1K_TOKENS: float = 0.003

# ---------------------------------------------------------------------------
# Routing prompt snippet length
# ---------------------------------------------------------------------------

# How many characters of the user prompt to include in the dynamic routing
# call to Claude.  Longer = more accurate routing, more tokens spent.
ROUTING_PROMPT_SNIPPET_LEN: int = 300

# ---------------------------------------------------------------------------
# Orchestrator defaults
# ---------------------------------------------------------------------------

# How many provider fallbacks to attempt before giving up on a task.
# e.g. 3 means: try primary → fallback 1 → fallback 2 → fail.
DEFAULT_MAX_FALLBACKS: int = 3

# ---------------------------------------------------------------------------
# Report / display
# ---------------------------------------------------------------------------

# Width of the separator line in report.py output.
REPORT_LINE_WIDTH: int = 60

# Each '#' in the distribution bar represents this many percent.
# Lower = finer resolution but longer bars.
REPORT_BAR_SCALE: int = 2

# ---------------------------------------------------------------------------
# Stats file
# ---------------------------------------------------------------------------

# Path relative to the project root where usage stats are persisted.
STATS_FILE_PATH: str = "stats/usage.json"

# ---------------------------------------------------------------------------
# Git
# ---------------------------------------------------------------------------

# Default remote and branch used by GitOps.push().
GIT_DEFAULT_REMOTE: str = "origin"
GIT_DEFAULT_BRANCH: str = "master"
