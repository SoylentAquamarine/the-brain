# Integrating with The Brain

Read this when setting up a new Claude Code window that wants to offload AI
calls to The Brain instead of burning Claude tokens.

## What The Brain does

The Brain is an AI orchestration layer at `SoylentAquamarine/the-brain`.
It routes prompts to free AI workers (Groq, Gemini, Mistral, Cerebras, etc.)
and logs every call to `stats/usage.json` for the nightly report.

You call it with one command. It prints the result. You read it and carry on.

---

## Setup for a new window

1. Clone the repo (if not already local):
   ```bash
   git clone https://github.com/SoylentAquamarine/the-brain C:\Claude\git\the-brain
   ```

2. Install dependencies:
   ```bash
   cd C:\Claude\git\the-brain
   pip install -r requirements.txt
   ```

3. Confirm `.env` exists in `C:\Claude\git\the-brain\` with API keys.
   The keys needed are: `GROQ_API_KEY`, `GEMINI_API_KEY`, `MISTRAL_API_KEY`,
   `CEREBRAS_API_KEY`. Pollinations and HuggingFace need no key.

---

## Calling delegate.py

```bash
python C:\Claude\git\the-brain\delegate.py \
  --provider <provider> \
  --type <task_type> \
  --prompt "<your prompt>"
```

The output is two parts:
- **Line 1:** metadata header — `[provider / model | tokens | latency | cost]`
- **Rest:** the actual response content

### Provider routing guide

| Task | Provider | Why |
|---|---|---|
| Classify, score, yes/no | `cerebras` | Fastest (~1500 tok/s) |
| Quick factual Q&A | `groq` | Fast, reliable |
| Summarise long text | `gemini` | 1M token context |
| Code generation | `mistral` | Best free coding model |
| Creative writing / drafting | `mistral` | Best quality output |
| Translation | `gemini` | Strongest multilingual |
| Image generation | `pollinations` | No key needed |
| General fallback | `groq` | Fast, always available |

### Task types for --type

`classification` `summarization` `coding` `creative` `reasoning`
`factual_qa` `extraction` `translation` `general`

### Examples

```bash
# Extract requirements from a job description
python C:\Claude\git\the-brain\delegate.py \
  --provider mistral --type extraction \
  --prompt "Extract the key requirements from this job posting: ..."

# Score resume fit
python C:\Claude\git\the-brain\delegate.py \
  --provider cerebras --type classification \
  --prompt "Score the fit 0-100 between this resume and job description. Resume: ... Job: ..."

# Draft a cover letter
python C:\Claude\git\the-brain\delegate.py \
  --provider mistral --type creative \
  --prompt "Write a cover letter for this role. Resume: ... Job: ..."

# Summarise a company
python C:\Claude\git\the-brain\delegate.py \
  --provider gemini --type summarization \
  --prompt "Summarise this company's mission, culture, and what they do: ..."

# Generate interview questions
python C:\Claude\git\the-brain\delegate.py \
  --provider groq --type factual_qa \
  --prompt "Generate 10 likely interview questions for this role: ..."
```

---

## Stats and the nightly report

Every `delegate.py` call automatically writes to
`C:\Claude\git\the-brain\stats\usage.json`.

The nightly GitHub Actions workflow (`nightly-stats.yml`) runs at midnight UTC,
reads that file, and updates the README with cumulative usage stats.

**For stats to appear in the nightly report, push usage.json after each session:**

```bash
cd C:\Claude\git\the-brain
git add stats/usage.json
git diff --staged --quiet || git commit -m "chore: session stats $(date -u +%Y-%m-%d)"
git push
```

Add this block at the end of any workflow script that calls `delegate.py`.

---

## Division of labour

| Responsibility | Handled by |
|---|---|
| File I/O (read resumes, write outputs) | Your workflow window |
| Workflow orchestration and decisions | Claude (your window) |
| AI-heavy tasks (write, score, extract) | The Brain via `delegate.py` |
| Stats logging | Automatic — `delegate.py` does it |
| Stats publishing | Your workflow — push `usage.json` at end |
| Nightly report | The Brain's GitHub Actions — midnight UTC |

---

## What stays in your window (do NOT offload)

- Reading and writing local files
- Deciding what steps to run
- Anything conversational or planning-based
- Short tasks answerable in a sentence or two

---

## Checking available providers

```bash
python C:\Claude\git\the-brain\delegate.py --provider groq --type factual_qa --prompt "ping"
```

If a provider is unavailable (missing key), it exits with code 1 and prints
`ERROR: Provider '...' is not available`.

---

## Report and stats commands

```bash
# Human-readable usage report
python C:\Claude\git\the-brain\report.py

# JSON output (for scripting)
python C:\Claude\git\the-brain\report.py --json

# Update README with latest stats and push to GitHub
python C:\Claude\git\the-brain\update_readme_stats.py --push
```
