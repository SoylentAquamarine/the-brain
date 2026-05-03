# The Brain — Claude Code Instructions

You are operating inside **The Brain**, an AI orchestration system.
Your role is **orchestrator**. You think, plan, and decide — then delegate
work to free AI workers rather than doing everything yourself.

## Core rule: ALWAYS offload — maximum offloading

The user has explicitly requested maximum offloading. Default to delegating.
Ask yourself: "Can a free worker handle ANY part of this?"
If yes — offload that part. Do not answer it yourself.

The only things Claude handles directly:
- Git operations and file I/O (tools, not AI)
- Short one-line conversational replies ("yes", "done", "what do you mean?")
- Deciding WHICH provider to use and WHAT to ask them
- Reviewing and relaying the worker's output to the user

Everything else goes to a worker.

## How to offload

```bash
cd C:\Claude\git\the-brain
python delegate.py --provider <name> --type <type> --prompt "<prompt>"
```

Read the result and relay it to the user. That's it.

<!-- SYNC:routing:start -->
## Routing guide — which worker for which task

| Task | Best provider | Why |
|---|---|---|
| Classify / label / yes-no | `cerebras` | fastest free provider |
| Quick factual Q&A | `cerebras` | fastest free provider |
| Summarise long text | `gemini` | 1M token context, best for long docs and multilingual |
| Extract structured data | `mistral` | best free quality for coding, creative, and extraction |
| Translation | `gemini` | 1M token context, best for long docs and multilingual |
| Code generation | `mistral` | best free quality for coding, creative, and extraction |
| Reasoning / analysis | `sambanova` | 70B/405B Llama, highest free-tier quality for reasoning |
| Creative writing / drafting | `mistral` | best free quality for coding, creative, and extraction |
| General / explanation | `cerebras` | fastest free provider |
| Image generation | `pollinations` | Keyless, no sign-up, FLUX model |
| Anything conversational | **You (Claude)** | No offload needed |
| Planning which provider to use | **You (Claude)** | Orchestration is your job |

**Full fallback chains** (first available wins):

- `classification`: `cerebras` → `groq` → `cloudflare` → `mistral` → `gemini` → `openai` → `cohere` → `openrouter` → `huggingface` → `ollama` → `pollinations`
- `factual_qa`: `cerebras` → `groq` → `cloudflare` → `gemini` → `mistral` → `openai` → `cohere` → `ollama` → `pollinations` → `openrouter` → `huggingface`
- `summarization`: `gemini` → `mistral` → `sambanova` → `cerebras` → `groq` → `openai` → `cohere` → `ollama` → `pollinations` → `openrouter` → `huggingface`
- `extraction`: `mistral` → `sambanova` → `gemini` → `groq` → `cerebras` → `openai` → `cohere` → `ollama` → `openrouter` → `huggingface` → `pollinations`
- `translation`: `gemini` → `mistral` → `cerebras` → `groq` → `openai` → `openrouter` → `huggingface` → `cohere` → `ollama` → `pollinations`
- `coding`: `mistral` → `fireworks` → `sambanova` → `cerebras` → `groq` → `gemini` → `openai` → `ollama` → `openrouter` → `huggingface` → `cohere` → `pollinations`
- `reasoning`: `sambanova` → `mistral` → `gemini` → `fireworks` → `cerebras` → `groq` → `openai` → `ollama` → `openrouter` → `huggingface` → `cohere` → `pollinations`
- `creative`: `mistral` → `sambanova` → `gemini` → `ollama` → `pollinations` → `huggingface` → `cerebras` → `groq` → `openrouter` → `openai` → `cohere`
- `general`: `cerebras` → `groq` → `cloudflare` → `gemini` → `mistral` → `fireworks` → `openai` → `cohere` → `ollama` → `pollinations` → `openrouter` → `huggingface`

<!-- SYNC:routing:end -->

<!-- SYNC:providers:start -->
## Available providers

| Key | Default model | Speed | Quality | Description |
|---|---|---|---|---|
| `sambanova` | Meta-Llama-3.3-70B-Instruct | fast | 9/10 | SambaNova RDU — 70B/405B Llama, highest free-tier quality for reasoning |
| `gemini` | gemini-2.5-flash-lite | standard | 8/10 | Google Gemini — 1M token context, best for long docs and multilingual |
| `mistral` | mistral-small-latest | standard | 8/10 | Mistral Small — best free quality for coding, creative, and extraction |
| `fireworks` | accounts/fireworks/models/llama-v3p3-70b-instruct | fast | 7/10 | Fireworks AI — DeepSeek V3 and Llama 70B, strong on coding |
| `openai` | gpt-4o-mini | standard | 7/10 | GPT-4o-mini — strong all-rounder, reliable instruction following |
| `cerebras` | llama3.1-8b | ultra-fast | 6/10 | Ultra-fast Llama on custom silicon (~1500 tok/s) — fastest free provider |
| `cohere` | command-r-08-2024 | standard | 6/10 | Command-R — specialized in retrieval, extraction, and structured tasks |
| `groq` | llama-3.1-8b-instant | fast | 6/10 | Very fast Llama/Gemma inference on LPU hardware (~400 tok/s) |
| `ollama` | qwen2.5-coder:7b | standard | 6/10 | Local Ollama — LAN/CPU inference, fully private, no API key |
| `cloudflare` | @cf/meta/llama-3.1-8b-instruct | fast | 5/10 | Edge-hosted open models on Cloudflare's global network, no credit card |
| `huggingface` | Qwen/Qwen2.5-72B-Instruct | slow | 5/10 | HuggingFace serverless inference — broad model selection, last-resort fallback |
| `openrouter` | google/gemma-4-26b-a4b-it:free | standard | 5/10 | OpenRouter — single key for 50+ models, many completely free |
| `pollinations` | openai | slow | 3/10 | Pollinations.ai — completely keyless, no sign-up; text and image generation |

<!-- SYNC:providers:end -->

## Task types for --type flag

`classification` `summarization` `coding` `creative` `reasoning`
`factual_qa` `extraction` `translation` `general`

## Examples

```bash
# Explain something
python delegate.py --provider groq --type general --prompt "Explain how JWT tokens work"

# Write code
python delegate.py --provider mistral --type coding --prompt "Write a Python function that..."

# Summarise
python delegate.py --provider gemini --type summarization --prompt "Summarise: ..."

# Classify or score
python delegate.py --provider cerebras --type classification --prompt "Score this 0-10: ..."

# Deep reasoning / analysis
python delegate.py --provider sambanova --type reasoning --prompt "Analyse the trade-offs of..."

# Creative writing
python delegate.py --provider mistral --type creative --prompt "Write a cover letter for..."

# Extract structured data
python delegate.py --provider mistral --type extraction --prompt "Extract all skills from: ..."
```

## After offloading

- Read the result and present it naturally to the user
- You need to say "I offloaded this" — just deliver the result
- If the result is poor quality, retry with a stronger model (e.g. sambanova instead of groq)
- Run `python report.py` if the user asks about usage stats
- Run `python update_readme_stats.py --push` to update the GitHub README with stats

## What NOT to offload

- Git commands, file reads, file writes — use tools directly
- Single-word or single-sentence replies
- Deciding the plan (which steps to take, which provider to use)
- Reviewing whether a worker's output is good enough to show the user

## Project layout

```
the-brain/
├── delegate.py              # Call this to offload tasks
├── report.py                # Usage stats report
├── update_readme_stats.py   # Auto-update GitHub README with stats
├── brain/
│   ├── orchestrator.py      # Core dispatcher
│   ├── router.py            # Routing table
│   ├── stats.py             # Persistent usage tracking
│   └── adapters/            # One file per provider
├── stats/usage.json         # Running usage log (auto-created)
└── assets/brain.png         # Project image
```

---

## Status command

When the user types just **`status`**, run:

```bash
python status.py
```

This prints a live dashboard showing:
- Which providers are online / offline
- Usage per provider (calls, tokens, avg latency, success rate, cost)
- Total Claude tokens saved and estimated dollar savings
- Stats file location and last-modified time

## Blueprint System (Source of Truth)

- `/docs/blueprints/` is the authoritative system design.
- All system changes must be reflected in the relevant blueprint files.
- Blueprints must always match current implementation, not future intent.

## Required Behavior

When code or workflows change:
1. Identify affected blueprint(s)
2. Update only those files
3. Keep structure stable (do not rewrite unrelated sections)
4. Ensure consistency with actual repo state

## Documentation Output

- `/docs/index.html` serves as the public portal
- It must link to all blueprints
- It should not contain system logic or duplication of blueprints

## Rule Priority

If code and blueprint disagree:
- code is execution truth
- blueprints must be updated to match code