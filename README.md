# The Brain

> Claude orchestrates. Free AI does the heavy lifting. Git remembers everything.

**The Brain** is an AI orchestration system where Claude acts as the intelligent dispatcher, routing every task to the cheapest and most capable AI provider available — then committing every result to Git so nothing is ever lost, overwritten, or untraceable.

```
You → Orchestrator (Claude) → Router
                                 ↓
              ┌──────────────────────────────────┐
              │  classification  →  Groq (FREE)  │
              │  summarisation   →  Gemini (FREE) │
              │  extraction      →  Cohere (FREE) │
              │  coding          →  OpenAI (paid) │
              │  complex reason  →  Claude (paid) │
              └──────────────────────────────────┘
                                 ↓
                         Git commit
                                 ↓
                         GitHub push → CI/CD
```

**Without Git, this breaks down.**  
Multiple AIs writing outputs with no history = chaos. With Git every AI action is reproducible, auditable, and reversible.

---

## Why this architecture

| Problem | Solution |
|---|---|
| Claude tokens are expensive | Route simple tasks to free models |
| Different AIs excel at different tasks | Smart routing table + optional Claude-assisted routing |
| AI outputs overwrite each other | Every output is a git commit with full metadata |
| Hard to debug AI pipelines | `git log` shows exactly which model ran what |
| Automation is fragile | Push to GitHub → Actions workflow triggers |

---

## Providers

| Provider | Tier | Best for | Sign up |
|---|---|---|---|
| **Anthropic Claude** | Paid | Complex reasoning, creative writing, orchestration | [console.anthropic.com](https://console.anthropic.com) |
| **OpenAI GPT-4o-mini** | Paid | Coding, instruction-following, structured output | [platform.openai.com](https://platform.openai.com) |
| **Google Gemini 1.5 Flash** | **FREE** ✓ | Long documents (1M token context), summarisation | [aistudio.google.com](https://aistudio.google.com/app/apikey) |
| **Groq / Llama 3** | **FREE** ✓ | Ultra-fast classification, short Q&A (400+ tok/s) | [console.groq.com](https://console.groq.com) |
| **Cohere Command-R** | **FREE** ✓ | Extraction, classification, structured tasks | [dashboard.cohere.com](https://dashboard.cohere.com) |

The system **automatically skips** any provider whose API key is missing — so you can start with just the providers you have.

---

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/the-brain.git
cd the-brain
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env and paste in your keys — see config/models.yaml for sign-up links
```

### 3. Run

```bash
# Interactive
python main.py "Explain quantum entanglement in one paragraph"

# Force a specific provider
python main.py --provider groq "What is 2 + 2?"

# Summarise a long document and commit the output to git
python main.py --type summarization --commit "$(cat my_document.txt)"

# Commit AND push to GitHub
python main.py --type coding --commit --push "Write a Python merge sort"

# Show which providers are available
python main.py --status

# Full demo (5 tasks across 5 providers)
python examples/demo.py
```

### 4. Run tests

```bash
pytest tests/ -v
```

---

## Project structure

```
the-brain/
├── brain/
│   ├── __init__.py            # Public API: Orchestrator, Task, TaskType
│   ├── orchestrator.py        # Central dispatcher + fallback logic
│   ├── router.py              # Static + dynamic routing table
│   ├── task.py                # Task / TaskResult dataclasses
│   ├── git_ops.py             # Git integration — write, commit, push
│   └── adapters/
│       ├── __init__.py        # Auto-discovery registry
│       ├── base.py            # Abstract base adapter
│       ├── anthropic_adapter.py
│       ├── openai_adapter.py
│       ├── gemini_adapter.py
│       ├── groq_adapter.py
│       └── cohere_adapter.py
├── config/
│   └── models.yaml            # Model reference — context limits, costs, sign-up links
├── examples/
│   └── demo.py                # Live 5-provider demonstration
├── tests/
│   └── test_adapters.py       # Unit tests (no API keys required)
├── outputs/                   # Git-committed AI outputs land here
├── .env.example               # Copy to .env and fill in keys
├── main.py                    # CLI entry point
└── requirements.txt
```

---

## How routing works

**Static routing** (default) uses a priority table — no extra tokens spent:

```python
FACTUAL_QA   → groq → gemini → cohere → openai → anthropic
SUMMARIZATION→ gemini → groq → cohere → openai → anthropic
CODING       → openai → anthropic → groq → gemini → cohere
REASONING    → anthropic → openai → gemini → groq → cohere
```

The first provider in the list that has a valid API key and is available wins.

**Dynamic routing** (opt-in, `--dynamic` flag or `BRAIN_DYNAMIC_ROUTING=1`) sends a 300-character snippet of your task to Claude, which overrides the static choice when the content warrants it.  Costs ~100 tokens per routing decision.

**Automatic fallback**: if the chosen provider returns an error, the orchestrator automatically tries the next available provider — transparent to the caller.

---

## Git as the control layer

Every `--commit` call produces a structured commit:

```
[brain] groq/llama3-8b-8192 — classification (42 tokens, 318ms)

Task ID  : 3f8a1b2c-...
Time     : 2025-04-17T14:23:01Z
Prompt   : Classify the sentiment of...
Provider : groq
Model    : llama3-8b-8192
Tokens   : 42
Latency  : 318ms
Cost     : free
```

This gives you:
- **`git log`** — full AI action history
- **`git diff`** — exactly what each model changed
- **`git revert`** — undo any AI output cleanly
- **GitHub Actions** — push triggers your CI/CD pipeline automatically

---

## Adding a new provider

1. Create `brain/adapters/your_provider_adapter.py` subclassing `BaseAdapter`
2. Set `PROVIDER_KEY`, `TIER`, `SUPPORTED_TASK_TYPES`
3. Implement `is_available()` and `complete()`
4. Register it in `brain/adapters/__init__.py`
5. Add a routing preference to `brain/router.py`

That's it — the orchestrator picks it up automatically.

---

## Environment variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | — | Claude orchestrator |
| `OPENAI_API_KEY` | No | — | GPT-4o-mini worker |
| `GEMINI_API_KEY` | No | — | Gemini 1.5 Flash worker |
| `GROQ_API_KEY` | No | — | Llama 3 on Groq worker |
| `COHERE_API_KEY` | No | — | Command-R worker |
| `BRAIN_DYNAMIC_ROUTING` | No | `0` | Set to `1` for Claude-assisted routing |
| `LOG_LEVEL` | No | `INFO` | Python logging level |
| `ANTHROPIC_MODEL` | No | `claude-sonnet-4-6` | Override Claude model |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | Override OpenAI model |
| `GEMINI_MODEL` | No | `gemini-1.5-flash` | Override Gemini model |
| `GROQ_MODEL` | No | `llama3-8b-8192` | Override Groq model |
| `COHERE_MODEL` | No | `command-r` | Override Cohere model |

---

<!-- BRAIN_STATS_START -->
*Last updated: 2026-04-17 17:58 UTC — auto-generated by `update_readme_stats.py`*

## Live Usage Stats

| Provider | Tier | Calls | Tokens | Avg Latency | Cost |
|---|---|---|---|---|---|
| **openai** | PAID | 1 | 89 | 2780ms | $0.0000 |
| **cerebras** | FREE | 1 | 84 | 223ms | free |
| **gemini** | FREE | 1 | 36 | 515ms | free |
| **deepseek** | FREE | 1 | 0 | 1007ms | free |

### Token Savings

| Metric | Value |
|---|---|
| Total calls | 4 |
| Calls handled by free workers | 4 |
| Tokens offloaded from Claude | 209 |
| % of tokens saved | 100.0% |
| Estimated savings (Claude Sonnet rate) | $0.0006 |
| Total spend on paid APIs | $0.0000 |

### Token Distribution

```
openai       [PAID]  █████████████████████                               42.6%
cerebras     [FREE]  ████████████████████                                40.2%
gemini       [FREE]  ████████                                            17.2%
deepseek     [FREE]                                                       0.0%
```
<!-- BRAIN_STATS_END -->

---

## License

MIT — build on it, fork it, ship it.
