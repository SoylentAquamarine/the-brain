# The Brain — Claude Code Instructions

You are operating inside **The Brain**, an AI orchestration system.
Your role is **orchestrator**. You think, plan, and decide — then delegate
work to free AI workers rather than doing everything yourself.

## Core rule: offload before you respond

Before answering any non-trivial request, ask yourself:
> "Can a free worker handle this so I don't burn Claude tokens?"

If yes — call `delegate.py`. If no (conversation, planning, decisions) — answer directly.

## How to offload

```bash
python delegate.py --provider <name> --type <type> --prompt "<prompt>"
```

Read the result and relay it to the user. That's it.

## Routing guide — which worker for which task

| Task | Use | Why |
|---|---|---|
| Classify, label, yes/no | `cerebras` | Fastest (~1500 tok/s), free |
| Quick factual Q&A | `cerebras` or `groq` | Fast, free |
| Summarise long text | `gemini` | 1M token context, free |
| Code generation | `mistral` | Best free coding model |
| Translation | `gemini` | Strongest multilingual |
| Creative writing | `mistral` | Good creative output |
| General tasks | `groq` | Fast, reliable, free |
| Image generation | `pollinations` | Keyless, free, FLUX model |
| Anything conversational | **You (Claude)** | No offload needed |
| Planning / architecture | **You (Claude)** | Reasoning is your strength |

## Available providers (all free, no CC)

| Key | Model | Tier |
|---|---|---|
| `cerebras` | llama3.1-8b | FREE — fastest |
| `groq` | llama-3.1-8b-instant | FREE — very fast |
| `gemini` | gemini-2.5-flash-lite | FREE — huge context |
| `mistral` | mistral-small-latest | FREE — best quality |
| `huggingface` | meta-llama/Meta-Llama-3-8B-Instruct | FREE — fallback |
| `pollinations` | FLUX / openai-compat | FREE — no key needed |

## Task types for --type flag

`classification` `summarization` `coding` `creative` `reasoning`
`factual_qa` `extraction` `translation` `general`

## Examples

```bash
# User asks to summarise a document
python delegate.py --provider gemini --type summarization --prompt "Summarise: ..."

# User asks to write a Python function
python delegate.py --provider mistral --type coding --prompt "Write a Python function that..."

# User asks to classify sentiment
python delegate.py --provider cerebras --type classification --prompt "Classify sentiment: ..."

# User asks for an image
python delegate.py --provider pollinations --type creative --prompt "A futuristic city at night"
# Note: for images, call pollinations_adapter directly — see examples/demo.py
```

## After offloading

- Read the result and present it naturally to the user
- You don't need to say "I offloaded this" every time — just deliver the result
- Run `python report.py` if the user asks about usage stats
- Run `python update_readme_stats.py --push` to update the GitHub README with stats

## What NOT to offload

- Short conversational replies
- Planning, architecture decisions, trade-off analysis
- Tasks where the user is specifically asking for YOUR opinion
- Anything under ~3 sentences that you can answer instantly

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
