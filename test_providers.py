"""
test_providers.py — Tests every ManyAI provider and model with "What is 2+2?"
Reports pass/fail and latency for each. Use results to update providers.ts.
"""

import os, time, json, requests
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

TIMEOUT = 20

PROVIDERS = {
    "cerebras": {
        "base_url": "https://api.cerebras.ai/v1",
        "key": os.getenv("CEREBRAS_API_KEY"),
        "type": "openai",
        "models": ["llama3.1-8b", "llama3.1-70b"],
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key": os.getenv("GROQ_API_KEY"),
        "type": "openai",
        "models": ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "deepseek-r1-distill-llama-70b"],
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "key": os.getenv("GEMINI_API_KEY"),
        "type": "gemini",
        "models": ["gemini-2.5-flash-lite", "gemini-2.5-flash"],
    },
    "mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "key": os.getenv("MISTRAL_API_KEY"),
        "type": "openai",
        "models": ["mistral-small-latest", "mistral-large-latest"],
    },
    "sambanova": {
        "base_url": "https://api.sambanova.ai/v1",
        "key": os.getenv("SAMBANOVA_API_KEY"),
        "type": "openai",
        "models": ["Meta-Llama-3.3-70B-Instruct", "Meta-Llama-3.1-405B-Instruct", "DeepSeek-R1"],
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key": os.getenv("OPENROUTER_API_KEY"),
        "type": "openai",
        "extra_headers": {"HTTP-Referer": "https://stevepleasants.com/manyai", "X-Title": "ManyAI"},
        "models": [
            "meta-llama/llama-3.2-3b-instruct:free",
            "google/gemma-3-27b-it:free",
            "mistralai/mistral-7b-instruct:free",
            "deepseek/deepseek-chat:free",
            "meta-llama/llama-3.3-70b-instruct",
        ],
    },
    "huggingface": {
        "base_url": "https://api-inference.huggingface.co/v1",
        "key": os.getenv("HUGGINGFACE_API_KEY"),
        "type": "openai",
        "models": ["meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-72B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3"],
    },
    "cohere": {
        "base_url": "https://api.cohere.com/compatibility/v1",
        "key": os.getenv("COHERE_API_KEY"),
        "type": "openai",
        "models": ["command-r", "command-r-plus", "command-a-03-2025"],
    },
    "fireworks": {
        "base_url": "https://api.fireworks.ai/inference/v1",
        "key": os.getenv("FIREWORKS_API_KEY"),
        "type": "openai",
        "models": [
            "accounts/fireworks/models/deepseek-v3p1",
            "accounts/fireworks/models/llama-v3p3-70b-instruct",
            "accounts/fireworks/models/qwen2p5-72b-instruct",
        ],
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "key": os.getenv("OPENAI_API_KEY"),
        "type": "openai",
        "models": ["gpt-4o-mini", "gpt-4o"],
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com/v1",
        "key": os.getenv("ANTHROPIC_API_KEY"),
        "type": "anthropic",
        "models": ["claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022", "claude-3-opus-20240229"],
    },
    "pollinations": {
        "base_url": "https://text.pollinations.ai",
        "key": None,
        "type": "pollinations",
        "models": ["openai", "mistral", "llama"],
    },
}

PROMPT = "What is 2+2? Reply with only the number."

def test_model(provider_key, provider, model):
    key = provider.get("key")
    base_url = provider["base_url"]
    ptype = provider["type"]
    extra_headers = provider.get("extra_headers", {})

    start = time.time()
    try:
        if ptype == "pollinations":
            url = f"{base_url}/{requests.utils.quote(PROMPT)}"
            r = requests.get(url, timeout=TIMEOUT)
            r.raise_for_status()
            content = r.text.strip()[:50]

        elif ptype == "gemini":
            url = f"{base_url}/models/{model}:generateContent?key={key}"
            body = {"contents": [{"role": "user", "parts": [{"text": PROMPT}]}]}
            r = requests.post(url, json=body, timeout=TIMEOUT)
            r.raise_for_status()
            content = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()[:50]

        elif ptype == "anthropic":
            headers = {
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            }
            body = {"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": PROMPT}]}
            r = requests.post(f"{base_url}/messages", headers=headers, json=body, timeout=TIMEOUT)
            r.raise_for_status()
            content = r.json()["content"][0]["text"].strip()[:50]

        else:  # openai-compatible
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json", **extra_headers}
            body = {"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": PROMPT}]}
            r = requests.post(f"{base_url}/chat/completions", headers=headers, json=body, timeout=TIMEOUT)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"].strip()[:50]

        ms = int((time.time() - start) * 1000)
        return (provider_key, model, "PASS", ms, content)

    except Exception as e:
        ms = int((time.time() - start) * 1000)
        msg = str(e)[:80]
        return (provider_key, model, "FAIL", ms, msg)


# ── Run all tests concurrently ────────────────────────────────────────────────

tasks = []
for pk, pdata in PROVIDERS.items():
    if not pdata.get("key") and pdata["type"] not in ("pollinations",):
        print(f"[SKIP] {pk} - no API key")
        continue
    for model in pdata["models"]:
        tasks.append((pk, pdata, model))

print(f"\nTesting {len(tasks)} provider/model combinations...\n")

results = []
with ThreadPoolExecutor(max_workers=10) as ex:
    futures = {ex.submit(test_model, pk, pdata, model): (pk, model) for pk, pdata, model in tasks}
    for future in as_completed(futures):
        result = future.result()
        results.append(result)
        pk, model, status, ms, content = result
        print(f"[{status}] {pk:15} {model:55} {ms:5}ms  {content}")

# ── Summary ───────────────────────────────────────────────────────────────────

print("\n" + "="*80)
passed = [r for r in results if r[2] == "PASS"]
failed = [r for r in results if r[2] == "FAIL"]
print(f"\nPassed: {len(passed)}   Failed: {len(failed)}\n")

if failed:
    print("Failed models:")
    for pk, model, _, ms, err in failed:
        print(f"  {pk} / {model}: {err}")

# Save results to JSON for reference
out = {"passed": [], "failed": []}
for pk, model, status, ms, content in results:
    entry = {"provider": pk, "model": model, "latency_ms": ms, "response": content}
    out["passed" if status == "PASS" else "failed"].append(entry)

with open("test_results.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nFull results saved to test_results.json")
