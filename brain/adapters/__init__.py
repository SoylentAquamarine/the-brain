"""
adapters/__init__.py — Active provider registry for The Brain.

Only truly free, no-credit-card-required providers are in the REGISTRY.
Adapter files for paid/CC providers still exist if you want to re-enable them
later — just move them from INACTIVE_ADAPTERS into ALL_ADAPTERS.
"""

# ── Active free-tier adapters ──────────────────────────────────────────────
from brain.adapters.groq_adapter        import GroqAdapter
from brain.adapters.gemini_adapter      import GeminiAdapter
from brain.adapters.mistral_adapter     import MistralAdapter
from brain.adapters.cerebras_adapter    import CerebrasAdapter
from brain.adapters.huggingface_adapter import HuggingFaceAdapter

# ── Inactive (require CC or paid balance) — import kept for future use ─────
# from brain.adapters.anthropic_adapter import AnthropicAdapter   # paid API
# from brain.adapters.openai_adapter    import OpenAIAdapter       # paid API
# from brain.adapters.cohere_adapter    import CohereAdapter       # broken key
# from brain.adapters.deepseek_adapter  import DeepSeekAdapter     # needs balance
# from brain.adapters.together_adapter  import TogetherAdapter     # requires CC

ALL_ADAPTERS = [
    GroqAdapter,
    GeminiAdapter,
    MistralAdapter,
    CerebrasAdapter,
    HuggingFaceAdapter,
]

REGISTRY: dict = {
    cls.PROVIDER_KEY: cls()
    for cls in ALL_ADAPTERS
}

__all__ = ["REGISTRY", "ALL_ADAPTERS"]
